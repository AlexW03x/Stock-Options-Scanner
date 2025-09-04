###calculator logic for the volatility measurements stored in here###

import yfinance as yf
import numpy as np
import logging
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple, Any, Callable
import concurrent.futures
from functools import lru_cache
import pandas as pd

# --- Configuration Constants ---
MIN_AVG_VOLUME = 1_500_000
MIN_IV_RV_RATIO = 1.25
MAX_TS_SLOPE = -0.00406
FUTURE_DATE_CUTOFF_DAYS = 45
HISTORICAL_DATA_PERIOD = '3mo'
VOLATILITY_WINDOW = 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@lru_cache(maxsize=256)
def get_ticker(symbol: str) -> yf.Ticker:
    """Gets a cached yfinance.Ticker object."""
    return yf.Ticker(symbol)

def filter_dates(dates: List[str]) -> List[str]:
    """Filters expiration dates to those within the cutoff period."""
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=FUTURE_DATE_CUTOFF_DAYS)
    sorted_dates = sorted([datetime.strptime(date, "%Y-%m-%d").date() for date in dates])
    future_dates = [d for d in sorted_dates if d >= cutoff_date]
    if not future_dates:
        raise ValueError(f"No option expiration date found {FUTURE_DATE_CUTOFF_DAYS}+ days in the future.")
    first_future_date = future_dates[0]
    cutoff_index = sorted_dates.index(first_future_date)
    return [d.strftime("%Y-%m-%d") for d in sorted_dates[:cutoff_index + 1]]

def yang_zhang(price_data: pd.DataFrame, window: int = VOLATILITY_WINDOW, trading_periods: int = 252) -> float:
    """Calculates the Yang-Zhang historical volatility."""
    log_ho = np.log(price_data['High'] / price_data['Open'])
    log_lo = np.log(price_data['Low'] / price_data['Open'])
    log_co = np.log(price_data['Close'] / price_data['Open'])
    log_oc = np.log(price_data['Open'] / price_data['Close'].shift(1))
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = (log_oc**2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    result = np.sqrt(close_vol + k * open_vol + (1 - k) * rs) * np.sqrt(trading_periods)
    return result.iloc[-1]

def build_term_structure(days: List[int], ivs: List[float]) -> Callable[[int], float]:
    """Builds a spline for interpolating implied volatility."""
    days_np, ivs_np = np.array(days), np.array(ivs)
    sort_idx = days_np.argsort()
    days_sorted, ivs_sorted = days_np[sort_idx], ivs_np[sort_idx]
    spline = interp1d(days_sorted, ivs_sorted, kind='linear', fill_value="extrapolate")
    return lambda dte: float(spline(dte))

def fetch_chain(ticker_obj: yf.Ticker, date: str) -> Tuple[str, Any]:
    """Helper function to fetch a single option chain."""
    return date, ticker_obj.option_chain(date)

def get_recommendation_text(result: Dict[str, bool]) -> str:
    """Determines recommendation string based on boolean checks."""
    if result.get('avg_volume') and result.get('iv30_rv30') and result.get('ts_slope_0_45'):
        return "Recommend"
    elif result.get('ts_slope_0_45') and (result.get('avg_volume') or result.get('iv30_rv30')):
        return "Consider"
    else:
        return "Avoid"

def process_stock(ticker_symbol: str) -> Dict[str, Any]:
    """
    Main logic function to fetch all data for a single stock and return a result dictionary.
    Handles errors gracefully for a single ticker.
    """
    try:
        ticker_symbol = ticker_symbol.strip().upper()
        if not ticker_symbol:
            return {"symbol": ticker_symbol, "error": "No symbol provided."}
        
        stock = get_ticker(ticker_symbol)
        
        # --- Fetch primary data concurrently ---
        info, calendar = {}, {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_info = executor.submit(lambda: stock.info)
            future_calendar = executor.submit(lambda: stock.calendar)
            info = future_info.result()
            calendar = future_calendar.result()
            
        price_history = stock.history(period=HISTORICAL_DATA_PERIOD)
        if price_history.empty:
            return {"symbol": ticker_symbol, "error": "Could not retrieve price history."}

        underlying_price = price_history['Close'].iloc[-1]
        exp_dates = list(stock.options)
        if not exp_dates:
            return {"symbol": ticker_symbol, "error": "No options contracts found."}

        try:
            filtered_exp_dates = filter_dates(exp_dates)
        except ValueError as e:
            return {"symbol": ticker_symbol, "error": str(e)}

        options_chains = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_date = {executor.submit(fetch_chain, stock, date): date for date in filtered_exp_dates}
            for future in concurrent.futures.as_completed(future_to_date):
                try:
                    d, chain = future.result()
                    options_chains[d] = chain
                except Exception as e:
                    logging.warning(f"Could not fetch option chain for {future_to_date[future]}: {e}")
        
        if not options_chains:
            return {"symbol": ticker_symbol, "error": "Failed to retrieve option chain data."}

        atm_iv, straddle_price = {}, None
        today = datetime.today().date()
        for i, exp_date in enumerate(sorted(options_chains.keys())):
            chain = options_chains[exp_date]
            calls, puts = chain.calls, chain.puts
            if calls.empty or puts.empty: continue

            atm_call = calls.iloc[(calls['strike'] - underlying_price).abs().idxmin()]
            atm_put = puts.iloc[(puts['strike'] - underlying_price).abs().idxmin()]
            avg_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2.0
            dte = (datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days
            atm_iv[dte] = avg_iv
            
            if i == 0:
                call_mid = (atm_call['bid'] + atm_call['ask']) / 2.0
                put_mid = (atm_put['bid'] + atm_put['ask']) / 2.0
                if call_mid > 0 and put_mid > 0:
                    straddle_price = call_mid + put_mid

        if not atm_iv:
            return {"symbol": ticker_symbol, "error": "Could not determine ATM IV."}

        dtes, ivs = list(atm_iv.keys()), list(atm_iv.values())
        term_spline = build_term_structure(dtes, ivs)
        ts_slope_0_45 = (term_spline(45) - term_spline(min(dtes))) / (45 - min(dtes)) if (45 - min(dtes)) != 0 else 0
        
        realized_volatility = yang_zhang(price_history)
        iv30_rv30 = term_spline(30) / realized_volatility if realized_volatility > 0 else float('inf')
        avg_volume = price_history['Volume'].rolling(30).mean().iloc[-1]
        
        forecasted_move_pct = (straddle_price / underlying_price * 100) if straddle_price else 0.0

        # --- Final result compilation ---
        check_results = {
            'avg_volume': avg_volume >= MIN_AVG_VOLUME,
            'iv30_rv30': iv30_rv30 >= MIN_IV_RV_RATIO,
            'ts_slope_0_45': ts_slope_0_45 <= MAX_TS_SLOPE,
        }
        
        earnings_date = calendar.get('Earnings Date', [None])[0]
        
        return {
            "symbol": ticker_symbol,
            "name": info.get('longName', ticker_symbol),
            "logo_url": info.get('logo_url', 'https://via.placeholder.com/40'),
            "recommendation": get_recommendation_text(check_results),
            "volatility": f"{iv30_rv30:.2f}",
            "earnings_date_raw": earnings_date.strftime('%Y-%m-%d') if earnings_date else None,
            "earnings_date_str": earnings_date.strftime('%b %d, %Y') if earnings_date else "N/A",
            "forecasted_move": forecasted_move_pct,
            "error": None
        }

    except Exception as e:
        logging.error(f"An unexpected error occurred for {ticker_symbol}: {e}")
        return {"symbol": ticker_symbol, "error": f"A processing error occurred: {str(e)}"}
