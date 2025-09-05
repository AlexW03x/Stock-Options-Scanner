import yfinance as yf
import numpy as np
import pandas as pd
import math
import requests
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration Constants from calculator.py ---
MIN_AVG_VOLUME = 1_500_000
MIN_IV_RV_RATIO = 1.25
MAX_TS_SLOPE = -0.00406
FUTURE_DATE_CUTOFF_DAYS = 45
HISTORICAL_DATA_PERIOD = '3mo'
VOLATILITY_WINDOW = 30

# -------------------------------
# Helper Functions from calculator.py
# -------------------------------

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
    
    # Return the last value, ensuring it's not NaN
    return result.dropna().iloc[-1] if not result.dropna().empty else 0.0

def build_term_structure(days: List[int], ivs: List[float]) -> Callable[[int], float]:
    """Builds a spline for interpolating implied volatility."""
    if not days or not ivs or len(days) < 2:
        # Return a lambda that always gives 0 if there's not enough data
        return lambda dte: 0.0
        
    days_np, ivs_np = np.array(days), np.array(ivs)

    sort_idx = days_np.argsort()
    days_sorted, ivs_sorted = days_np[sort_idx], ivs_np[sort_idx]

    spline = interp1d(days_sorted, ivs_sorted, kind='linear', fill_value="extrapolate")
    return lambda dte: float(spline(dte))

# -------------------------------
# Core Stock Processing (Updated)
# -------------------------------
def process_stock(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        name = info.get("longName") or info.get("shortName") or symbol
        logo_url = info.get("logo_url", "")

        # --- Data Fetching ---
        price_history = ticker.history(period=HISTORICAL_DATA_PERIOD)
        if price_history.empty:
            return {"symbol": symbol, "error": "No price history."}

        underlying_price = price_history['Close'].iloc[-1]
        
        exp_dates = list(ticker.options)
        if not exp_dates:
            return {"symbol": symbol, "error": "No options contracts."}

        # --- Calculations ---
        atm_iv = {}
        straddle_price = None
        today = datetime.today().date()
        
        # We only need the first few expiration dates
        for i, exp_date in enumerate(exp_dates[:4]): # Limit to first 4 expirations for speed
            try:
                chain = ticker.option_chain(exp_date)
            except Exception:
                continue # Skip if a single chain fails

            calls, puts = chain.calls, chain.puts
            if calls.empty or puts.empty:
                continue

            atm_call = calls.iloc[(calls['strike'] - underlying_price).abs().idxmin()]
            atm_put = puts.iloc[(puts['strike'] - underlying_price).abs().idxmin()]

            avg_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2.0
            d_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            dte = (d_obj - today).days
            if dte > 0: # Only consider future dates
                atm_iv[dte] = avg_iv

            # Calculate straddle for the nearest expiration
            if i == 0:
                call_mid = (atm_call['bid'] + atm_call['ask']) / 2.0
                put_mid = (atm_put['bid'] + atm_put['ask']) / 2.0
                if call_mid > 0 and put_mid > 0:
                    straddle_price = call_mid + put_mid

        if not atm_iv:
            return {"symbol": symbol, "error": "Could not determine ATM IV."}

        dtes, ivs = list(atm_iv.keys()), list(atm_iv.values())
        term_spline = build_term_structure(dtes, ivs)
        
        # Term structure slope (0 to 45 days)
        ts_slope_0_45 = (term_spline(45) - term_spline(min(dtes))) / (45 - min(dtes)) if min(dtes) < 45 else 0.0

        # IV / RV Ratio
        rv = yang_zhang(price_history)
        iv30_rv30 = term_spline(30) / rv if rv > 0 else 0.0
        
        # 30-day average volume
        avg_volume = price_history['Volume'].rolling(30).mean().iloc[-1]

        # Expected move from straddle
        forecasted_move_pct = (straddle_price / underlying_price * 100) if straddle_price else 0.0

        # --- Recommendation Logic from calculator.py ---
        avg_volume_bool = avg_volume >= MIN_AVG_VOLUME
        iv30_rv30_bool = iv30_rv30 >= MIN_IV_RV_RATIO
        ts_slope_bool = ts_slope_0_45 <= MAX_TS_SLOPE

        if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
            recommendation = "Recommend"
        elif ts_slope_bool and (avg_volume_bool or iv30_rv30_bool):
            recommendation = "Consider"
        else:
            recommendation = "Avoid"
            
        # Earnings date
        try:
            earnings_date = pd.to_datetime(ticker.calendar.get('Earnings Date', [None])[0])
            earnings_date_str = earnings_date.strftime("%Y-%m-%d") if not pd.isna(earnings_date) else "N/A"
            earnings_date_raw = earnings_date.isoformat() if not pd.isna(earnings_date) else None
        except Exception:
            earnings_date_str = "N/A"
            earnings_date_raw = None

        return {
            "symbol": symbol,
            "name": name,
            "logo_url": logo_url,
            "price": underlying_price,
            "forecasted_move": forecasted_move_pct,
            "volatility": iv30_rv30, 
            "earnings_date_str": earnings_date_str,
            "earnings_date_raw": earnings_date_raw,
            "recommendation": recommendation,
        }

    except Exception as e:
        return {"symbol": symbol, "error": f"Processing error: {str(e)}"}

# -------------------------------
# Batch Processing (No changes needed)
# -------------------------------
# in logic.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import concurrent.futures
import time # <-- IMPORT TIME

# ... (keep your existing process_stock function with the new print statements) ...

def process_stock_list(symbols):
    """
    Processes a list of stock symbols in parallel with retries for failed symbols.
    """
    results = []
    MAX_RETRIES = 2
    RETRY_DELAY = 1 # seconds

    symbols_to_process = list(symbols)

    for attempt in range(MAX_RETRIES + 1):
        if not symbols_to_process:
            break # Exit if all symbols are processed

        print(f"\n--- Attempt {attempt + 1} of {MAX_RETRIES + 1} for {len(symbols_to_process)} symbols ---")

        failed_symbols = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(process_stock, symbol): symbol for symbol in symbols_to_process}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    # Only add successful results or results with non-retryable errors
                    if "error" not in result or "No historical data found" in result["error"]:
                         results.append(result)
                    else:
                        # This was a network error or similar, so we should retry
                        failed_symbols.append(symbol)
                except Exception as exc:
                    print(f"Exception for {symbol}: {exc}")
                    failed_symbols.append(symbol)

        if failed_symbols:
            print(f"Retrying {len(failed_symbols)} failed symbols in {RETRY_DELAY} seconds...")
            symbols_to_process = failed_symbols # Set the list for the next attempt
            time.sleep(RETRY_DELAY)
        else:
            print("All symbols processed successfully.")
            break # No failures, exit the loop

    return results