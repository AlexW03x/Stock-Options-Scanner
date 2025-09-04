import yfinance as yf
import numpy as np
import pandas as pd
import math
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------
# Black-Scholes Model
# -------------------------------
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

# -------------------------------
# Core Stock Processing
# -------------------------------
def process_stock(symbol):
    """
    Process a single stock symbol with robust error handling.
    Returns a dictionary with result fields or an 'error'.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Fetch info & calendar concurrently
        with ThreadPoolExecutor() as executor:
            future_info = executor.submit(lambda: ticker.info)
            future_calendar = executor.submit(lambda: ticker.calendar)

            info = future_info.result() or {}
            calendar = future_calendar.result() or {}

        name = info.get("longName") or info.get("shortName") or symbol
        logo_url = info.get("logo_url", "")

        # Market data
        hist = ticker.history(period="1y", interval="1d")
        if hist.empty:
            return {"symbol": symbol, "error": "No historical data found."}

        underlying_price = hist["Close"].iloc[-1]

        # Realized volatility (30-day)
        log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        rv30 = log_returns.rolling(window=30).std().iloc[-1] * np.sqrt(252)

        # Option chain
        expirations = ticker.options
        if not expirations:
            return {"symbol": symbol, "error": "No options data available."}

        front_expiry = expirations[0]
        try:
            opt_chain = ticker.option_chain(front_expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts
        except Exception:
            return {"symbol": symbol, "error": "Error fetching option chain."}

        if calls.empty or puts.empty:
            return {"symbol": symbol, "error": "Incomplete option chain."}

        # ATM strike
        atm_strike = min(
            calls["strike"],
            key=lambda k: abs(k - underlying_price),
        )

        # Calls & puts near ATM
        atm_calls = calls[calls["strike"] == atm_strike]
        atm_puts = puts[puts["strike"] == atm_strike]

        if atm_calls.empty or atm_puts.empty:
            return {"symbol": symbol, "error": "Could not find ATM options."}

        # Safe extraction
        try:
            call_bid = float(atm_calls["bid"].iloc[0] or 0)
            call_ask = float(atm_calls["ask"].iloc[0] or 0)
            put_bid = float(atm_puts["bid"].iloc[0] or 0)
            put_ask = float(atm_puts["ask"].iloc[0] or 0)
        except Exception:
            return {"symbol": symbol, "error": "Error parsing bid/ask."}

        call_mid = (call_bid + call_ask) / 2
        put_mid = (put_bid + put_ask) / 2

        # Days to expiry
        expiry_dt = pd.to_datetime(front_expiry)
        T = max((expiry_dt - datetime.now()).days / 365, 1 / 365)

        # Risk-free rate
        r = 0.05

        # Implied vols
        call_iv = atm_calls["impliedVolatility"].iloc[0]
        put_iv = atm_puts["impliedVolatility"].iloc[0]
        iv30 = (call_iv + put_iv) / 2 if not (pd.isna(call_iv) or pd.isna(put_iv)) else None

        if iv30 is None or iv30 <= 0:
            return {"symbol": symbol, "error": "Could not determine ATM IV."}

        # Expected move (1 std deviation)
        forecasted_move = underlying_price * iv30 * math.sqrt(T)

        # Relative value: IV vs RV
        rel_value = iv30 / rv30 if rv30 and rv30 > 0 else None

        # Earnings date extraction
        earnings_date = None
        try:
            if isinstance(calendar, dict):
                val = calendar.get("Earnings Date")
                if isinstance(val, (list, tuple)) and val:
                    earnings_date = val[0]
                else:
                    earnings_date = val
            elif hasattr(calendar, "to_dict"):
                cal_dict = calendar.to_dict()
                for k, v in cal_dict.items():
                    if "earn" in str(k).lower():
                        earnings_date = v[0] if isinstance(v, (list, tuple)) and v else v
                        break
                if earnings_date is None and cal_dict:
                    first = list(cal_dict.values())[0]
                    earnings_date = first[0] if isinstance(first, (list, tuple)) and first else first
            else:
                get_fn = getattr(calendar, "get", None)
                if callable(get_fn):
                    earnings_date = get_fn("Earnings Date", None)
                    if isinstance(earnings_date, (list, tuple)) and earnings_date:
                        earnings_date = earnings_date[0]
        except Exception:
            earnings_date = None

        # Normalize earnings_date
        if earnings_date is not None:
            try:
                if hasattr(earnings_date, "to_pydatetime"):
                    earnings_date = earnings_date.to_pydatetime()
                elif not isinstance(earnings_date, datetime):
                    earnings_date = pd.to_datetime(earnings_date)
                    if pd.isna(earnings_date):
                        earnings_date = None
                    else:
                        earnings_date = earnings_date.to_pydatetime()
            except Exception:
                earnings_date = None

        # Recommendation
        recommendation = (
            "Recommend"
            if rel_value and rel_value > 1.2
            else "Avoid" if rel_value and rel_value < 0.8
            else "Consider"
        )

        return {
            "symbol": symbol,
            "name": name,
            "logo_url": logo_url,
            "price": underlying_price,
            "rv30": rv30,
            "iv30": iv30,
            "forecasted_move": forecasted_move,
            "volatility": rel_value,
            "earnings_date_str": earnings_date.strftime("%Y-%m-%d") if earnings_date else "N/A",
            "earnings_date_raw": earnings_date.isoformat() if earnings_date else None,
            "recommendation": recommendation,
        }

    except Exception as e:
        return {"symbol": symbol, "error": f"A processing error occurred: {str(e)}"}

# -------------------------------
# Batch Processing
# -------------------------------
def process_stock_list(symbols):
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {executor.submit(process_stock, sym): sym for sym in symbols}
        for future in as_completed(future_to_symbol):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                results.append({
                    "symbol": future_to_symbol[future],
                    "error": f"A processing error occurred: {str(e)}"
                })
    return results
