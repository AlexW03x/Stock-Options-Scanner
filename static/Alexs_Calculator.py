"""
DISCLAIMER: 

This software is provided solely for educational and research purposes. 
It is not intended to provide investment advice, and no investment recommendations are made herein. 
The developers are not financial advisors and accept no responsibility for any financial decisions or losses resulting from the use of this software. 
Always consult a professional financial advisor before making any investment decisions.
"""


# Please install the follow modules if you haven't already:
# pip install yfinance pandas numpy scipy PySimpleGUI

import FreeSimpleGUI as sg
import yfinance as yf
import numpy as np
import threading
import logging
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple, Any, Callable, Optional
import concurrent.futures
from functools import lru_cache
import pandas as pd

# --- Configuration Constants ---
# Using constants makes the logic clearer and easier to adjust.
MIN_AVG_VOLUME = 1_500_000
MIN_IV_RV_RATIO = 1.25
MAX_TS_SLOPE = -0.00406
FUTURE_DATE_CUTOFF_DAYS = 45
HISTORICAL_DATA_PERIOD = '3mo'
VOLATILITY_WINDOW = 30

# --- Setup Logging ---
# Using the logging module is better practice than print statements for errors.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@lru_cache(maxsize=128)
def get_ticker(symbol: str) -> yf.Ticker:
    """Gets a cached yfinance.Ticker object to avoid repeated API info calls."""
    return yf.Ticker(symbol)


def filter_dates(dates: List[str]) -> List[str]:
    """
    Filters and sorts expiration dates, returning only those up to the first
    date that is at least 45 days in the future.
    """
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=FUTURE_DATE_CUTOFF_DAYS)
    
    # This list comprehension is more concise and efficient than the original loop.
    sorted_dates = sorted([datetime.strptime(date, "%Y-%m-%d").date() for date in dates])

    future_dates = [d for d in sorted_dates if d >= cutoff_date]
    if not future_dates:
        raise ValueError(f"No option expiration date found {FUTURE_DATE_CUTOFF_DAYS}+ days in the future.")

    # Find the index of the first date that meets the cutoff and slice the sorted list.
    first_future_date = future_dates[0]
    cutoff_index = sorted_dates.index(first_future_date)
    
    return [d.strftime("%Y-%m-%d") for d in sorted_dates[:cutoff_index + 1]]


def yang_zhang(price_data: pd.DataFrame, window: int = VOLATILITY_WINDOW, trading_periods: int = 252) -> float:
    """
    Calculates the Yang-Zhang historical volatility.
    This implementation is already vectorized and efficient.
    """
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

    # Use linear interpolation and extrapolate for dates outside the known range.
    spline = interp1d(days_sorted, ivs_sorted, kind='linear', fill_value="extrapolate")

    return lambda dte: float(spline(dte))


def fetch_chain(ticker_obj: yf.Ticker, date: str) -> Tuple[str, Any]:
    """Helper function to fetch a single option chain for concurrent execution."""
    return date, ticker_obj.option_chain(date)


def compute_recommendation(ticker_symbol: str) -> Tuple[bool, Dict[str, Any] | str]:
    """
    Main logic function to fetch data and compute the trading recommendation.
    
    Returns:
        A tuple: (success_boolean, data_or_error_message)
        - (True, result_dictionary) on success.
        - (False, error_string) on failure.
    """
    try:
        ticker_symbol = ticker_symbol.strip().upper()
        if not ticker_symbol:
            return False, "No stock symbol provided."
        
        stock = get_ticker(ticker_symbol)
        
        # --- Data Fetching ---
        # Fetch historical data ONCE.
        price_history = stock.history(period=HISTORICAL_DATA_PERIOD)
        if price_history.empty:
            return False, f"Could not retrieve price history for '{ticker_symbol}'."
        
        # Fix the FutureWarning by using .iloc for positional access.
        underlying_price = price_history['Close'].iloc[-1]
        
        exp_dates = list(stock.options)
        if not exp_dates:
            return False, f"No options contracts found for '{ticker_symbol}'."

        try:
            filtered_exp_dates = filter_dates(exp_dates)
        except ValueError as e:
            return False, str(e)
            
        # --- Concurrent Option Chain Fetching ---
        # This is much faster than a sequential loop for network-bound tasks.
        options_chains = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_date = {executor.submit(fetch_chain, stock, date): date for date in filtered_exp_dates}
            for future in concurrent.futures.as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    d, chain = future.result()
                    options_chains[d] = chain
                except Exception as e:
                    logging.error(f"Could not fetch option chain for {date}: {e}")
        
        if not options_chains:
            return False, "Failed to retrieve any option chain data."

        # --- Calculations ---
        atm_iv, straddle_price = {}, None
        today = datetime.today().date()

        # Iterate through sorted dates to ensure correct straddle calculation.
        for i, exp_date in enumerate(sorted(options_chains.keys())):
            chain = options_chains[exp_date]
            calls, puts = chain.calls, chain.puts

            if calls.empty or puts.empty:
                continue

            # Find ATM options by minimum absolute difference in strike price.
            atm_call = calls.iloc[(calls['strike'] - underlying_price).abs().idxmin()]
            atm_put = puts.iloc[(puts['strike'] - underlying_price).abs().idxmin()]

            # Calculate the average IV.
            avg_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2.0
            d_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            dte = (d_obj - today).days
            atm_iv[dte] = avg_iv

            # Calculate straddle price for the nearest expiration.
            if i == 0:
                call_mid = (atm_call['bid'] + atm_call['ask']) / 2.0
                put_mid = (atm_put['bid'] + atm_put['ask']) / 2.0
                if call_mid > 0 and put_mid > 0:
                    straddle_price = call_mid + put_mid
        
        if not atm_iv:
            return False, "Could not determine ATM IV for any expiration dates."

        dtes, ivs = list(atm_iv.keys()), list(atm_iv.values())
        term_spline = build_term_structure(dtes, ivs)
        
        # Use the nearest expiration DTE instead of a fixed '0'.
        ts_slope_0_45 = (term_spline(45) - term_spline(min(dtes))) / (45 - min(dtes))
        
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)
        
        # Calculate 30-day average volume from the end of the series.
        avg_volume = price_history['Volume'].rolling(30).mean().iloc[-1]

        expected_move = f"{straddle_price / underlying_price * 100:.2f}%" if straddle_price else "N/A"

        # Return a dictionary on success.
        result = {
            'avg_volume': avg_volume >= MIN_AVG_VOLUME,
            'iv30_rv30': iv30_rv30 >= MIN_IV_RV_RATIO,
            'ts_slope_0_45': ts_slope_0_45 <= MAX_TS_SLOPE,
            'expected_move': expected_move
        }
        return True, result

    except Exception as e:
        logging.error(f"An unexpected error occurred in compute_recommendation for {ticker_symbol}: {e}")
        return False, f"Processing error: {e}"


def create_result_window(result: Dict[str, Any]) -> sg.Window:
    """Creates the popup window to display the recommendation."""
    avg_volume_bool = result['avg_volume']
    iv30_rv30_bool = result['iv30_rv30']
    ts_slope_bool = result['ts_slope_0_45']
    
    if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
        title, color = "Recommended ✅", "#008000"
    elif ts_slope_bool and (avg_volume_bool or iv30_rv30_bool):
        title, color = "Consider 🤔", "#FFA500"
    else:
        title, color = "Avoid ❌", "#B22222"
    
    def get_status_text(label: str, success: bool):
        status = "PASS" if success else "FAIL"
        text_color = "#008000" if success else "#B22222"
        return [sg.Text(f"{label}:"), sg.Text(status, text_color=text_color, justification='right')]

    layout = [
        [sg.Text(title, text_color=color, font=("Helvetica", 18, "bold"), justification='center', expand_x=True)],
        [sg.HSeparator()],
        [*get_status_text("Average Volume", avg_volume_bool)],
        [*get_status_text("IV / RV Ratio", iv30_rv30_bool)],
        [*get_status_text("Term Structure Slope", ts_slope_bool)],
        [sg.HSeparator()],
        [sg.Text("Expected Move:"), sg.Text(result['expected_move'], text_color="#00008B", justification='right')],
        [sg.Button("OK", expand_x=True)]
    ]
    return sg.Window("Recommendation", layout, modal=True, finalize=True, size=(300, 220))


def main_gui():
    """Main function to run the GUI application."""
    sg.theme("SystemDefault")
    
    main_layout = [
        [sg.Text("Enter Stock Symbol:"), sg.Input(key="-STOCK-", size=(15, 1), focus=True)],
        [sg.Button("Submit", bind_return_key=True), sg.Button("Exit")],
        [sg.Text("", key="-STATUS-", size=(50, 1), text_color='red')]
    ]
    
    window = sg.Window("Earnings Position Checker", main_layout)
    
    result_holder = {}

    def worker(stock_symbol: str):
        """Worker function to be run in a separate thread."""
        result_holder['result'] = compute_recommendation(stock_symbol)

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Submit":
            stock_symbol = values["-STOCK-"]
            if not stock_symbol:
                window["-STATUS-"].update("Please enter a stock symbol.")
                continue

            window["-STATUS-"].update("Loading, please wait...")
            window.refresh() # Force GUI update

            thread = threading.Thread(target=worker, args=(stock_symbol,), daemon=True)
            thread.start()
            
            # Wait for the thread to finish. This simplified approach is fine
            # since the GUI shows a "Loading" message.
            thread.join()
            
            # This is the fix for the crash. We now expect a tuple.
            success, data = result_holder.get('result', (False, "An unknown error occurred."))
            
            if success:
                window["-STATUS-"].update("") # Clear status on success
                result_window = create_result_window(data)
                while True:
                    event_res, _ = result_window.read()
                    if event_res in (sg.WINDOW_CLOSED, "OK"):
                        break
                result_window.close()
            else:
                # Display the error message from the tuple.
                window["-STATUS-"].update(f"Error: {data}")
    
    window.close()


if __name__ == "__main__":
    main_gui()