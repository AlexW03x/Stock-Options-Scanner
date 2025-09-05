from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file, send_from_directory
from logic import process_stock_list # Updated logic.py from previous step
from stocks import STOCK_LISTS
import concurrent.futures
import json # Import the json library

app = Flask(__name__)

CACHE = {} #original cache for single stock data

ALL_STOCKS_CACHE = {} #cache for all stocks data in the form of a dictionary for better information extraction and storage

_LAST_SERIES = None # Initialize _LAST_SERIES to None

def next_noon():
    """Return datetime for next 12 PM local time."""
    now = datetime.now()
    # Reset at midnight
    tomorrow_midnight = datetime(now.year, now.month, now.day) + timedelta(days=1)
    return tomorrow_midnight

def get_all_stocks_cached():
    """Get all stocks from the cache if not expired."""
    entry = ALL_STOCKS_CACHE.get("all_stocks")
    if entry and datetime.now() < entry["expiry"]:
        return entry["data"]
    return None

def set_all_stocks_cached(data):
    """Set the cache for all stocks."""
    ALL_STOCKS_CACHE["all_stocks"] = {"data": data, "expiry": next_noon()}


@app.route("/")
def home():
    today = datetime.today().date()
    data = {
        "date": today, #extract todays date for front-end display
        "top_stocks": ["AAPL", "TSLA", "NVDA"], #debug stock list to test the code
        # Add this line to pass the stock lists as a JSON string
        "stock_lists_json": json.dumps(STOCK_LISTS)
    }
    return render_template("index.html", data=data)


@app.route("/api/get_volatility") #extract volatility index data for front-end charting
def get_volatility():
    global _LAST_SERIES
    ticker = "^VIX"
    try:
        vix = yf.Ticker(ticker)


        intraday = vix.history(period="1d", interval="1m") #grab historic data for volatility index "Daily" with "Minute" candles
        if not intraday.empty: #if results are not empty continue to grab data
            df = intraday[['Close']].reset_index()

            if 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'Date'}, inplace=True) #rename each datetime column attribute to date for better front-end fetch

            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S') #use pandas to manipulate date-time string to Year, Month, Day, Hour, Minutes, Seconds
            series = df[['Date', 'Close']].to_dict(orient="records") #array of dataframe to be stored as series, can be used as historic _LAST_SERIES

            _LAST_SERIES = series
            return jsonify({"volatility_data": series, "mode": "intraday"})

        #If current data unavailable check historic data, change "Daily to 10 Days" and "Minute to Daily Candle"
        daily = vix.history(period="10d", interval="1d")
        if not daily.empty:
            df = daily[['Close']].reset_index()
            # keep only weekdays for security of data representation
            df = df[pd.to_datetime(df['Date']).dt.weekday < 5]
            df = df.tail(5)

            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            series = df[['Date', 'Close']].to_dict(orient="records")

            _LAST_SERIES = series
            return jsonify({"volatility_data": series, "mode": "history"})

        #If both fetching data fails, return previously stored data else, empty chart.
        if _LAST_SERIES:
            return jsonify({"volatility_data": _LAST_SERIES, "mode": "cached"})

        return jsonify({"volatility_data": [], "mode": "empty"})

    except Exception as e:
        if _LAST_SERIES:
            return jsonify({"volatility_data": _LAST_SERIES, "mode": "cached"})
        return jsonify({"volatility_data": [], "mode": "error", "message": str(e)})




# for calculating the stock option considerations
@app.route("/api/calculate")
def calculate():
    try:
        force_rescan = request.args.get('rescan', 'false').lower() == 'true'
        print(f"Received request for /api/calculate. Rescan: {force_rescan}") # <-- ADD THIS

        if not force_rescan:
            cached_data = get_all_stocks_cached()
            if cached_data:
                print("Returning cached data.") # <-- ADD THIS
                return jsonify(cached_data)

        all_symbols = set()
        for stock_list in STOCK_LISTS.values():
            all_symbols.update(stock_list)

        print(f"Cache empty or rescan forced. Fetching data for {len(all_symbols)} symbols...") # <-- ADD THIS
        results = process_stock_list(list(all_symbols))
        print(f"Finished fetching data. Found {len(results)} valid results.") # <-- ADD THIS
        
        results_dict = {res["symbol"]: res for res in results if "error" not in res}

        set_all_stocks_cached(results_dict)
        print("Data cached. Sending response.") # <-- ADD THIS
        return jsonify(results_dict)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {str(e)}") # <-- ADD THIS
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    
    
# code for downloading our calculator app seperately
@app.route("/download-calculator")
def download_calculator():
    try:
        return send_from_directory(directory='static', path='Alexs_Calculator.py', as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"File not found: {str(e)}"}), 404


if __name__ == "__main__":
    app.run(debug=True)