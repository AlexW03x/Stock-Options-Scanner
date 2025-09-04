from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from flask import Flask, render_template, jsonify, request
from logic import process_stock, process_stock_list
from stocks import STOCK_LISTS
import concurrent.futures

app = Flask(__name__)

# Simple in-memory cache of last good series
_LAST_SERIES = []

# caching our requests of volatility
CACHE = {}

def next_noon():
    """Return datetime for next 12 PM local time."""
    now = datetime.now()
    tomorrow_noon = datetime(now.year, now.month, now.day, 12, 0)
    if now >= tomorrow_noon:
        tomorrow_noon += timedelta(days=1)
    return tomorrow_noon

def get_cached(key):
    entry = CACHE.get(key)
    if entry and datetime.now() < entry["expiry"]:
        return entry["data"]
    return None

def set_cached(key, data):
    CACHE[key] = {"data": data, "expiry": next_noon()}

@app.route("/")
def home():
    today = datetime.today().date()
    data = {"date": today, "top_stocks": ["AAPL", "TSLA", "NVDA"]}
    return render_template("index.html", data=data)

@app.route("/api/get_volatility")
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
        list_name = request.args.get("list")
        symbols = request.args.get("symbols")

        if list_name:
            cache_key = ("list", list_name)
            cached = get_cached(cache_key)
            if cached:
                return jsonify(cached)

            symbols_list = STOCK_LISTS.get(list_name, [])
            results = process_stock_list(symbols_list)
            set_cached(cache_key, results)
            return jsonify(results)

        elif symbols:
            requested = [s.strip().upper() for s in symbols.split(",")]
            results = []
            to_fetch = []

            for sym in requested:
                cached = get_cached(("symbol", sym))
                if cached:
                    results.append(cached)
                else:
                    to_fetch.append(sym)

            if to_fetch:
                fetched = process_stock_list(to_fetch)
                for res in fetched:
                    if "error" not in res:
                        set_cached(("symbol", res["symbol"]), res)
                    results.append(res)

            return jsonify(results)

        else:
            return jsonify({"error": "No list or symbols provided"}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()   # print full error to server logs
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
