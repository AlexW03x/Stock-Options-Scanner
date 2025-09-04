import datetime
import yfinance as yf
import pandas as pd
from flask import Flask, render_template, jsonify, request
from logic import process_stock
from stocks import STOCK_LISTS
import concurrent.futures

app = Flask(__name__)

# Simple in-memory cache of last good series
_LAST_SERIES = []

@app.route("/")
def home():
    today = datetime.date.today()
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
    stock_symbols = [] #create new list of stock symbols
    list_name = request.args.get('list')
    custom_symbols = request.args.get('symbols')

    if list_name and list_name in STOCK_LISTS:
        stock_symbols = STOCK_LISTS[list_name]
    elif custom_symbols:
        stock_symbols = [s.strip().upper() for s in custom_symbols.split(',') if s.strip()]

    if not stock_symbols:
        return jsonify({"error": "No valid stock symbols provided."}), 400

    results = []
    # Using multi-thread to asynchronously fetch stock data to help quicken up the process
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # We use a dict to map future to symbol for easier error tracking
        future_to_stock = {executor.submit(process_stock, symbol): symbol for symbol in stock_symbols}
        for future in concurrent.futures.as_completed(future_to_stock):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                stock_symbol = future_to_stock[future]
                results.append({"symbol": stock_symbol, "error": str(e)})
                
    return jsonify(results)    #return results of new symbols

if __name__ == "__main__":
    app.run(debug=True)
