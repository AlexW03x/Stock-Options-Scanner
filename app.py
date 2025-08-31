import yfinance as yf
from flask import jsonify
from flask import Flask, render_template
import datetime

app = Flask(__name__)

@app.route("/")
def home():
    today = datetime.date.today()
    
    data = {
        "date": today,
        "top_stocks": ["AAPL", "TSLA", "NVDA"]
    }
    
    return render_template("index.html", data=data)

@app.route("/api/get_volatility")
def get_volatility():
    ticker = "^VIX"
    vix_data = yf.Ticker(ticker)
    
    #fetching the last month's daily volatility data to showcase trends on the about section
    vix_info = vix_data.history(period="1m", interval="1d")
    data = vix_info['Close'].reset_index().to_dict(orient='records')
    return jsonify({"volatility_data": data})

if __name__ == "__main__":
    app.run(debug=True)