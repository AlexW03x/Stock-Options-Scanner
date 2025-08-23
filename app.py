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

if __name__ == "__main__":
    app.run(debug=True)