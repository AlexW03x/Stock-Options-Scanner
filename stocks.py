import requests
import csv

def fetch_sp500_tickers():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    resp = requests.get(url)
    resp.raise_for_status()
    reader = csv.DictReader(resp.text.splitlines())
    return [row["Symbol"] for row in reader]

STOCK_LISTS = {
    "nasdaq100": [
        "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
        "AMZN", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG", "BKR",
        "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO",
        "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG",
        "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC",
        "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "MDLZ", "MCHP", "MELI",
        "META", "MNST", "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NVDA", "NXPI", "ODFL",
        "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR", "PYPL", "QCOM", "REGN",
        "ROP", "ROST", "SBUX", "SHOP", "SNPS", "TEAM", "TMUS", "TRI", "TSLA", "TTD",
        "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS"
    ],
    "sp500": [
        # Placeholder, will be populated by fetch_sp500_tickers()
    ],
    "us30": [ # Dow Jones Industrial Average
        "AXP", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
        "KO", "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "CRM", "VZ", "V", "WBA", "WMT", "DIS"
    ],
}

STOCK_LISTS["sp500"] = fetch_sp500_tickers()