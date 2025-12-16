import re
import yfinance as yf

def download_and_concat_tickers(tickers, start_date=None, end_date=None, interval='1d'):
    """Download data of provided tickers for the specified interval"""
    if start_date:
        df = yf.download(tickers, start=start_date, end=end_date, interval=interval)
    else:
        df = yf.download(tickers, period="max", interval=interval)

    df.columns = [f"{ticker}_{field}" for field, ticker in df.columns]

    return df

def clean_name(name: str):
    s = str(name)

    # 1) remove leading non-alphanumeric characters (anything not [A-Za-z0-9])
    s = re.sub(r'^[^A-Za-z0-9]+', '', s)

    # 2) replace any remaining non-word characters with underscores
    # \w = [A-Za-z0-9_]; anything else becomes '_'
    s = re.sub(r'\W+', '_', s)

    # 3) collapse multiple underscores
    s = re.sub(r'_+', '_', s)

    # 4) strip trailing underscores (optional but tidy)
    s = s.strip('_')

    print("✅ data cleaned")

    return s
