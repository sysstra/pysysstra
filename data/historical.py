import requests
from data import api_key, data_url


def fetch_eod_candles(exchange, symbol, start_date, end_date):
    """ Function to fetch End of Day Candles for symbol """
    try:
        headers = {"x-api-key": api_key, "content-type": "application/json"}
        request_data = {"exchange": exchange, "symbol": symbol, "from_date": start_date, "to_date": end_date}
        request_url = f"{data_url}/fetch-eod-data"
        eod_data = requests.post(url=request_url, headers=headers, json=request_data)
        return eod_data
    except Exception as e:
        print(f"Exception in fetching eod candles : {e}")
        return []


def fetch_index_candles(exchange, symbol, start_date, end_date, granularity=1):
    """ Function to fetch candles for the respective date """
    try:
        headers = {"x-api-key": api_key, "content-type": "application/json"}
        request_data = {"exchange": exchange, "symbol": symbol, "from_date": start_date, "to_date": end_date,
                        "granularity": granularity}
        request_url = f"{data_url}/fetch-index-data"
        candles_data = requests.post(url=request_url, headers=headers, json=request_data)
        return candles_data
    except Exception as e:
        print(f"Exception in fetching date candles : {e}")
        return []


def fetch_option_candles(exchange, underlying_symbol, start_date, end_date, option_type, strike_price, expiry="near", granularity=1, timestamp=None):
    """ Function to Fetch Options Trade Data """
    try:
        headers = {"x-api-key": api_key, "content-type": "application/json"}
        request_data = {"exchange": exchange, "underlying_symbol": underlying_symbol, "from_date": start_date,
                        "to_date": end_date,  "option_type": option_type, "strike_price": strike_price,
                        "expiry": expiry, "granularity": granularity}
        request_url = f"{data_url}/fetch-options-data"
        candles_data = requests.post(url=request_url, headers=headers, json=request_data)
        return candles_data
    except Exception as e:
        print(f"Exception in fetching options candle : {e}")
        return []


def fetch_option_candles_by_symbol(exchange, symbol, start_date, end_date, granularity=1):
    """ Function to Fetch Options Trade Data """
    try:
        headers = {"x-api-key": api_key, "content-type": "application/json"}
        request_data = {"exchange": exchange, "symbol": symbol, "from_date": start_date, "to_date": end_date, "granularity": granularity}
        request_url = f"{data_url}/fetch-options-data-by-symbol"
        candles_data = requests.post(url=request_url, headers=headers, json=request_data)
        return candles_data

    except Exception as e:
        print(f"Exception in fetching options candle : {e}")
        return []


def fetch_option_candle_by_timestamp(exchange, underlying_symbol, strike_price, option_type, timestamp, expiry="near"):
    """ Function to Fetch Order Candle based on timestamp """
    try:
        headers = {"x-api-key": api_key, "content-type": "application/json"}
        request_data = {"exchange": exchange, "underlying_symbol": underlying_symbol, "option_type": option_type,
                        "strike_price": strike_price, "timestamp": timestamp, "expiry": expiry}
        request_url = f"{data_url}/fetch-option-data-by-timestamp"
        candles_data = requests.post(url=request_url, headers=headers, json=request_data)
        return candles_data

    except Exception as e:
        print(f"Exception in fetching order candle : {e}")
        return []


if __name__ == '__main__':
    # eod_candles = fetch_eod_candles(exchange="NSE", symbol="NIFTY 50", start_date="01/01/2024", end_date="05/01/2024")
    # candles = fetch_index_candles(exchange="NSE", symbol="NIFTY 50", start_date="01/01/2024", end_date="05/01/2024")
    # candles = fetch_option_candles(exchange="NSE", underlying_symbol="SENSEX", start_date="19/12/2024",
    #                                end_date="19/12/2024", strike_price=80200, option_type="CE", expiry="near")
    # candles = fetch_option_candles_by_symbol(exchange="NSE", symbol="SENSEX24D2080200CE", start_date="19/12/2024",
    #                                          end_date="19/12/2024")
    candle = fetch_option_candle_by_timestamp(exchange="NSE", underlying_symbol="SENSEX",  strike_price=80200, option_type="CE", expiry="near",
                                              timestamp="2024-12-19 14:45:00")
    print(candle)
