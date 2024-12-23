def fetch_ticks(rdb_cursor, stocks_list):
    """Function to fetch new ticks from Redis Cursor"""
    try:
        print("Fetching New Ticks from Redis")
        sub = rdb_cursor.pubsub()
        sub.subscribe(stocks_list)
        for message in sub.listen():
            if message is not None and isinstance(message, dict):
                print("message : {}".format(message))
    except Exception as e:
        print("Exception in fetching ticks : {}".format(e))
        pass


def fetch_current_day_open(rdb_cursor, symbol):
    """ Function to Fetch Current Day Open """
    try:
        logger.info(msg="fetching current day open price*****")
        candles_list = rdb_cursor.lrange(symbol, 0, 1)
        if candles_list:
            candles_dict = json.loads(candles_list[0])
            logger.info(msg="current day open : {}".format(candles_dict["open"]))
            return candles_dict["open"]
    except Exception as e:
        logger.exception(msg="Exception in fetching current day open : {}".format(e))
        return None


def fetch_live_option_candle(symbol, strike_price, option_type):
    """Function to fetch recent option candle"""
    try:
        logger.info(msg="fetching live option candle")
        if symbol == "NIFTY BANK":
            poc_result = rdb_cursor.get('poc')
        elif symbol == "NIFTY 50":
            poc_result = rdb_cursor.get('n50_poc')
        else:
            poc_result = rdb_cursor.get('poc')

        options_dict = json.loads(poc_result)

        symbol_ = symbol.replace(" ", "_")
        key_name = f"{symbol_}_{str(int(strike_price))}_{option_type}"
        option_candle = options_dict[key_name]

        # Manipulating Tick Values
        option_candle["close"] = option_candle["last_price"]
        option_candle["timestamp"] = option_candle["exchange_timestamp"]
        option_candle["timestamp"] = datetime.datetime.strptime(str(option_candle["timestamp"]), '%Y-%m-%d %H:%M:%S')
        option_candle["date"] = datetime.datetime.strptime(str(option_candle["timestamp"].date()), '%Y-%m-%d')
        # option_candle["expiry"] = datetime.datetime.today().date().isoformat()

        logger.info(msg="live option_candle : {}".format(option_candle))
        return option_candle
    except Exception as e:
        logger.exception(msg="Exception Fetching Option Candle : {}".format(e))
        return None


def fetch_live_option_candles(symbol, strike_price, option_type):
    """ Function to fetch current-day option candles """
    try:
        logger.info("fetchin day option candles")
        symbol_ = symbol.replace(" ", "_")
        key_name = f"{symbol_}_{str(int(strike_price))}_{option_type}"
        logger.info(f"key_name : {key_name}")
        candles = rdb_cursor.lrange(key_name, 0, -1)
        if candles:
            candles_list = [json.loads(i) for i in candles]
            logger.info(f"total live options candles till now : {len(candles_list)}")
            return candles_list
        else:
            return None
    except Exception as e:
        print(f"Exception in fetching day option candles : {e}")
        return None

