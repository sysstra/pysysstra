def add_order_to_redis(request_id, order_dict, mode):
    """Function to add order to redis"""
    try:
        logger.info(msg="Adding Order to Redis DB for : {}".format(request_id))
        rdb_cursor.rpush(str(request_id) + "_orders", json.dumps(order_dict, default=str))
        rdb_cursor.publish(str(request_id) + "_orders", json.dumps(order_dict, default=str))
        rdb_cursor.publish(str(order_dict["user_id"]) + "_{}".format(mode) + "_orders", json.dumps(order_dict, default=str))
        logger.info(msg="order added in redis")
    except Exception as e:
        logger.exception(msg="Exception in adding order in redis : {}".format(e))
        pass


def fetch_orders_list(request_id):
    """ Function to fetch an orders list for request_id """
    try:
        logger.info(msg="Fetching Orders List for : {}".format(request_id))
        orders_list_json = rdb_cursor.lrange(str(request_id)+"_orders", 0, -1)
        orders_list = [json.loads(i) for i in orders_list_json]
        # logger.info(msg="Orders List : {}".format(orders_list))
        return orders_list
    except Exception as e:
        logger.exception(msg="Exception in fetching orders list : {}".format(e))
        pass


def fetch_last_order(request_id):
    """Function to fetch last order from redis database"""
    try:
        logger.info(msg="Fetching Last Order from Redis")
        last_order = json.loads(rdb_cursor.lindex(str(request_id) + "_orders", -1))
        logger.info("last_order in redis : {}".format(last_order))
        return last_order
    except Exception as e:
        logger.exception(msg="Exception in fetching last order : {}".format(e))
        pass


def save_bt_report(report_dict):
    """Function to Save Backtest Report in Database"""
    try:
        logger.info(msg="saving backtest report in DB")
        app_db_cursor[bt_reports_col].insert_one(report_dict)
        logger.info(msg="updating request status")
        app_db_cursor[bt_request_col].update_one({"_id": report_dict["request_id"]},
                                                 {"$set": {"status": "done"}})
    except Exception as e:
        logger.exception("Exception in saving BT Report : {}".format(e))
        pass


def place_bt_order(order_candle, option_type, strike_price, position_type, quantity, transaction_type, order_type, exit_type=None, quantity_left=0, params=None, market_type="cash", trade_type=None, trigger_price=None, lot_size=25, user_id=None, strategy_id=None, request_id=None, exchange="NSE"):
    """ Function to place Backtesting Order """
    try:
        logger.info(msg="* placing backtesting order")
        order_dict = {"exchange": exchange, "user_id": str(user_id), "strategy_id": str(strategy_id),
                      "request_id": str(request_id), "order_type": order_type, "position_type": position_type, "quantity": quantity,
                      "transaction_type": transaction_type, "option_type": option_type, "strike_price": strike_price,
                      "exit_type": exit_type, "quantity_left": quantity_left, "lot_size": lot_size,
                      "trade_type": trade_type}
        if trigger_price:
            order_dict["trigger_price"] = trigger_price
        else:
            order_dict["trigger_price"] = order_candle["close"]

        order_dict["order_timestamp"] = str(order_candle["timestamp"])
        order_dict["tradingsymbol"] = order_candle["symbol"]
        order_dict["date"] = str(order_candle["date"])
        if market_type == "cash":
            order_dict["expiry"] = ""
        else:
            order_dict["expiry"] = order_candle["expiry"]

        order_dict["day"] = order_candle["date"].strftime("%A")
        if params:
            order_dict.update(params)
        logger.info(msg="* bt_order : {}".format(order_dict))
        # orders_list.append(order_dict)

        add_order_to_redis(request_id=str(request_id), order_dict=order_dict, mode="bt")
        orders_list = fetch_orders_list(request_id=str(request_id))

        return orders_list

    except Exception as e:
        logger.exception(msg="Exception in placing backtesting order : {}".format(e))
        pass


def place_vt_order(order_candle, option_type, strike_price, position_type, quantity, transaction_type,
                   order_type, exit_type=None, quantity_left=0, params=None, market_type="options", trade_type=None,
                   trigger_price=None, lot_size=15, user_id=None, strategy_id=None, request_id=None, exchange="NSE"):
    """Function to Place Virtual Trading Order"""

    try:
        logger.info(msg="* placing virtual trade order")
        order_dict = {"exchange": exchange,
                      "user_id": user_id,
                      "strategy_id": strategy_id,
                      "request_id": request_id,
                      "option_type": option_type,
                      "strike_price": strike_price,
                      "quantity": quantity,
                      "position_type": position_type,
                      "transaction_type": transaction_type,
                      "trade_type": trade_type,
                      "trade_action": trade_type,
                      "order_type": order_type,
                      "exit_type": exit_type,
                      "quantity_left": quantity_left,
                      "lot_size": lot_size
                      }

        if trigger_price:
            order_dict["trigger_price"] = trigger_price
        else:
            order_dict["trigger_price"] = order_candle["close"]
        order_dict["order_timestamp"] = datetime.datetime.now().replace(microsecond=0)
        order_dict["tradingsymbol"] = order_candle["symbol"]
        order_dict["date"] = order_candle["date"]
        if market_type == "cash":
            order_dict["expiry"] = ""
        else:
            order_dict["expiry"] = order_candle["expiry"]

        order_dict["day"] = order_candle["timestamp"].strftime("%A")
        if params:
            order_dict.update(params)

        logger.info(msg="* vt_order : {}".format(order_dict))

        # Saving Order Details to Database
        save_vt_order(order_dict=order_dict.copy())

        # orders_list.append(order_dict)

        order_dict["user_id"] = str(order_dict["user_id"])
        order_dict["strategy_id"] = str(order_dict["strategy_id"])
        order_dict["request_id"] = str(order_dict["request_id"])
        order_dict["order_timestamp"] = str(order_dict["order_timestamp"])
        order_dict["date"] = str(order_dict["date"])
        order_dict["expiry"] = str(order_dict["expiry"])

        add_order_to_redis(str(request_id), order_dict, mode="vt")
        orders_list = fetch_orders_list(str(request_id))

        # Creating Alert Dict
        alert_dict = {"user_id": str(order_dict["user_id"]),
                      "strategy_id": str(order_dict["strategy_id"]),
                      "request_id": str(order_dict["request_id"]),
                      "mode": "vt",
                      "exit_type": exit_type,
                      "symbol": order_candle["symbol"],
                      "quantity": quantity,
                      "price": order_dict["trigger_price"],
                      "quantity_left": quantity_left,
                      "trade_type": trade_type,
                      "template_id": 0
                      }

        # Sending Alert
        send_order_alert(alert_dict)

        return orders_list
    except Exception as e:
        logger.exception(msg="Exception in placing virtual trade : {}".format(e))
        pass


def save_vt_order(order_dict):
    """Function to save order in Database"""
    try:
        logger.info(msg="* saving VT order to DB *****")

        app_db_cursor[vt_orders_col].insert_one(order_dict)
    except Exception as e:
        logger.exception(msg="Exception in saving VT order in DB : {}".format(e))
        pass


def save_vt_trade(trade_dict):
    """Function to save order in Database"""
    try:
        logger.info(msg="* saving VT trade to DB *****")
        app_db_cursor[vt_trades_col].insert_one(trade_dict)
    except Exception as e:
        logger.exception(msg="Exception in saving VT trade in DB : {}".format(e))
        pass


def place_lt_order(tradingsymbol, quantity, transaction_type, order_type, lot_size=15, exchange="NSE",
                   credential_id=None, trigger_price=None, order_price=None):
    """ Function to Place Live Trading Order """
    try:
        logger.info(msg="* Placing live trade order")

        if order_type == "MARKET":
            order_data_params = {"tradingsymbol": tradingsymbol,
                                 "exchange": exchange,
                                 "transaction_type": transaction_type,
                                 "quantity": quantity * lot_size,
                                 "order_type": order_type,
                                 "product": "MIS",
                                 "validity": "DAY"}

            order_response = place_live_order(credential_id=credential_id, order_details=order_data_params)
            logger.info(msg="order_response : {}".format(order_response))

            if order_response["status"] == "COMPLETE":
                # return "success", order_response["broker_response"][0]["broker_response"]
                return "success", order_response
            else:
                return "failed", None

        elif order_type == "SL":
            logger.info(msg="*** Placing SL Limit Order ***")
            order_data_params = {"tradingsymbol": tradingsymbol, "exchange": exchange,
                                 "transaction_type": transaction_type,
                                 "quantity": quantity * lot_size, "product": "MIS", "validity": "DAY",
                                 "order_type": "SL", "trigger_price": trigger_price, "price": order_price}
            logger.info(f"order_data_params : {order_data_params}")
            order_response = place_live_order(credential_id=credential_id, order_details=order_data_params)
            logger.info(msg="sl_order_response : {}".format(order_response))

            if order_response["status"] == "success":
                # return "success", order_response["data"]["order_id"]
                # return "success", order_response["order_id"][0]
                return "success", order_response["order_id"]
            else:
                return "failed", None

        elif order_type == "LIMIT":
            logger.info(msg="*** Placing Limit BUY Order ***")
            order_data_params = {"tradingsymbol": tradingsymbol, "exchange": exchange,
                                 "transaction_type": transaction_type, "quantity": quantity * lot_size,
                                 "product": "MIS", "validity": "TTL", "validity_ttl": 1,
                                 "order_type": "LIMIT",  "price": order_price}

            order_response = place_live_order(credential_id=credential_id, order_details=order_data_params)
            logger.info(msg="sl_order_response : {}".format(order_response))

            if order_response["status"] == "success":
                # return "success", order_response["data"]["order_id"]
                # return "success", order_response["order_id"][0]
                return "success", order_response["order_id"]
            else:
                return "failed", None

    except Exception as e:
        logger.exception(msg="Exception in placing live trade : {}".format(e))
        return "failed", None


def save_lt_order(tradingsymbol, option_type, strike_price, position_type, quantity, transaction_type, order_type,
                  orders_list, exit_type=None, quantity_left=0, params=None, market_type="cash", trade_type=None,
                  expiry=None, trigger_price=None, lot_size=25, user_id=None, strategy_id=None, request_id=None,
                  exchange="NSE", exchange_timestamp=None, order_id=None, broker_response=None, sl_order_id=None):

    """Function to save order in Database"""
    try:
        logger.info(msg="* Saving LT order to DB")
        order_dict = {"exchange": exchange, "user_id": user_id, "strategy_id": strategy_id,
                      "request_id": request_id, "tradingsymbol": tradingsymbol, "transaction_type": transaction_type,
                      "quantity": quantity, "position_type": position_type, "order_type": order_type,
                      "exit_type": exit_type, "quantity_left": quantity_left, "lot_size": lot_size, "trade_type": trade_type,
                      "trade_action": trade_type,
                      "exchange_timestamp": exchange_timestamp, "status": "COMPLETE", "trigger_price": trigger_price, "order_id": order_id}

        if sl_order_id:
            order_dict["sl_order_id"] = sl_order_id

        if market_type == "cash":
            order_dict["expiry"] = ""
            order_dict["option_type"] = ""
            order_dict["strike_price"] = ""
        else:
            order_dict["expiry"] = expiry
            order_dict["option_type"] = option_type
            order_dict["strike_price"] = strike_price

        order_dict["order_timestamp"] = exchange_timestamp
        order_dict["date"] = datetime.datetime.strptime(str(datetime.datetime.today().date()), '%Y-%m-%d')
        order_dict["day"] = order_dict["date"].strftime("%A")

        if params:
            order_dict.update(params)
        logger.info(msg="* lt_order : {}".format(order_dict))

        # Creating New Dict for saving data in to db
        lt_order_dict = {}
        for key in order_dict.keys():
            lt_order_dict[key] = order_dict[key]

        lt_order_dict["order_id"] = order_id
        lt_order_dict["broker_response"] = broker_response
        lt_order_dict["trade_action"] = lt_order_dict["trade_type"]

        # Saving Order Details to Database
        app_db_cursor[lt_orders_col].insert_one(lt_order_dict)

        order_dict["strategy_id"] = str(order_dict["strategy_id"])
        order_dict["request_id"] = str(order_dict["request_id"])
        order_dict["user_id"] = str(order_dict["user_id"])
        order_dict["order_timestamp"] = str(order_dict["order_timestamp"])
        order_dict["exchange_timestamp"] = str(order_dict["exchange_timestamp"])
        order_dict["expiry"] = str(order_dict["expiry"])
        order_dict["date"] = str(order_dict["date"])
        order_dict["order_id"] = order_id

        add_order_to_redis(str(request_id), order_dict, mode="lt")
        orders_list = fetch_orders_list(str(request_id))
        logger.info(msg="Order List Now : {}".format(orders_list))

        # Creating Alert Dict
        alert_dict = {"user_id": str(order_dict["user_id"]),
                      "strategy_id": str(order_dict["strategy_id"]),
                      "request_id": str(order_dict["request_id"]),
                      "mode": "lt",
                      "exit_type": exit_type,
                      "symbol": tradingsymbol,
                      "quantity": quantity,
                      "price": trigger_price,
                      "quantity_left": quantity_left,
                      "trade_type": trade_type,
                      "template_id": 0
                      }

        # Sending Alert
        send_order_alert(alert_dict)
        return "success", orders_list

    except Exception as e:
        logger.exception(msg="Exception in Saving Order in DB : {}".format(e))
        return "failed", orders_list


def save_lt_trade(trade_dict):
    """Function to save order in Database"""
    try:
        logger.info(msg="* saving LT trade to DB *****")
        app_db_cursor[lt_trades_col].insert_one(trade_dict)
    except Exception as e:
        logger.exception(msg="Exception in saving LT trade in DB : {}".format(e))
        pass


def check_open_orders(orders_list):
    """ Function to open orders available """
    try:
        if orders_list:
            quantity_dict = {}
            for order in orders_list:
                trade_symbol = order["tradingsymbol"]
                quantity_dict[trade_symbol] = {}
                quantity_dict[trade_symbol]["buy_quantity"] = 0
                quantity_dict[trade_symbol]["sell_quantity"] = 0
                quantity_dict[trade_symbol]["quantity"] = 0
                quantity_dict[trade_symbol]["option_type"] = ""
                quantity_dict[trade_symbol]["strike_price"] = ""
                quantity_dict[trade_symbol]["exit_levels"] = []
                quantity_dict[trade_symbol]["order_timestamp"] = ""
                quantity_dict[trade_symbol]["quantity_left"] = 0
                quantity_dict[trade_symbol]["bnf_price"] = 0
                quantity_dict[trade_symbol]["expiry"] = ""
                quantity_dict[trade_symbol]["sl_order_id"] = ""
                quantity_dict[trade_symbol]["option_order_price"] = 0
                quantity_dict[trade_symbol]["hka_option_order_price"] = 0
                quantity_dict[trade_symbol]["trailing_sl"] = 0

            for order in orders_list:
                trade_symbol = order["tradingsymbol"]
                if order["trade_type"] == "ENTRY":
                    quantity_dict[trade_symbol]["buy_quantity"] = order["quantity"]
                    quantity_dict[trade_symbol]["quantity"] = order["quantity"]
                    quantity_dict[trade_symbol]["option_type"] = order["option_type"]
                    quantity_dict[trade_symbol]["strike_price"] = order["strike_price"]
                    quantity_dict[trade_symbol]["trigger_price"] = order["trigger_price"]
                    quantity_dict[trade_symbol]["order_timestamp"] = datetime.datetime.strptime(str(order["order_timestamp"]), '%Y-%m-%d %H:%M:%S')
                    quantity_dict[trade_symbol]["quantity_left"] = order["quantity_left"]
                    quantity_dict[trade_symbol]["bnf_price"] = order["bnf_price"]
                    quantity_dict[trade_symbol]["hka_option_order_price"] = order["hka_option_order_price"]
                    quantity_dict[trade_symbol]["option_order_price"] = order["option_order_price"]

                if "expiry" in order:
                    quantity_dict[trade_symbol]["expiry"] = order["expiry"]

                if "sl_order_id" in order:
                    quantity_dict[trade_symbol]["sl_order_id"] = order["sl_order_id"]

                if "trailing_sl" in order:
                    quantity_dict[trade_symbol]["trailing_sl"] = order["trailing_sl"]

                elif order["trade_type"] == "EXIT":
                    quantity_dict[trade_symbol]["sell_quantity"] += order["quantity"]
                    quantity_dict[trade_symbol]["quantity_left"] = order["quantity_left"]
                    quantity_dict[trade_symbol]["exit_levels"].append(order["exit_type"])

            final_out = {}
            for entries in quantity_dict:
                # if quantity_dict[entries]["buy_quantity"] - quantity_dict[entries]["sell_quantity"] > 0:
                if quantity_dict[entries]["quantity_left"] > 0:
                    final_out[entries] = quantity_dict[entries]
            return final_out

        else:
            return {}
    except Exception as e:
        logger.exception("Exception in checking open orders : {}".format(e))
        return {}


def convert_to_trades(orders_list, market_type, order_exit_levels, mode, broker):
    """Function to convert Orders to Trades """
    try:
        logger.info(msg="* Converting Orders to Trades")
        trade_dict = {}
        trades_array = []
        for order in orders_list:
            # logger.info(msg="order : {}".format(order))
            if order["trade_type"] == "ENTRY":
                trade_dict["date"] = order["date"]
                trade_dict["stock"] = order["tradingsymbol"]
                trade_dict["lot_size"] = order["lot_size"]
                trade_dict["trade_type"] = order["trade_type"]
                trade_dict["bnf_price"] = order["bnf_price"]
                trade_dict["bar_color"] = order["bar_color"]
                trade_dict["entry_time"] = order["order_timestamp"]
                trade_dict["entry_price"] = order["trigger_price"]
                trade_dict["quantity"] = order["quantity"]
                trade_dict["pnl"] = 0
                trade_dict["points"] = 0
                trade_dict["exit_time"] = None
                trade_dict["exit_price"] = None
                trade_dict["exit_type"] = ""
                trade_dict["day"] = order["day"]
                trade_dict["expiry"] = order["expiry"]
                trade_dict["brokerage"] = 0
                trade_dict["net_pnl"] = 0

                if mode == "lt" or mode == "vt":
                    trade_dict["date"] = datetime.datetime.strptime(str(datetime.datetime.today().date()), '%Y-%m-%d')
                    trade_dict["user_id"] = bson.ObjectId(order["user_id"])
                    trade_dict["strategy_id"] = bson.ObjectId(order["strategy_id"])
                    trade_dict["request_id"] = bson.ObjectId(order["request_id"])

            else:
                if market_type == "cash":
                    if order["trade_type"] == "SHORT":
                        points = trade_dict["entry_price"] - order["trigger_price"]
                        trade_dict["points"] += round(points)
                        trade_dict["pnl"] += round(order["quantity"] * points)
                    else:
                        points = order["trigger_price"] - trade_dict["entry_price"]
                        trade_dict["points"] += round(points)
                        trade_dict["pnl"] += round(order["quantity"] * points)
                    # trade_dict["pnl"] += round(order["quantity"] * trade_dict["points"])
                else:
                    points = order["trigger_price"] - trade_dict["entry_price"]
                    trade_dict["points"] += round(points)
                    trade_dict["pnl"] += round(order["quantity"] * trade_dict["points"] * trade_dict["lot_size"])

                if trade_dict["exit_type"]:
                    trade_dict["exit_type"] += "|" + order["exit_type"]
                else:
                    trade_dict["exit_type"] = order["exit_type"]

                if order["exit_type"] in order_exit_levels:
                    trade_dict["exit_time"] = order["order_timestamp"]
                    trade_dict["exit_price"] = order["trigger_price"]

                    brokerage, net_pnl = calculate_brokerage(buy_price=trade_dict["entry_price"], sell_price=trade_dict["exit_price"],
                                                             quantity=order["quantity"] * trade_dict["lot_size"], broker=broker)
                    trade_dict["brokerage"] += brokerage
                    trade_dict["net_pnl"] += net_pnl

                    trades_array.append(trade_dict)

                    # Emptying Trade Dict for next trade
                    trade_dict = {}
                else:
                    brokerage, net_pnl = calculate_brokerage(buy_price=trade_dict["entry_price"], sell_price=order["trigger_price"],
                                                             quantity=order["quantity"] * trade_dict["lot_size"], broker=broker)
                    trade_dict["brokerage"] += brokerage
                    trade_dict["net_pnl"] += net_pnl
        logger.info(msg="* total trades : {}".format(len(trades_array)))
        return trades_array
    except Exception as e:
        logger.exception("Exception in converting orders to trades : {}".format(e))
        pass
