import datetime
import json
import math
import itertools
import traceback as tb
import numpy as np
import pandas_ta as ta
import requests
import pandas as pd
# from custom_indicators import *
# from config import *


def change_granularity(data_df, granularity):
    """ Function to Change Granularity for Provided Dataframe """
    try:
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        if 'date' in data_df.columns:
            data_df['date'] = pd.to_datetime(data_df['date'])

        if 'symbol' in data_df.columns:
            symbols = sorted(set(data_df['symbol']))
        else:
            symbols = [""]

        final_data_li = list()

        def create_row_dict(gran_calc_list):
            open_ = gran_calc_list[0]['open']
            close_ = gran_calc_list[-1]['close']
            high_ = max(gran_calc_list, key=lambda x: x['high'])['high']
            low_ = min(gran_calc_list, key=lambda x: x['low'])['low']
            timestamp_ = gran_calc_list[0]['timestamp']

            row_dict = {'open': open_, 'high': high_, 'low': low_, 'close': close_,
                        'timestamp': timestamp_}

            if 'oi' in gran_calc_list[-1]:
                row_dict['oi'] = sum([i["oi"] for i in gran_calc_list])
            if 'volume' in gran_calc_list[-1]:
                row_dict['volume'] = sum([i["volume"] for i in gran_calc_list])
            if 'previous_day_close' in gran_calc_list[0]:
                row_dict['previous_day_close'] = gran_calc_list[0]['previous_day_close']
            if 'curr_day_open' in gran_calc_list[0]:
                row_dict['curr_day_open'] = gran_calc_list[0]['curr_day_open']
            if 'symbol' in gran_calc_list[0]:
                row_dict['symbol'] = gran_calc_list[0]['symbol']
            if 'date' in gran_calc_list[0]:
                row_dict['date'] = gran_calc_list[0]['date']
            if 'underlying_stock' in gran_calc_list[0]:
                row_dict['underlying_stock'] = gran_calc_list[0]['underlying_stock']
            if 'strike' in gran_calc_list[0]:
                row_dict['strike'] = gran_calc_list[0]['strike']
            if 'option_type' in gran_calc_list[0]:
                row_dict['option_type'] = gran_calc_list[0]['option_type']
            if 'expiry' in gran_calc_list[0]:
                row_dict['expiry'] = gran_calc_list[0]['expiry']

            return row_dict

        for symbol in symbols:
            if symbol != 0:
                sym_data_df = data_df[data_df['symbol'] == symbol]
            else:
                sym_data_df = data_df

            # sym_data_df = sym_data_df.sort_values('timestamp', inplace=True)
            sorted_sym_data_df = sym_data_df.sort_values('timestamp')

            data_li = sorted_sym_data_df.to_dict('records')

            st_time = data_li[0]['timestamp'].time()

            gran_rows = list()
            gran_calc_list = list()

            for row in data_li:
                if row['timestamp'].time() == st_time:
                    if gran_calc_list:
                        row_dict = create_row_dict(gran_calc_list)

                        gran_rows.append(row_dict)

                    gran_calc_list = list()

                gran_calc_list.append(row)

                if len(gran_calc_list) == granularity:
                    row_dict = create_row_dict(gran_calc_list)

                    gran_rows.append(row_dict)
                    gran_calc_list = list()

            if gran_calc_list:
                row_dict = create_row_dict(gran_calc_list)

                gran_rows.append(row_dict)

            new_df = pd.DataFrame(gran_rows)
            new_df.dropna(subset=['close'], inplace=True)
            final_data_li.extend(new_df.to_dict('records'))

        final_df = pd.DataFrame(final_data_li)

    except Exception as e:
        final_df = data_df
        log_message = f"Exception in changing the granularity of provided dataframe.\nReturning the same dataframe\n{e}\n"
        print(log_message)

    return final_df


def calculate_swing(df, swing_setup=2, ignore_last_bar=False):
    """ Function to calculate swing """

    def func_upswing():
        """ Checks up swing continuation conditions """

        nonlocal data, horizontal_line, prev_candle, current_candle, swings, outside_bars, prev_idx, current_idx, p_hlines, p_swings

        try:
            _outsidebar_hh_ = max(*data[data.index(horizontal_line): current_idx], horizontal_line,
                                  key=lambda bar: bar['High'])
        except:
            _outsidebar_hh_ = swings[-1][0]

        # If Outside bar and curr_candle makes highest of high and close below prev_candle low then convert to up swing
        if (current_candle['High'] > prev_candle['High'] and current_candle['Low'] < prev_candle['Low']) and \
                (current_candle['Close'] < prev_candle['Low']) and (current_candle['High'] > _outsidebar_hh_['High']):
            p_hlines.append([horizontal_line, data[data.index(horizontal_line) + 1], "DOWN"])
            try:
                p_swings.append(
                    [max(*data[data.index(horizontal_line): current_idx + 1], horizontal_line,
                         key=lambda bar: bar['High']), "DOWN"])
            except:
                pass
            outside_bars.append([current_candle, "DOWN"])
            swings.append([current_candle, "DOWN"])
            horizontal_line = max(data[prev_idx - swing_setup: current_idx + 1], key=lambda bar: bar['High'])
            return horizontal_line, "DOWN"

        # Check swing break means down swing starts
        if current_candle['Close'] < horizontal_line['Low']:
            p_hlines.append([horizontal_line, data[data.index(horizontal_line) + 1], "DOWN"])
            try:
                p_swings.append(
                    [max(*data[data.index(horizontal_line): current_idx + 1], horizontal_line,
                         key=lambda bar: bar['High']), "DOWN"])
            except:
                pass

            swings.append([current_candle, "DOWN"])
            horizontal_line = max(data[prev_idx - swing_setup: current_idx + 1], key=lambda bar: bar['High'])
            return horizontal_line, "DOWN"

        # Check is there any upswing shift
        if current_candle['High'] > swings[-1][0]['High']:
            swings.append([current_candle, "UP"])
            horizontal_line = min(data[prev_idx - swing_setup: current_idx + 1], key=lambda bar: bar['Low'])
            return horizontal_line, "UP"

        # Return Up swing continues because no condition met, swing just rolling
        return horizontal_line, "UP"

    def func_downswing():
        """ Checks down swing continuation conditions """

        nonlocal data, horizontal_line, prev_candle, current_candle, swings, outside_bars, prev_idx, current_idx, p_hlines, p_swings

        try:
            _outsidebar_ll_ = min(*data[data.index(horizontal_line): current_idx], horizontal_line,
                                  key=lambda bar: bar['Low'])
        except:
            _outsidebar_ll_ = swings[-1][0]

        # If Outside bar and curr_candle makes lowest of low and close above prev_candle high then convert to up swing
        if (current_candle['High'] > prev_candle['High'] and current_candle['Low'] < prev_candle['Low']) and \
                (current_candle['Close'] > prev_candle['High']) and (current_candle['Low'] < _outsidebar_ll_['Low']):
            p_hlines.append([horizontal_line, data[data.index(horizontal_line) + 1], "UP"])
            try:
                p_swings.append(
                    [min(*data[data.index(horizontal_line): current_idx + 1], horizontal_line,
                         key=lambda bar: bar['Low']), "UP"])
            except:
                pass
            outside_bars.append([current_candle, "UP"])
            swings.append([current_candle, "UP"])
            horizontal_line = min(data[prev_idx - swing_setup: current_idx + 1], key=lambda bar: bar['Low'])
            return horizontal_line, "UP"

        # Check swing break means up swing starts
        if current_candle['Close'] > horizontal_line['High']:
            p_hlines.append([horizontal_line, data[data.index(horizontal_line) + 1], "UP"])
            try:
                p_swings.append([min(*data[data.index(horizontal_line): current_idx + 1], horizontal_line, key=lambda bar: bar['Low']), "UP"])
            except Exception as e:
                print("exception here : {}".format(e))
                pass

            swings.append([current_candle, "UP"])
            horizontal_line = min(data[prev_idx - swing_setup: current_idx + 1], key=lambda bar: bar['Low'])
            return horizontal_line, "UP"

        # Check is there any downswing shift
        if current_candle['Low'] < swings[-1][0]['Low']:
            swings.append([current_candle, 'DOWN'])
            horizontal_line = max(data[prev_idx - swing_setup: current_idx + 1], key=lambda bar: bar['High'])
            return horizontal_line, "DOWN"

        # Return Down swing continues because no condition met, swing just rolling
        return horizontal_line, "DOWN"

    def func_check_swing_chg():
        if len(swings) >= 2:
            if swings[-2][-1] != swings[-1][-1] and swings[-1][0]['timestamp'] == current_candle['timestamp']:
                return True
        return False

    try:
        # Initialization
        swing_setup -= 2
        prev_idx = 0
        current_idx = 1
        horizontal_line = ""
        current_swing_type = ""
        swings = []
        p_hlines = []
        p_swings = []
        outside_bars = []

        # Formatting data
        df.reset_index(inplace=True)
        df.rename({"datetime": "timestamp", "open": "Open", "high": "High", "low": "Low", "close": "Close"}, axis=1,
                  inplace=True)
        df['swing'], df['swing_change'] = '', False

        # Avoiding current day bar because its not completely build [Keep `False` while backtesting only]
        if ignore_last_bar:
            df = df.iloc[:-1, :]

        data = df.to_dict('index')
        data = [data[i] for i in sorted(list(data.keys()))]

        # Start Swing creation
        while current_idx < (len(data)):
            prev_candle = data[prev_idx]
            current_candle = data[current_idx]
            # Wait till minimum bars built
            if prev_idx <= swing_setup:
                prev_idx += 1
                current_idx += 1
                continue

            ##################################################
            # Swing Initialization
            if horizontal_line == "":
                horizontal_line = prev_candle
                upside_horizontal_line = max(data[prev_idx - swing_setup: prev_idx + 1], key=lambda x: x['High'])
                downside_horizontal_line = min(data[prev_idx - swing_setup: prev_idx + 1], key=lambda x: x['Low'])

            if current_swing_type == "":
                if current_candle['Close'] > horizontal_line['High']:
                    swings.append([upside_horizontal_line, "UP"])
                    horizontal_line = min(data[prev_idx - swing_setup: current_idx + 1], key=lambda bar: bar['Low'])
                    current_swing_type = "UP"
                elif current_candle['Close'] < horizontal_line['Low']:
                    swings.append([downside_horizontal_line, "DOWN"])
                    horizontal_line = max(data[prev_idx - swing_setup: current_idx + 1], key=lambda bar: bar['High'])
                    current_swing_type = "DOWN"
            ##################################################

            # Continue Swing building
            if current_swing_type == "UP":
                # Continue loop till swing change
                while current_swing_type != "DOWN":
                    if horizontal_line != current_candle:
                        horizontal_line, current_swing_type = func_upswing()

                    # Update df
                    df.loc[current_idx, 'swing'] = current_swing_type
                    df.loc[current_idx, 'swing_change'] = func_check_swing_chg()

                    prev_idx += 1
                    current_idx += 1

                    # Checking whether the last candle is current candle
                    if len(data) == current_idx:
                        break
                    else:
                        prev_candle = data[prev_idx]
                        current_candle = data[current_idx]

            elif current_swing_type == "DOWN":
                # Continue loop till swing change
                while current_swing_type != "UP":
                    if horizontal_line != current_candle:
                        horizontal_line, current_swing_type = func_downswing()

                    # Update df
                    df.loc[current_idx, 'swing'] = current_swing_type
                    df.loc[current_idx, 'swing_change'] = func_check_swing_chg()

                    prev_idx += 1
                    current_idx += 1

                    if len(data) == current_idx:
                        break
                    else:
                        prev_candle = data[prev_idx]
                        current_candle = data[current_idx]
            else:
                prev_idx += 1
                current_idx += 1
        df.rename({"datetime": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close"}, axis=1,
                  inplace=True)
        return df
    except Exception as e:
        print("Exception in Swing Calculation : {}".format(e))
        return df


def calculate_rolling_swing(df):
    """ Function to calculate Rolling Swing """
    try:
        print("Calculating Rolling Swing")
        swing_list = []
        candles = df.to_dict('records')
        last_swing = None
        for indx, candle in enumerate(candles):
            if indx >= 2:
                prev_candle_1 = candles[indx - 1]
                prev_candle_2 = candles[indx - 2]

                # Conditions for Upswing
                up_condition = candle["open"] > prev_candle_1["close"] and candle["open"] > prev_candle_2["close"]

                # Conditions for DownSwing
                down_condition = candle["open"] < prev_candle_1["close"] and candle["open"] < prev_candle_2["close"]

                if up_condition:
                    current_swing = "UP"
                    if current_swing != last_swing:
                        last_swing = "UP"
                        swing_change = True
                    else:
                        swing_change = False

                elif down_condition:
                    current_swing = "DOWN"
                    if current_swing != last_swing:
                        last_swing = "DOWN"
                        swing_change = True
                    else:
                        swing_change = False
                else:
                    swing_change = False
                    current_swing = last_swing

                candle["swing"] = current_swing
                candle["swing_change"] = swing_change

                swing_list.append(candle)
            else:
                candle["swing"] = None
                candle["swing_change"] = False
                swing_list.append(candle)

        # Converting Swings Candles list to Dataframe
        swing_df = pd.DataFrame(swing_list)
        return swing_df
    except Exception as e:
        print("Exception in calculating rolling swing : {}".format(e))
        pass


def apply_indicators(dataframe, indicators_dict, logger=None):
    """ Function to apply indicators on the dataframe provided """
    try:
        if isinstance(indicators_dict, tuple):
            indicators_dict = indicators_dict[0]

        if "ATR" in indicators_dict:
            atr_values = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=indicators_dict["ATR"]["length"],
                                mamode=indicators_dict["ATR"]["mamode"])
            atr_values = round(atr_values, 2).fillna(0)
            dataframe["atr"] = atr_values.values.tolist()

        if "ADX" in indicators_dict:
            adx_values = ta.adx(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=indicators_dict["ADX"]["length"],
                                lensig=indicators_dict["ADX"]["lensig"])
            dataframe["adx"] = round(adx_values.iloc[:, 0], 2).fillna(0).values
            dataframe["adx_dmp"] = round(adx_values.iloc[:, 1], 2).fillna(0).values
            dataframe["adx_dmn"] = round(adx_values.iloc[:, 2], 2).fillna(0).values

        if "EMA" in indicators_dict:
            ema_values = ta.ema(dataframe["close"], length=indicators_dict["EMA"]["length"])
            ema_values = round(ema_values, 2).fillna(0)
            dataframe["ema"] = ema_values.values.tolist()

        if "EMA5" in indicators_dict:
            input_series = dataframe["close"]

            if indicators_dict["EMA5"]["source"] == "close":
                input_series = dataframe["close"]

            elif indicators_dict["EMA5"]["source"] == "open":
                input_series = dataframe["open"]

            elif indicators_dict["EMA5"]["source"] == "hlc3":
                input_series = (dataframe['high'] + dataframe["low"] + dataframe["close"])/3

            ema5_values = ta.ema(input_series, length=indicators_dict["EMA5"]["length"])
            ema5_values = round(ema5_values, 2).fillna(0)
            dataframe["ema5"] = ema5_values.values.tolist()

        if "EMA9" in indicators_dict:
            ema9_values = ta.ema(dataframe["close"], length=indicators_dict["EMA9"]["length"])
            ema9_values = round(ema9_values, 2).fillna(0)
            dataframe["ema9"] = ema9_values.values.tolist()

        if "EMA12" in indicators_dict:
            ema12_values = ta.ema(dataframe["close"], length=indicators_dict["EMA12"]["length"])
            ema12_values = round(ema12_values, 2).fillna(0)
            dataframe["ema12"] = ema12_values.values.tolist()

        if "EMA15" in indicators_dict:
            ema15_values = ta.ema(dataframe["close"], length=indicators_dict["EMA15"]["length"])
            ema15_values = round(ema15_values, 2).fillna(0)
            dataframe["ema15"] = ema15_values.values.tolist()

        if "EMA20" in indicators_dict:
            ema20_values = ta.ema(dataframe["close"], length=indicators_dict["EMA20"]["length"])
            ema20_values = round(ema20_values, 2).fillna(0)
            dataframe["ema20"] = ema20_values.values.tolist()

        if "EMA50" in indicators_dict:
            ema50_values = ta.ema(dataframe["close"], length=indicators_dict["EMA50"]["length"])
            ema50_values = round(ema50_values, 2).fillna(0)
            dataframe["ema50"] = ema50_values.values.tolist()

        if "EMA60" in indicators_dict:
            ema60_values = ta.ema(dataframe["close"], length=indicators_dict["EMA60"]["length"])
            ema60_values = round(ema60_values, 2).fillna(0)
            dataframe["ema60"] = ema60_values.values.tolist()

        if "SMA" in indicators_dict:
            sma_values = ta.sma(dataframe["close"], length=indicators_dict["SMA"]["length"])
            sma_values = round(sma_values, 2).fillna(0)
            dataframe["sma"] = sma_values.values.tolist()

        if "SMA5" in indicators_dict:
            sma_values = ta.sma(dataframe["close"], length=indicators_dict["SMA5"]["length"])
            sma_values = round(sma_values, 2).fillna(0)
            dataframe["sma5"] = sma_values.values.tolist()

        if "SMA9" in indicators_dict:
            sma_values = ta.sma(dataframe["close"], length=indicators_dict["SMA9"]["length"])
            sma_values = round(sma_values, 2).fillna(0)
            dataframe["sma9"] = sma_values.values.tolist()

        if "SMA15" in indicators_dict:
            sma_values = ta.sma(dataframe["close"], length=indicators_dict["SMA15"]["length"])
            sma_values = round(sma_values, 2).fillna(0)
            dataframe["sma15"] = sma_values.values.tolist()

        if "SMA20" in indicators_dict:
            sma_values = ta.sma(dataframe["close"], length=indicators_dict["SMA20"]["length"])
            sma_values = round(sma_values, 2).fillna(0)
            dataframe["sma20"] = sma_values.values.tolist()

        if "SMA200" in indicators_dict:
            sma_values = ta.sma(dataframe["close"], length=indicators_dict["SMA200"]["length"])
            if sma_values is not None:
                sma_values = round(sma_values, 2).fillna(0)
                dataframe["sma200"] = sma_values.values.tolist()
            else:
                dataframe["sma200"] = 0

        if "ECR" in indicators_dict:
            ecr_values = dataframe["ema"] - dataframe["ema70"]
            ecr_values = round(ecr_values, 2).fillna(0)
            dataframe["ecr"] = ecr_values.tolist()

        if "STOCH" in indicators_dict:
            stoch_df = stochastic_oscillator(dataframe=dataframe, k=indicators_dict["STOCH"]["k"], d=indicators_dict["STOCH"]["d"],
                                             smooth_k=indicators_dict["STOCH"]["smooth"])
            # stoch_df = TA.stoch(high=dataframe["high"], low=dataframe["low"], close=dataframe["close"],
            #                     k=indicators_dict["STOCH"]["k"], d=indicators_dict["STOCH"]["d"], smooth_k=indicators_dict["STOCH"]["smooth"], offset=0)
            stoch_k = round(stoch_df.iloc[:, 0], 2).fillna(0).values
            stoch_d = round(stoch_df.iloc[:, 1], 2).fillna(0).values
            dataframe["stoch_k"] = stoch_k
            dataframe["stoch_d"] = stoch_d

        if "RSI" in indicators_dict:
            rsi_df = ta.rsi(close=dataframe.close, length=indicators_dict["RSI"]["length"])
            dataframe["rsi"] = round(rsi_df.iloc[:], 2).fillna(0).values

        if "TVRSI" in indicators_dict:
            tv_rsi_df = tv_rsi(dataframe=dataframe, rsi_length=indicators_dict["TVRSI"]["rsi_length"], source=indicators_dict["TVRSI"]["source"],
                               ma_length=indicators_dict["TVRSI"]["ma_length"], ma_type=indicators_dict["TVRSI"]["ma_type"],
                               bb_mult=indicators_dict["TVRSI"]["bb_mult"])
            dataframe["tv_rsi"] = round(tv_rsi_df['rsi'], 2).fillna(0).values
            dataframe["tv_rsi_ma"] = round(tv_rsi_df['rsi_ma'], 2).fillna(0).values

        if "STOCHRSI" in indicators_dict:
            out = ta.stochrsi(dataframe[indicators_dict["STOCHRSI"]["rsi_source"]], length=indicators_dict["STOCHRSI"]["stochastic_length"], k=indicators_dict["STOCHRSI"]["k"], d=indicators_dict["STOCHRSI"]["d"],
                              rsi_length=indicators_dict["STOCHRSI"]["rsi_length"])
            dataframe["stochrsi_k"] = round(out.iloc[:, 0], 2).fillna(0).values
            dataframe["stochrsi_d"] = round(out.iloc[:, 1], 2).fillna(0).values

        if "MACD" in indicators_dict:
            macd_df = ta.macd(close=dataframe[indicators_dict["MACD"]["source"]], fast=indicators_dict["MACD"]["fast_length"],
                              slow=indicators_dict["MACD"]["slow_length"], signal=indicators_dict["MACD"]["signal_smoothing"])
            dataframe["macd"] = round(macd_df.iloc[:, 0], 2).fillna(0).values
            dataframe["macd_h"] = round(macd_df.iloc[:, 1], 2).fillna(0).values
            dataframe["macd_s"] = round(macd_df.iloc[:, 2], 2).fillna(0).values

        if "BBAND" in indicators_dict:
            bband_df = ta.bbands(close=dataframe[indicators_dict["BBAND"]["source"]], length=indicators_dict["BBAND"]["length"], std=indicators_dict["BBAND"]["std_dev"],
                                 mamode=indicators_dict["BBAND"]["basis_ma_type"], offset=indicators_dict["BBAND"]["offset"])
            dataframe["bband_l"] = round(bband_df.iloc[:, 0], 2).fillna(0).values
            dataframe["bband_m"] = round(bband_df.iloc[:, 1], 2).fillna(0).values
            dataframe["bband_u"] = round(bband_df.iloc[:, 2], 2).fillna(0).values
            dataframe["bband_b"] = round(bband_df.iloc[:, 3], 2).fillna(0).values
            dataframe["bband_p"] = round(bband_df.iloc[:, 4], 2).fillna(0).values

        if "FISHER" in indicators_dict:
            fisher_df = ta.fisher(high=dataframe["high"], low=dataframe["low"], length=indicators_dict["FISHER"]["length"], signal=indicators_dict["FISHER"]["signal"])
            dataframe["fisher_t"] = round(fisher_df.iloc[:, 0], 2).fillna(0).values
            dataframe["fisher_s"] = round(fisher_df.iloc[:, 1], 2).fillna(0).values

        if "MOM" in indicators_dict:
            mom_df = ta.mom(close=dataframe[indicators_dict["MOM"]["source"]], length=indicators_dict["MOM"]["length"])
            dataframe["mom"] = round(mom_df.iloc[:], 2).fillna(0).values

        if "ROC" in indicators_dict:
            roc_df = ta.roc(close=dataframe[indicators_dict["ROC"]["source"]], length=indicators_dict["ROC"]["length"])
            dataframe["roc"] = round(roc_df.iloc[:], 2).fillna(0).values

        if "DPO" in indicators_dict:
            dpo_df = ta.dpo(close=dataframe["close"], length=indicators_dict["DPO"]["length"])
            dataframe["dpo"] = round(dpo_df.iloc[:], 2).fillna(0).values

        if "RVGI" in indicators_dict:
            rvgi_df = ta.rvgi(open_=dataframe["open"], high=dataframe["high"], low=dataframe["low"], close=dataframe["close"], length=indicators_dict["RVGI"]["length"],)
            dataframe["rvgi"] = round(rvgi_df.iloc[:, 0], 2).fillna(0).values
            dataframe["rvgi_s"] = round(rvgi_df.iloc[:, 1], 2).fillna(0).values

        if "WILLR" in indicators_dict:
            willr_df = ta.willr(high=dataframe["high"], low=dataframe["low"], close=dataframe["close"], length=indicators_dict["WILLR"]["length"])
            dataframe["willr"] = round(willr_df.iloc[:], 2).fillna(0).values

        if "VFI" in indicators_dict:
            vfi_df = volume_flow_indicator(dataframe=dataframe, length=indicators_dict["VFI"]["length"], coef=indicators_dict["VFI"]["coef"],
                                           vcoef=indicators_dict["VFI"]["vcoef"], signal_length=indicators_dict["VFI"]["signal_length"],
                                           smooth_vfi=indicators_dict["VFI"]["smooth_vfi"])
            dataframe["vfi"] = round(vfi_df['vfi'], 2).fillna(0).values
            dataframe["vfi_ma"] = round(vfi_df['vfi_ma'], 2).fillna(0).values
            dataframe["vfi_d"] = round(vfi_df['vfi_d'], 2).fillna(0).values

        if "VORTEX" in indicators_dict:
            vortex_df = ta.vortex(high=dataframe["high"], low=dataframe["low"], close=dataframe["close"], length=indicators_dict["VORTEX"]["length"])
            dataframe["vtx_p"] = round(vortex_df.iloc[:, 0], 2).fillna(0).values
            dataframe["vtx_m"] = round(vortex_df.iloc[:, 1], 2).fillna(0).values

        if "SUPERTREND" in indicators_dict:
            supertrend_df = ta.supertrend(high=dataframe["high"], low=dataframe["low"], close=dataframe["close"], length=indicators_dict["SUPERTREND"]["length"], multiplier=indicators_dict["SUPERTREND"]["multiplier"])
            dataframe["supertrend"] = round(supertrend_df.iloc[:, 0], 2).fillna(0).values
            dataframe["supertrend_d"] = round(supertrend_df.iloc[:, 1], 2).fillna(0).values

        if "FRACTAL" in indicators_dict:
            dataframe = williams_fractal(dataframe, fractal_window=indicators_dict["FRACTAL"]["length"])
            dataframe["fractal_up"] = dataframe["fractal_up"].fillna(0).values
            dataframe["fractal_down"] = dataframe["fractal_down"].fillna(0).values

        if "TSI" in indicators_dict:
            tsi_df = ta.tsi(close=dataframe[indicators_dict["TSI"]["source"]], fast=indicators_dict["TSI"]["fast"], slow=indicators_dict["TSI"]["slow"], signal=indicators_dict["TSI"]["signal"])
            dataframe["tsi"] = round(tsi_df.iloc[:, 0]/100, 4).fillna(0).values
            dataframe["tsi_s"] = round(tsi_df.iloc[:, 1]/100, 4).fillna(0).values

        if "SMI" in indicators_dict:
            smi_df = ta.smi(close=dataframe[indicators_dict["SMI"]["source"]], fast=indicators_dict["SMI"]["fast"], slow=indicators_dict["SMI"]["slow"], signal=indicators_dict["SMI"]["signal"])
            dataframe["smi"] = round(smi_df.iloc[:, 0], 2).fillna(0).values
            dataframe["smi_s"] = round(smi_df.iloc[:, 1], 2).fillna(0).values
            dataframe["smi_o"] = round(smi_df.iloc[:, 2], 2).fillna(0).values

        if "MCGD" in indicators_dict:
            mcgd_df = mcgd(close=dataframe["close"], length=indicators_dict["MCGD"]["length"])
            dataframe["mcgd"] = round(mcgd_df.iloc[:], 2).fillna(0).values

        if "TPR" in indicators_dict:
            dataframe["tpr"] = round((dataframe["high"] + dataframe["low"] + dataframe["close"])/3, 2)

        if "MAC" in indicators_dict:
            mac_u = ta.sma(close=dataframe["high"], length=indicators_dict["MAC"]["upper_length"])
            mac_l = ta.sma(close=dataframe["low"], length=indicators_dict["MAC"]["lower_length"])
            dataframe["mac_u"] = round(mac_u, 2).fillna(0).values
            dataframe["mac_l"] = round(mac_l, 2).fillna(0).values

        if "SQZMOM" in indicators_dict:
            sqzmom_df = squeeze_momentum(dataframe=dataframe, bb_length=indicators_dict["SQZMOM"]["bb_length"], bb_mult=indicators_dict["SQZMOM"]["bb_mult"],
                                         kc_length=indicators_dict["SQZMOM"]["kc_length"], kc_mult=indicators_dict["SQZMOM"]["kc_mult"],use_true_range=indicators_dict["SQZMOM"]["use_true_range"])
            dataframe["sqzmom"] = round(sqzmom_df, 2).fillna(0).values

        if "CHAIKIN" in indicators_dict:
            cv_df = chaikin_volatility(dataframe=dataframe, length=indicators_dict["CHAIKIN"]["length"], roc_length=indicators_dict["CHAIKIN"]["roc_length"])
            dataframe["chaikin"] = round(cv_df.iloc[:], 2).fillna(0).values

        if "VWAP" in indicators_dict:
            # vwap_df = TA.vwap(high=dataframe["high"], low=dataframe["low"], close=dataframe["close"], volume=dataframe["volume"])
            vwap_df = calculate_vwap(dataframe=dataframe)
            dataframe["vwap"] = round(vwap_df.iloc[:], 2).fillna(0).values

        if "DMI" in indicators_dict:
            dmi_df = calculate_dmi(dataframe=dataframe, adx_smoothing=indicators_dict["DMI"]["adx_smoothing"], di_length=indicators_dict["DMI"]["di_length"])
            dataframe["di_plus"] = round(dmi_df['di_plus'], 2).fillna(0).values
            dataframe["di_minus"] = round(dmi_df['di_minus'], 2).fillna(0).values
            dataframe["di_adx"] = round(dmi_df['di_adx'], 2).fillna(0).values

        if "YONO" in indicators_dict:
            # dc_df = direction_change(dataframe=dataframe, depth=indicators_dict["DC"]["depth"], deviation=indicators_dict["DC"]["deviation"], backstep=indicators_dict["DC"]["backstep"])
            yono_df = yono(dataframe=dataframe, depth=indicators_dict["YONO"]["depth"], deviation=indicators_dict["YONO"]["deviation"], backstep=indicators_dict["YONO"]["backstep"])
            # yono_df = calculate_direction(dataframe=dataframe, depth=indicators_dict["YONO"]["depth"], deviation=indicators_dict["YONO"]["deviation"], backstep=indicators_dict["YONO"]["backstep"])
            dataframe["yono"] = round(yono_df.iloc[:], 2).fillna(0).values

        if "DEMA" in indicators_dict:
            dema_df = ta.dema(close=dataframe[indicators_dict["DEMA"]["source"]], length=indicators_dict["DEMA"]["length"])
            dataframe["dema"] = round(dema_df.iloc[:], 2).fillna(0).values

        if "TEMA" in indicators_dict:
            tema_df = ta.tema(close=dataframe[indicators_dict["TEMA"]["source"]], length=indicators_dict["TEMA"]["length"])
            dataframe["tema"] = round(tema_df.iloc[:], 2).fillna(0).values

        if "RVI" in indicators_dict:
            rvi_df = ta.rvi(close=dataframe["close"], high=dataframe["high"], low=dataframe["low"], length=indicators_dict["RVI"]["length"])
            dataframe["rvi"] = round(rvi_df.iloc[:], 2).fillna(0).values

        if "SHA" in indicators_dict:
            sha_df = heikin_ashi_smoothed(dataframe=dataframe, ema_length=indicators_dict["SHA"]["length"])
            dataframe["sha_1"] = round(sha_df.iloc[:, 0], 2).fillna(0).values
            dataframe["sha_2"] = round(sha_df.iloc[:, 1], 2).fillna(0).values

        if "JMA" in indicators_dict:
            jma_df = jurik_moving_average(dataframe=dataframe, length=indicators_dict["JMA"]["length"], phase=indicators_dict["JMA"]["phase"],
                                          power=indicators_dict["JMA"]["power"], source=indicators_dict["JMA"]["source"])
            dataframe["jma"] = round(jma_df.iloc[:, 0], 2).fillna(0).values
            dataframe["jma_c"] = jma_df.iloc[:, 1].fillna(0).values

        if "EFI" in indicators_dict:
            efi_df = ta.efi(close=dataframe["close"], volume=dataframe["volume"], length=indicators_dict["EFI"]["length"])
            dataframe["efi"] = round(efi_df, 2).fillna(0).values

        if "PSAR" in indicators_dict:
            psar_df = ta.psar(high=dataframe["high"], low=dataframe["low"], close=dataframe["close"], af0=indicators_dict["PSAR"]["start"],
                              af=indicators_dict["PSAR"]["increment"], max_af=indicators_dict["PSAR"]["max_value"])
            dataframe["psar_l"] = round(psar_df.iloc[:, 0], 2).fillna(0).values
            dataframe["psar_s"] = round(psar_df.iloc[:, 1], 2).fillna(0).values
            dataframe["psar_af"] = round(psar_df.iloc[:, 2], 2).fillna(0).values
            dataframe["psar_r"] = round(psar_df.iloc[:, 3], 2).fillna(0).values

        if "AVGVOL" in indicators_dict:
            avg_vol = average_volume(dataframe["volume"], length=indicators_dict["AVGVOL"]["length"])
            dataframe["avg_vol"] = round(avg_vol, 2).fillna(0).values

        if "BBPS" in indicators_dict:

            bbps_df = bb_sideways(dataframe=dataframe, bb_length=indicators_dict["BBPS"]["bb_length"], bb_mult=indicators_dict["BBPS"]["bb_mult"],
                                  bbr_len=indicators_dict["BBPS"]["bbr_len"], bbr_std_thresh=indicators_dict["BBPS"]["bbr_std_thresh"])
            dataframe["bbps_sideways"] = bbps_df.iloc[:, 0].fillna(False).values
            dataframe["bbps_color"] = bbps_df.iloc[:, 1].fillna(False).values

        if "RDX" in indicators_dict:
            rdx_df = calculate_rdx(dataframe=dataframe)
            dataframe["rdx"] = round(rdx_df.iloc[:, 0]).values

        if "MTI" in indicators_dict:
            mti_df = calculate_mti(dataframe=dataframe, bb_length=indicators_dict["MTI"]["bb_length"], bb_mult=indicators_dict["MTI"]["bb_mult"],
                                   adx_length=indicators_dict["MTI"]["adx_length"], rsi_length=indicators_dict["MTI"]["rsi_length"])

        if "SADX" in indicators_dict:
            s_adx = smoothed_adx(dataframe=dataframe, adx_length=indicators_dict["SADX"]["adx_length"], di_length=indicators_dict["SADX"]["di_length"],
                                 smoothing_length=indicators_dict["SADX"]["smoothing_length"], mamode=indicators_dict["SADX"]["mamode"])
            dataframe["adx"] = round(s_adx.iloc[:, 0], 2).fillna(0).values
            dataframe["s_adx"] = round(s_adx.iloc[:, 1], 2).fillna(0).values

        if "NETVOLUME" in indicators_dict:
            net_volume = calculate_net_volume(dataframe=dataframe)
            dataframe["net_volume"] = round(net_volume, 2).fillna(0).values

        if "PPO" in indicators_dict:
            ppo_df = ta.ppo(close=dataframe["close"], fast=indicators_dict["PPO"]["fast_length"],
                            slow=indicators_dict["PPO"]["slow_length"])
            dataframe["ppo"] = round(ppo_df.iloc[:, 0], 2).fillna(0).values
            dataframe["ppo_h"] = round(ppo_df.iloc[:, 1], 2).fillna(0).values
            dataframe["ppo_s"] = round(ppo_df.iloc[:, 2], 2).fillna(0).values

        if "VOSC" in indicators_dict:
            v_osc = volume_oscillator(dataframe=dataframe, short_length=indicators_dict["VOSC"]["short_length"],
                                      long_length=indicators_dict["VOSC"]["long_length"])
            dataframe["v_osc"] = round(v_osc, 2).fillna(0).values

        if "VOLUMESLOPE" in indicators_dict:
            vol_slope_df = volume_slope(volume=dataframe["volume"], length=indicators_dict["VOLUMESLOPE"]["length"])
            dataframe["vol_ma"] = round(vol_slope_df.iloc[:, 0], 2).fillna(0).values
            dataframe["vol_slope"] = vol_slope_df.iloc[:, 1]

        if "OISLOPE" in indicators_dict:
            oi_slope_df = oi_slope(oi=dataframe["oi"], length=indicators_dict["OISLOPE"]["length"])
            dataframe["oi_ma"] = round(oi_slope_df.iloc[:, 0], 2).fillna(0).values
            dataframe["oi_slope"] = oi_slope_df.iloc[:, 1]

        if "BBW" in indicators_dict:
            bbw_df = bollinger_bandwidth(dataframe=dataframe, length=indicators_dict["BBW"]["length"], std_dev=indicators_dict["BBW"]["std_dev"],
                                         source=indicators_dict["BBW"]["source"], he_length=indicators_dict["BBW"]["he_length"],
                                         lc_length=indicators_dict["BBW"]["lc_length"])
            dataframe["bbw"] = round(bbw_df, 2).fillna(0).values

        if "BBWRANGE" in indicators_dict:
            bbw_range = calculate_bbw_range(dataframe=dataframe, length=indicators_dict["BBWRANGE"]["length"], deviation=indicators_dict["BBWRANGE"]["deviation"])
            dataframe["bbw_range"] = bbw_range

        if "TSI" in indicators_dict:
            tsi_df = calculate_tsi(close=dataframe["close"], period=indicators_dict["TSI"]["period"])
            dataframe["tsi"] = round(tsi_df, 2).fillna(0).values

        return dataframe

    except Exception as e:
        if logger:
            logger.exception(f"Exception in adding indicators : {e}")
        print("Exception in adding indicators : {}".format(e))
        pass


def create_mt_report(trade_report, initial_deposit, file_name, logs_coll):
    """Function to Create Report in MetaTrader Report Format"""
    try:
        if not trade_report:
            return {}

        report_df = pd.DataFrame(trade_report)

        mt_dict = dict()
        mt_dict['initial_deposit'] = initial_deposit

        profit_trades = list(report_df[report_df['pnl'] > 0]['pnl'])
        loss_trades = list(report_df[report_df['pnl'] < 0]['pnl'])

        # Calculating Gross Profit
        gross_profit = round(sum(profit_trades), 2) if profit_trades else 0
        mt_dict['gross_profit'] = gross_profit

        # Calculating Gross Loss
        gross_loss = round(sum(loss_trades), 2) if loss_trades else 0
        mt_dict['gross_loss'] = gross_loss

        # Calculating Profit Factor
        mt_dict['profit_factor'] = round(abs(gross_profit / gross_loss), 2) if gross_loss else 0

        profitside = (len(profit_trades) / len(report_df)) * (gross_profit / len(profit_trades)) if profit_trades else 0
        lossside = (len(loss_trades) / len(report_df)) * (gross_loss / len(loss_trades)) if loss_trades else 0

        expected_payoff = round(profitside - lossside, 2)
        mt_dict['expected_payoff'] = expected_payoff

        equity_history = [initial_deposit]

        # report_pnl = [initial_deposit]
        # dates = [list(report_df['date'])[0]] + sorted(set(report_df['date']))

        report_df['date'] = pd.to_datetime(report_df['date'], format="%Y-%m-%d %H:%M:%S")

        eq_dates = [list(report_df['date'])[0].strftime('%Y-%m-%d')] + [i.strftime('%Y-%m-%d') for i in sorted(set(report_df['date']))]

        report_dt_group = report_df.groupby('date')

        for grp in report_dt_group.groups:
            equity_history.append(equity_history[-1]+sum(list(report_dt_group.get_group(grp)['pnl'])))

        # for pl in report_df['pnl']:
        #     equity_history.append(equity_history[-1] + pl)

        mt_dict['equity_history'] = list(zip(eq_dates, equity_history))
        # mt_dict['equity_history'] = list(zip(dates, report_pnl))

        # Calculating Absolute Drawdown
        min_drawdown = round(equity_history[0] - min(equity_history), 2)
        mt_dict['abs_drawdown'] = min_drawdown

        diff_points = list()

        if len(equity_history) >= 2:
            for prev_idx, curr_balance in enumerate(equity_history[1:], start=0):
                prev_balance = equity_history[prev_idx]
                # If the next balance is less, then only check that
                if prev_balance >= curr_balance:
                    difference = prev_balance - curr_balance
                    diff_points.append([prev_balance, curr_balance, difference])

        if diff_points:
            difference_li = [val[-1] for val in diff_points]
            relative_points = [val[2] / val[0] for val in diff_points]

            max_drawdown = max(difference_li)
            max_drawdown_per = round((max_drawdown / diff_points[difference_li.index(max_drawdown)][0]) * 100, 2)

            relative_drawdown = round(diff_points[relative_points.index(max(relative_points))][2], 2)
            relative_drawdown_per = round(max(relative_points) * 100, 2)
        else:
            max_drawdown = 0
            max_drawdown_per = 0
            relative_drawdown = 0
            relative_drawdown_per = 0

        mt_dict['maximal_drawdown'] = max_drawdown
        mt_dict['maximal_drawdown%'] = max_drawdown_per

        mt_dict['relative_drawdown'] = relative_drawdown
        mt_dict['relative_drawdown_per'] = relative_drawdown_per

        mt_dict['total_trades'] = len(report_df)

        signal_cnts = report_df['trade_type'].value_counts().to_dict()

        if "LONG" not in signal_cnts:
            signal_cnts['LONG'] = 0
            long_pos_win = 0
        else:
            long_pos_win = round((len(report_df[report_df['trade_type'] == "LONG"][report_df['pnl'] > 0]) / len(report_df) * 100), 2)

        if "SHORT" not in signal_cnts:
            signal_cnts['SHORT'] = 0
            short_pos_win = 0
        else:
            short_pos_win = round((len(report_df[report_df['trade_type'] == "SHORT"][report_df['pnl'] > 0]) / len(report_df) * 100), 2)

        # # Short Position (Won%)
        mt_dict['short_position'] = signal_cnts['SHORT']
        mt_dict['short_position_won%'] = short_pos_win

        # # Long Position (Won%)
        mt_dict['long_position'] = signal_cnts['LONG']
        mt_dict['long_position_won%'] = long_pos_win

        # # Profit trades %
        mt_dict['profit_trades_num'] = len(profit_trades)
        mt_dict['profit_trades_num%'] = round((len(profit_trades) / len(report_df)) * 100, 2)

        # # Loss trades %
        mt_dict['loss_trades_num'] = len(loss_trades)
        mt_dict['loss_trades_num%'] = round((len(loss_trades) / len(report_df)) * 100, 2)

        # # The Largest profit trade
        if profit_trades:
            mt_dict['largest_profit_trade'] = max(profit_trades)
        else:
            mt_dict['largest_profit_trade'] = ""

        # # The Largest loss trade
        mt_dict['largest_loss_trade'] = min(loss_trades)

        # # Average Profit trade
        mt_dict['avg_profit_trade'] = round(gross_profit / len(profit_trades), 2)

        # # Average Loss trade
        mt_dict['avg_loss_trade'] = round(gross_loss / len(loss_trades), 2)

        consecutive_out = [1 if pnl > 0 else -1 for pnl in report_df['pnl']]
        cons_counts = [(x[0], len(list(x[1]))) for x in itertools.groupby(consecutive_out)]

        # Get Sum of those consecutive series
        final_cons_vals = []
        profit_loss_values = report_df['pnl'].values.tolist()
        st_idx = 0
        for (cons_type, cons_cnt) in cons_counts:
            final_cons_vals.append([cons_type, cons_cnt, profit_loss_values[st_idx: st_idx + cons_cnt]])
            st_idx += cons_cnt

        # Filter out the consecutive values
        cons_counts = {1: [], -1: []}

        for _ in final_cons_vals:
            if _[0] > 0:
                cons_counts[1].append(_)
            else:
                cons_counts[-1].append(_)
        # {cons_counts[1].append(_) if _[0] > 0 else cons_counts[-1].append(_) for _ in final_cons_vals}

        # # Maximum consecutive wins
        if len(cons_counts[1]) > 0:
            mt_dict['max_cons_wins_cnt'] = max(cons_counts[1], key=lambda x: x[1])[1]

            # # Maximal consecutive profit
            mt_dict['max_cons_wins_value'] = sum(max(cons_counts[1], key=lambda x: x[1])[2])

            # # Average consecutive wins
            mt_dict['avg_cons_wins'] = sum([val[1] for val in cons_counts[1]]) / len(cons_counts[1])
        else:
            mt_dict['max_cons_wins_cnt'] = 0
            mt_dict['max_cons_wins_value'] = 0
            mt_dict['avg_cons_wins'] = 0

        if len(cons_counts[-1]) > 0:
            # # Maximum consecutive loss
            mt_dict['max_cons_loss_cnt'] = max(cons_counts[-1], key=lambda x: x[1])[1]

            # # Maximal consecutive loss
            mt_dict['max_cons_loss_value'] = sum(max(cons_counts[-1], key=lambda x: x[1])[2])

            # # Average consecutive loss
            mt_dict['avg_cons_loss'] = sum([val[1] for val in cons_counts[-1]]) / len(cons_counts[-1])
        else:
            # # Maximum consecutive loss
            mt_dict['max_cons_loss_cnt'] = 0
            mt_dict['max_cons_loss_value'] = 0
            mt_dict['avg_cons_loss'] = 0

        for k, v in mt_dict.items():
            if type(v) in [np.float64, float]:
                mt_dict[k] = round(v, 2)

        return mt_dict

    except Exception as e:
        log_message = f"Exception in creating mt_report.\n{e}"
        print(log_message)
        mg_insert_log(logs_coll, file_name, 'ERROR', log_message, log_traceback=''.join(tb.format_exception(None, e, e.__traceback__)))
        return {}


def find_last_trading_date(trade_collection):
    """Function to Fetch Last Trading Date"""
    last_trading_date = None
    try:
        last_trading_date = list(trade_collection.find({'symbol': 'NIFTY 50'}).sort('date', -1).limit(1))[0]['date']
        return last_trading_date
    except Exception as e:
        print("Exception in fetching last trading date : {}".format(e))
        return last_trading_date


def mg_insert_log(collection, file_name, log_level, log_message, request_id=None, strategy_id=None, user_id=None, log_traceback=None, api_route=None):
    """ Function to Insert Logs in MongoDB Logs"""
    log_dict = {'api_route': api_route, 'request_id': request_id, 'strategy_id': strategy_id, 'user_id': user_id,
                'file_name': file_name, 'log_level': log_level.upper(), 'timestamp': datetime.datetime.now(),
                'message': log_message, 'traceback': log_traceback}

    if not request_id:
        del log_dict['request_id']
    if not strategy_id:
        del log_dict['strategy_id']
    if not user_id:
        del log_dict['user_id']
    if not log_traceback:
        del log_dict['traceback']
    if not api_route:
        del log_dict['api_route']

    collection.insert_one(log_dict)


def dates_list_from_start_end(start, end):
    delta = end - start
    days = [start + datetime.timedelta(days=i) for i in range(delta.days + 1)]
    return days


def trading_dates_from_given_dates(st_date, en_date, db_collection, exchange='NSE', file_name=None, logs_coll=None):
    try:
        if exchange == 'NSE':
            symbol = 'NIFTY 50'
        else:
            symbol = 'NIFTY 50'

        trading_dates = db_collection.find({'symbol': symbol, 'date': {'$gte': st_date, '$lte': en_date}}).distinct('date')
    except Exception as e:
        log_message = f"Could not find TradingDates using database.\nReturning date list from given start date and end date.\n{e}\n"
        mg_insert_log(logs_coll, file_name, 'ERROR', log_message,
                      log_traceback=''.join(tb.format_exception(None, e, e.__traceback__)))

        trading_dates = dates_list_from_start_end(st_date, en_date)

    return trading_dates


def gnl_on_dates_on_given_time(dates_list, gnl_time, symbols_list, db_collection, gnl_stocks_length, file_name, logs_coll):
    gnl_dict = dict()
    top_gnl_length = int(gnl_stocks_length // 2)
    for dt in dates_list:
        try:
            timestamp = datetime.datetime.combine(dt.date(), gnl_time)
            data_li = list(db_collection.find({'symbol': {'$in': symbols_list}, 'timestamp': timestamp}))

            change_dict_at_given_time = dict()
            for data in data_li:
                symbol = data['symbol']
                try:
                    this_close = data['close']
                    previous_day_close = data['previous_day_close']
                    percentage_change = ((this_close - previous_day_close) / previous_day_close) * 100
                    change_dict_at_given_time[symbol] = percentage_change
                except Exception as e:
                    log_message = f"{symbol}: {dt}\n{e}\n"
                    mg_insert_log(logs_coll, file_name, 'ERROR', log_message,
                                  log_traceback=''.join(tb.format_exception(None, e, e.__traceback__)))
                    continue

            change_sorted = sorted(change_dict_at_given_time.items(), key=lambda x: x[1], reverse=True)
            gainers = [x[0] for x in change_sorted[:top_gnl_length]]
            losers = [x[0] for x in change_sorted[top_gnl_length.__neg__():]]
            gnl_dict[dt] = {'gainers': gainers, 'losers': losers}

        except Exception as e:
            log_message = f"Exception in getting GnL Dict for date: {dt}\n{e}\n"
            mg_insert_log(logs_coll, file_name, 'ERROR', log_message,
                          log_traceback=''.join(tb.format_exception(None, e, e.__traceback__)))
            continue

    return gnl_dict


def get_fibonacci_pivot_points(high, low, close, level1=0.382, level2=0.618, level3=1.0, file_name=None, logs_coll=None):
    """ Function to get previous date ohlc and then apply fibonacci pivot points """
    '''
    Formula:
        - pivot = (h + l + c) / 3  # variants duplicate close or add open
        - support1 = p - level1 * (high - low)  # level1 0.382
        - support2 = p - level2 * (high - low)  # level2 0.618
        - support3 = p - level3 * (high - low)  # level3 1.000
        - resistance1 = p + level1 * (high - low)  # level1 0.382
        - resistance2 = p + level2 * (high - low)  # level2 0.618
        - resistance3 = p + level3 * (high - low)  # level3 1.000
    '''

    try:
        candle_length = abs(high - low)

        p = round((high + low + close) / 3, 2)

        r1 = round(p + level1 * candle_length, 2)
        r2 = round(p + level2 * candle_length, 2)
        r3 = round(p + level3 * candle_length, 2)

        s1 = round(p - level1 * candle_length, 2)
        s2 = round(p - level2 * candle_length, 2)
        s3 = round(p - level3 * candle_length, 2)
    except Exception as e:
        log_message = f"Exception in calculating Fibonacci Pivot points.\nReturning None\n{e}\n"
        mg_insert_log(logs_coll, file_name, 'ERROR', log_message,
                      log_traceback=''.join(tb.format_exception(None, e, e.__traceback__)))
        return None, None, None, None, None, None, None

    return p, r1, r2, r3, s1, s2, s3


def get_lot_size(stocks_li, exchange='NSE', segment='future', kite_api_key=None, file_name=None, logs_coll=None):
    lot_size_dict = dict()
    try:
        if exchange == 'NSE':
            from kiteconnect import KiteConnect
            kite_obj = KiteConnect(kite_api_key)
            stocks_df = pd.DataFrame(kite_obj.instruments(exchange='NFO'))
            if segment == 'future':
                fut_df = stocks_df[stocks_df['instrument_type'] == 'FUT']

                for symbol in stocks_li:
                    try:
                        if symbol == 'NIFTY 50':
                            symbol = 'NIFTY'
                        if symbol == 'NIFTY BANK':
                            symbol = 'BANKNIFTY'

                        stock_df = fut_df[fut_df['name'] == symbol]
                        lot_size = list(stock_df['lot_size'])[0]

                        if symbol == 'NIFTY':
                            symbol = 'NIFTY 50'
                        if symbol == 'BANKNIFTY':
                            symbol = 'NIFTY BANK'

                        lot_size_dict[symbol] = lot_size
                    except Exception as e:
                        lot_size_dict[symbol] = 1
                        log_message = f"Exception in getting lot size of {symbol}.\nSetting lot_size of this symbol to 1\n{e}\n"
                        mg_insert_log(logs_coll, file_name, 'ERROR', log_message,
                                      log_traceback=''.join(tb.format_exception(None, e, e.__traceback__)))

    except Exception as e:
        log_message = f"Exception in get_lot_size function\n{e}\n"
        mg_insert_log(logs_coll, file_name, 'ERROR', log_message,
                      log_traceback=''.join(tb.format_exception(None, e, e.__traceback__)))

    return lot_size_dict


def near_value(trade_type, operation, current_value, near_pct, file_name, logs_coll):
    value = current_value
    try:
        near_pct /= 100
        if trade_type == 'LONG':
            if operation == 'ENTRY':
                value = current_value * (1 + near_pct)
            elif operation == 'EXIT':
                value = current_value * (1 - near_pct)

        elif trade_type == 'SHORT':
            if operation == 'ENTRY':
                value = current_value * (1 - near_pct)
            elif operation == 'EXIT':
                value = current_value * (1 + near_pct)
        else:
            value = current_value
            log_message = f"trade type does not match with 'LONG' or 'SHORT'"
            mg_insert_log(logs_coll, file_name, 'ERROR', log_message)
    except Exception as e:
        log_message = f"Exception in finding near value.\nReturning same value as given\n{e}\n"
        mg_insert_log(logs_coll, file_name, 'ERROR', log_message,
                      log_traceback=''.join(tb.format_exception(None, e, e.__traceback__)))
        value = current_value

    return value


def get_pnl(trade_type, entry_p, exit_p, lot_size, file_name, logs_coll):
    try:
        if trade_type == 'LONG':
            points_gain = (exit_p - entry_p)
            pnl = points_gain * lot_size

        elif trade_type == 'SHORT':
            points_gain = (entry_p - exit_p)
            pnl = points_gain * lot_size

        else:
            points_gain = "Error in points gain calculation"
            pnl = 'Error in pnl calculation'

        return points_gain, pnl
    except Exception as e:
        log_message = f"Could not calculate pnl using get_pnl function."
        mg_insert_log(logs_coll, file_name, 'ERROR', log_message,
                      log_traceback=''.join(tb.format_exception(None, e, e.__traceback__)))
        return 0, 0


def calculate_margin(price, lot_sizes):
    """ Function to calculate future margin of Top GL """
    future_margin = 0
    margin = {}
    for stock in price:
        margin[stock] = price[stock] * lot_sizes[stock]
        future_margin += price[stock] * lot_sizes[stock]

    return future_margin,margin


def calculate_margin_percent(price, margin, lot_sizes):
    """ Function to calculate margin percentage """
    margin_percent = {}
    for stock in price:
        percent = (price[stock]*lot_sizes[stock])* (100/margin)
        margin_percent[stock] = percent
    return margin_percent


def calculate_amount_distribution(percent_distribution, investment_amount):
    """ Function to calculate Investment Amount using percentage distribution """
    amount_distribution = {}
    for stock in percent_distribution:
        amount_distribution[stock] = investment_amount * percent_distribution[stock]/100
    return amount_distribution


def calculate_share_distribution(price, amount_distribution, file_name, logs_coll):
    """ Function to calculate share distribution of Top GL Stocks """
    share_distribution = {}
    for stock in price:
        try:
            _shares = round(amount_distribution[stock] / price[stock])
            if _shares <= 0:
                _shares = 1
            share_distribution[stock] = _shares
        except Exception as e:
            log_message = f"Error in calculating share_distribution - {stock} {amount_distribution} {price}"
            mg_insert_log(logs_coll, file_name, 'ERROR', log_message, log_traceback=''.join(tb.format_exception(None, e, e.__traceback__)))
            share_distribution[stock] = 1
    return share_distribution


def calculate_distribution(price, lot_sizes, investment_amount):
    """ Function to calculate number of shares to buy/sell """

    future_margin, margin = calculate_margin(price, lot_sizes)
    margin_percent = calculate_margin_percent(price, future_margin, lot_sizes)
    amount_distribution = calculate_amount_distribution(margin_percent, investment_amount)
    # share_distribution = calculate_share_distribution(price, amount_distribution)

    return future_margin, margin_percent, amount_distribution #, share_distribution


def fetch_available_funds(credential_id):
    """Function to fetch current available funds in the account"""
    try:
        print("***** Fetching Available Funds for : {} *****".format(credential_id))
        response = requests.get(orders_url+"get_broker_fund?credential_id={}".format(credential_id))
        response_dict = response.json()
        if response_dict["status"] == "success" and response_dict["message"] == "Fund Found":
            fund_available = response_dict["data"]["net_balance"]
            print("fund_available : {}".format(fund_available))
            return fund_available
    except Exception as e:
        print("Exception in Fetching Available Funds : {}".format(e))
        return None


def calculate_funds(investment, available_funds):
    """ Function to calculate Funds required while placing order """
    try:
        print("Calculating Funds Required")
        if investment < available_funds:
            return investment
        else:
            return available_funds

    except Exception as e:
        print("Exception in calculating funds : {}".format(e))
        pass


def round_to_tick_multiple(input_price):
    """Function to round price to nearest tick multiple"""
    try:
        rounded_price = round((math.ceil(input_price * 20) / 20), 2)
        # print("price : {} | rounded_price : {}".format(input_price, rounded_price))
        return rounded_price
    except Exception as e:
        print("Exception in round to tick multiple : {}".format(e))
        pass


def send_order_alert(alert_dict):
    """Function to Send Alert on Order"""
    try:
        print("Sending Alert")
        response = requests.post(url=orders_url+"send_notification", json=alert_dict)
        print("response : {}".format(response.json()))
    except Exception as e:
        print("Exception in sending alert : {}".format(e))
        pass


def check_order_status(credential_id, order_id, exchange):
    """Function to fetch current order status"""
    try:
        print("Fetching Order Status")
        get_order_params = {'order_id': order_id, "exchange": exchange}
        last_order = requests.post(url=orders_url+"get_order_by_id", params={"credential_id": credential_id, 'order_details': json.dumps(get_order_params)}).json()
        print("order_status_response : {}".format(last_order))
        if last_order["status"] == "COMPLETE":
            last_order["exchange_timestamp"] = datetime.datetime.strptime(str(last_order["exchange_timestamp"]), '%Y-%m-%d %H:%M:%S')
            return "success", last_order

        elif last_order["status"] == "CANCELLED":
            return "cancelled", None
        else:
            return "failed", None
    except Exception as e:
        print("Exception in Check Order Status Function : {}".format(e))
        return "failed", None


def calculate_brokerage(buy_price, sell_price, quantity, broker="zerodha", market_type="options"):
    """Function to calculate brokerage"""
    try:
        if broker in ["zerodha", "dhan", "icici direct"]:

            gst = 0.18
            sebi_charges = 0.000001
            clearing_charge = 0

            if market_type == "options":
                stt_charges = 0.0625
                exchange_charges = 0.0505
                stampduty_charges = 0.00003
                _kite_brokerage = 40
            else:
                stt_charges = 0.01
                exchange_charges = 0.002
                stampduty_charges = 0.00002
                _kite_brokerage = 0.03

            # Calculating Turn Over
            turnover = (buy_price * quantity) + (sell_price * quantity)

            # Calculating Brokerage
            if market_type=="options":
                brokerage = _kite_brokerage
            else:
                brokerage = round(turnover * (_kite_brokerage / 100), 2)
                brokerage = min(40, brokerage)

            # Calculating STT
            stt = round(quantity * sell_price * stt_charges / 100)

            # Calculating Exchange Transaction Charges
            exchange_charges = round((turnover / 100) * exchange_charges, 2)

            # Calculating Sebi Charges
            sebi_charges = round(turnover * sebi_charges, 2)

            # Calculating Stamp Duty
            stamp_duty = round(quantity * buy_price * stampduty_charges)

            total_charges = brokerage + sebi_charges + exchange_charges

            # Total GST Charges
            gst = round(total_charges * gst, 2)

            # Calculating Total tax and Charges
            total_charges = round((brokerage + sebi_charges + exchange_charges + stt + clearing_charge + stamp_duty + gst), 2)

            # Calculating Net PnL
            net_pl = ((sell_price - buy_price) * quantity) - total_charges


            return total_charges, net_pl

        elif broker == "kotak":
            gst = 0.18
            sebi_charges = 0.000001
            clearing_charge = 0
            brokerage = 0

            if market_type == "options":
                stt_charges = 0.0625
                exchange_charges = 0.0505
                stampduty_charges = 0.00003
            else:
                stt_charges = 0.01
                exchange_charges = 0.002
                stampduty_charges = 0.00002

            # Calculating Turn Over
            turnover = (buy_price * quantity) + (sell_price * quantity)

            # Calculating STT
            stt = round(quantity * sell_price * stt_charges / 100)

            # Calculating Exchange Transaction Charges
            exchange_charges = round((turnover / 100) * exchange_charges, 2)

            # Calculating Sebi Charges
            sebi_charges = round(turnover * sebi_charges, 2)

            # Calculating Stamp Duty
            stamp_duty = round(quantity * buy_price * stampduty_charges)

            total_charges = brokerage + sebi_charges + exchange_charges

            # Total GST Charges
            gst = round(total_charges * gst, 2)

            # Calculating Total tax and Charges
            total_charges = round(
                (brokerage + sebi_charges + exchange_charges + stt + clearing_charge + stamp_duty + gst), 2)

            # Calculating Net PnL
            net_pl = ((sell_price - buy_price) * quantity) - total_charges

            return total_charges, net_pl
    except Exception as e:
        print("Exception in calculating brokerage : {}".format(e))
        pass


def merge_candle(candle_1, candle_2):
    """ Function to merge input candles """
    try:
        output_candle = candle_1.copy()
        output_candle["high"] = max([candle_1["high"], candle_2["high"]])
        output_candle["low"] = min([candle_1["low"], candle_2["low"]])
        output_candle["timestamp"] = candle_2["timestamp"]
        if "volume" in candle_1:
            output_candle["volume"] = sum([candle_1["volume"], candle_2["volume"]])
        if "oi" in candle_1:
            output_candle["oi"] = sum([candle_1["oi"], candle_2["oi"]])
        return output_candle
    except Exception as e:
        print(f"Exception in merging candle : {e}")
        pass


def convert_candle_pattern(dataframe, pattern):
    """ Function to convert a candle pattern """
    try:
        print("converting candle dataframe to : {}".format(pattern))
        if pattern == 'heikin_ashi':
            ha_df = dataframe.copy()
            ha_df['open'] = 0.0
            # Setting Close Values
            ha_df['close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

            for i in range(0, len(dataframe)):
                if i == 0:
                    ha_df.loc[0, 'open'] = (dataframe['open'][i] + dataframe['close'][i]) / 2
                else:
                    ha_df.loc[i, 'open'] = (ha_df.loc[i - 1, 'open'] + ha_df.loc[i - 1, 'close']) / 2

                # ha_df.loc[i, 'ha_color'] = 'green' if ha_df['close'][i] > ha_df['open'][i] else 'red'

            ha_df['high'] = ha_df[['high', 'open', 'close']].max(axis=1)
            ha_df['low'] = ha_df[['low', 'open', 'close']].min(axis=1)

            return ha_df

        if pattern == 'ask':
            last_candle = None
            output_list = []
            candles_list = dataframe.to_dict('records')

            for c_index, candle in enumerate(candles_list):
                if c_index == 0:
                    last_candle = candle
                else:
                    # Finding Highest between Open and Close Price
                    hp = max([last_candle["open"], last_candle["close"]])

                    # Finding Lowest between Open and Close Price
                    lp = min([last_candle["open"], last_candle["close"]])

                    market_exit_time = datetime.datetime.strptime("15:20", '%H:%M').time()

                    if candle["timestamp"].time() < market_exit_time:
                        # Checking Inside Body Candle condition
                        if lp <= candle['open'] <= hp and lp <= candle['close'] <= hp:
                            # Merging To Last Candle
                            last_candle = merge_candle(candle_1=last_candle, candle_2=candle)
                        else:
                            output_list.append(last_candle)

                            # Reassigning Last Candle
                            last_candle = candle
                    else:
                        output_list.append(candle)

            ask_df = pd.DataFrame(output_list)
            return ask_df
    except Exception as e:
        print("exception in converting candle pattern : {}".format(e))
        pass


def nearest_expiry_option_data(option_candle_list):
    try:
        expiry_dates = sorted([datetime.datetime.strptime(candle['expiry'], "%Y-%m-%d") for candle in option_candle_list])
        expiry_find = datetime.datetime.strftime(expiry_dates[0], "%Y-%m-%d")
        option_candle_df = pd.DataFrame(option_candle_list)
        option_candle_df.drop_duplicates(inplace=True)
        option_candle_df = option_candle_df[option_candle_df['expiry'] == expiry_find]

        return option_candle_df.to_dict('records')
    except Exception as e:
        print("error in filtering option data for nearest expiry")
        return []

