import pandas as pd
from pandas_ta.utils import get_offset, non_zero_range, verify_series
import numpy as np
import pandas_ta as ta


def stochastic_oscillator(dataframe, k=14, d=3, smooth_k=3, mamode='sma'):
    """ Function to calculate stochastic oscillator """
    try:
        # Calculate the rolling lowest low and highest high over the K period
        k = k if k and k > 0 else 14
        d = d if d and d > 0 else 3

        smooth_k = smooth_k if smooth_k and smooth_k > 0 else 3
        _length = max(k, d, smooth_k)

        high = verify_series(dataframe["high"], _length)
        low = verify_series(dataframe["low"], _length)
        close = verify_series(dataframe["close"], _length)

        if high is None or low is None or close is None: return

        # Calculate Result
        lowest_low = low.rolling(k).min()
        highest_high = high.rolling(k).max()

        stoch = 100 * (close - lowest_low)
        stoch /= non_zero_range(highest_high, lowest_low)
        stoch_k = stoch.rolling(smooth_k).mean()
        stoch_d = stoch_k.rolling(d).mean()

        new_df = pd.DataFrame()
        new_df["k"] = stoch_k
        new_df["d"] = stoch_d
        return new_df
    except Exception as e:
        print("Exception in calculating stochastic oscillator : {}".format(e))
        return None


def volume_flow_indicator(dataframe, length, coef, vcoef, signal_length, smooth_vfi=False):
    """ Function to calculate volume Flow Indicator Values """
    try:
        print("Calculating Volume Flow")
        df = pd.DataFrame()
        # Calculate 'hlc3' (Typical Price)
        df['typical'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3

        # Calculate intermediary variables
        df['inter'] = np.log(df['typical']) - np.log(df['typical'].shift(1))
        df['vinter'] = df['inter'].rolling(window=30).std()
        df['cutoff'] = coef * df['vinter'] * dataframe['close']
        df['vave'] = ta.sma(dataframe['volume'], length).shift(1)
        df['vmax'] = df['vave'] * vcoef

        # Calculate volume cutoff (vcp)
        df['vc'] = np.where(dataframe['volume'] < df['vmax'], dataframe['volume'], df['vmax'])

        # Calculate Money Flow (mf)
        df['mf'] = df['typical'] - df['typical'].shift(1)
        df['vcp'] = np.where(df['mf'] > df['cutoff'], df['vc'], np.where(df['mf'] < -df['cutoff'], -df['vc'], 0))

        # Calculate VFI
        df['vfi'] = df['vcp'].rolling(window=length).sum() / df['vave']

        # Smooth VFI if specified
        if smooth_vfi:
            df['vfi'] = ta.sma(df['vfi'], 3)

        # Calculate EMA of VFI
        df['vfi_ma'] = ta.ema(df['vfi'], signal_length)

        # Calculate difference
        df['vfi_d'] = df['vfi'] - df['vfi_ma']

        return df
    except Exception as e:
        print(f"Exception in calculating volume flow indicator : {e}")
        pass


def williams_fractal(dataframe, fractal_window=2):
    """ Function to calculate williams fractal"""
    try:
        # Initialize empty columns for bullish (up) and bearish (down) fractals
        dataframe['fractal_up'] = np.nan
        dataframe['fractal_down'] = np.nan

        # Iterate over the dataframe and check for fractals
        for i in range(fractal_window, len(dataframe) - fractal_window):
            # Bullish Fractal (Local High)
            if dataframe['high'][i] == max(dataframe['high'][i - fractal_window:i + fractal_window + 1]):
                dataframe.loc[i, 'fractal_up'] = dataframe['high'][i]

            # Bearish Fractal (Local Low)
            if dataframe['low'][i] == min(dataframe['low'][i - fractal_window:i + fractal_window + 1]):
                dataframe.loc[i, 'fractal_down'] = dataframe['low'][i]

        return dataframe
    except Exception as e:
        print(f"Exception in Williams Fractal : {e}")
        pass


def mcgd(close, length=None, offset=None, c=None, **kwargs):
    """ Indicator: McGinley Dynamic Indicator """
    # Validate arguments
    length = int(length) if length and length > 0 else 10
    c = float(c) if c and 0 < c <= 1 else 1
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None: return

    # Calculate Result
    close = close.copy()

    def mcg_(series):
        denom = (c * length * (series.iloc[1] / series.iloc[0]) ** 4)
        series.iloc[1] = (series.iloc[0] + ((series.iloc[1] - series.iloc[0]) / denom))
        return series.iloc[1]

    mcg_cell = close[0:].rolling(2, min_periods=2).apply(mcg_, raw=False)
    mcg_ds = close[:1]._append(mcg_cell[1:])

    # Offset
    if offset != 0:
        mcg_ds = mcg_ds.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        mcg_ds.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        mcg_ds.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    mcg_ds.name = f"MCGD_{length}"
    mcg_ds.category = "overlap"

    return mcg_ds


def squeeze_momentum(dataframe, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5, use_true_range=True):
    """ Function to calculate squeeze momentum """
    try:
        df = dataframe.copy()

        # Bollinger Bands (BB) Calculation
        source = df['close']
        basis = ta.sma(source, length=bb_length)
        dev = kc_mult * source.rolling(window=bb_length).std()
        df['upperBB'] = basis + dev
        df['lowerBB'] = basis - dev

        # Keltner Channels (KC) Calculation
        range_kc = ta.true_range(df['high'], df['low'], df['close']) if use_true_range else df['high'] - df['low']
        range_ma = ta.sma(range_kc, kc_length)

        df['maKC'] = ta.sma(df['close'], kc_length)
        df['upperKC'] = df['maKC'] + range_ma * kc_mult
        df['lowerKC'] = df['maKC'] - range_ma * kc_mult

        # Squeeze conditions
        df['sqzOn'] = (df['lowerBB'] > df['lowerKC']) & (df['upperBB'] < df['upperKC'])
        df['sqzOff'] = (df['lowerBB'] < df['lowerKC']) & (df['upperBB'] > df['upperKC'])
        df['noSqz'] = ~(df['sqzOn'] | df['sqzOff'])

        highest_high = df['high'].rolling(window=kc_length).max()
        lowest_low = df['low'].rolling(window=kc_length).min()
        hh_ll_avg = (highest_high + lowest_low)/2
        avg_2 = (hh_ll_avg + ta.sma(source, kc_length))/2
        df['val'] = ta.linreg((source - avg_2), length=kc_length, offset=0)

        return df[['val']]
    except Exception as e:
        print(f"Exception in Squeeze Momentum : {e}")
        pass


def tv_rsi(dataframe, rsi_length=14, source="close", ma_type="sma", ma_length=14, bb_mult=2.0):
    """ Function to calculate RSI similar to Trading View """
    try:
        df = dataframe.copy()
        lookback_right = 5
        lookback_left = 5

        # Calculate RSI
        df['rsi'] = ta.rsi(df[source], timeperiod=rsi_length)

        # Moving Average of RSI
        df['rsi_ma'] = ta.sma(close=df['rsi'], length=ma_length)

        # Calculate Bollinger Bands
        df['basis'] = ta.sma(df['close'], timeperiod=ma_length)
        df['stddev'] = ta.stdev(df['close'], timeperiod=ma_length)
        df['upper_bb'] = df['basis'] + bb_mult * df['stddev']
        df['lower_bb'] = df['basis'] - bb_mult * df['stddev']

        # KC Calculation (simplified as ATR-based KC)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], timeperiod=ma_length)
        df['upper_kc'] = df['basis'] + df['atr'] * bb_mult
        df['lower_kc'] = df['basis'] - df['atr'] * bb_mult

        # Squeeze conditions
        df['sqz_on'] = (df['lower_bb'] > df['lower_kc']) & (df['upper_bb'] < df['upper_kc'])
        df['sqz_off'] = (df['lower_bb'] < df['lower_kc']) & (df['upper_bb'] > df['upper_kc'])

        # Pivot-based divergence detection
        df['pivot_low'] = df['low'].rolling(window=lookback_left + lookback_right).apply(lambda x: x.idxmin(), raw=False)
        df['pivot_high'] = df['high'].rolling(window=lookback_left + lookback_right).apply(lambda x: x.idxmax(), raw=False)

        # Define bullish and bearish divergence conditions
        df['rsi_lbr'] = df['rsi'].shift(lookback_right)
        df['bullish_div'] = (df['low'] < df['low'].shift(lookback_right)) & (df['rsi'] > df['rsi'].shift(lookback_right))
        df['bearish_div'] = (df['high'] > df['high'].shift(lookback_right)) & (df['rsi'] < df['rsi'].shift(lookback_right))

        return df[['rsi', 'rsi_ma']]
    except Exception as e:
        print(f"Exception in TV RSI : {e}")
        pass


def chaikin_volatility(dataframe, length=10, roc_length=10):
    """ Function to calculate Chaikin Volatility """
    try:
        price_diff = dataframe['high'] - dataframe['low']
        ema_price_diff = ta.ema(price_diff, timeperiod=length)
        chaikin_volatility = ta.roc(ema_price_diff, timeperiod=roc_length)
        return chaikin_volatility
    except Exception as e:
        print(f"Exception in chaikin volatility : {e}")
        pass


def calculate_dmi(dataframe, adx_smoothing=14, di_length=14):
    """ Function to calculate Directional Moving Index """
    try:
        df = dataframe.copy()

        # Calculate price changes
        df['up'] = df['high'].diff()
        df['down'] = -df['low'].diff()

        # Initialize +DM and -DM
        df['dm_plus'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
        df['dm_minus'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)

        # Calculate True Range (TR)
        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - df['close'].shift())
        tr3 = np.abs(df['low'] - df['close'].shift())
        df['tr'] = np.max([tr1, tr2, tr3], axis=0)

        # Smooth True Range (TR) using the exponential moving average (EMA)
        df['tr_smooth'] = df['tr'].rolling(window=di_length).mean()

        # Smooth +DM and -DM using rolling mean (EMA)
        df['dm_plus_smooth'] = df['dm_plus'].rolling(window=di_length).mean()
        df['dm_minus_smooth'] = df['dm_minus'].rolling(window=di_length).mean()

        # Calculate +DI and -DI
        df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['tr_smooth'])
        df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['tr_smooth'])

        # Calculate DX (Directional Movement Index)
        df['dx'] = 100 * np.abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])

        # Calculate ADX by smoothing DX using the specified lensig period
        df['di_adx'] = df['dx'].rolling(window=adx_smoothing).mean()

        # Return only the relevant columns
        return df[['di_plus', 'di_minus', 'di_adx']]

    except Exception as e:
        print(f"Exception in calculating DMI : {e}")
        pass


def calculate_vwap(dataframe, anchor='session'):
    """ Function to calculate VWAP """
    try:
        df = dataframe.copy()
        df['cumulative_volume'] = df['volume'].cumsum()
        df['cumulative_volume_price'] = (df['close'] * df['volume']).cumsum()
        df['vwap'] = df['cumulative_volume_price'] / df['cumulative_volume']
        return df[['vwap']]
    except Exception as e:
        print(f"Exception in calculating vwap : {e}")
        pass


def direction_change(dataframe, depth=12, deviation=5, backstep=2):
    """ Function to calculate Direction Changes """
    try:
        df = dataframe.copy()
        # ta.barssince(not (higher[-ta.highestbars(depth)] - higher > deviation)[1])

        df['highest_high'] = df['high'].rolling(window=depth).max()
        df['lowest_low'] = df['low'].rolling(window=depth).min()

        # Calculating deviation from last value
        df['dev_high'] = (df['highest_high'].shift(1) - df['high']) > deviation
        df['dev_low'] = (df['low'] - df['lowest_low'].shift(1)) > deviation

        # Calculate direction
        df['hr'] = df['dev_high'].apply(lambda x: 1 if x else 0)
        df['lr'] = df['dev_low'].apply(lambda x: 1 if x else 0)

        df['direction'] = 0
        last_direction = 0

        for i in range(len(df)):
            if last_direction == df['lr'][i] == df['hr'][i] == 0:
                df.loc[i, 'direction'] = -1
                last_direction = -1

            # elif last_direction == -1 and df['llr'][i] == 1 and df['llr'][i-1] == 1 and df['hhr'][i] == 0:
            elif last_direction == -1 and df['lr'][i] == 1 and df['lr'][i-1] == 0:
                df.loc[i, 'direction'] = 1
                last_direction = 1

            # elif last_direction == 1 and df['hhr'][i] == 1 and df['hhr'][i-1] == 1:
            elif last_direction == 1 and df['hr'][i] == 0 and df['hr'][i-1] == 1:
                df.loc[i, 'direction'] = -1
                last_direction = -1

            # elif last_direction == 1 and df['lr'][i] == 0 and df['lr'][i-1] == 1:
            #     df.loc[i, 'direction'] = -1
            #     last_direction = -1
            else:
                df.loc[i, 'direction'] = last_direction

        print("current df")
        print(df)

    except Exception as e:
        print(f"Exception in Direction Change : {e}")
        pass


def yono(dataframe, depth=12, deviation=5, backstep=2, tick_size=0.05):
    try:
        df = dataframe.copy()
        df['highest_high'] = df['high'].rolling(window=depth).max()
        df['lowest_low'] = df['low'].rolling(window=depth).min()

        candles_list = df.to_dict('records')
        last_direction = 0

        for i in range(0, len(candles_list)):
            print(f"indx : {i} | candle : {candles_list[i]}")
            candles_list[i]['hr'] = 0
            candles_list[i]['lr'] = 0
            candles_list[i]["direction"] = 0
            hr_idx = []
            lr_idx = []
            if i < depth:
                continue
            else:
                highest_high = candles_list[i]["highest_high"]
                past_highs = [i['high'] for i in candles_list[i+1 - depth: i+1]]
                highest_high_index = past_highs.index(highest_high)

                for idx, high_val in enumerate(past_highs[highest_high_index+1:]):
                    if not (highest_high - high_val) > deviation * tick_size:
                        hr_idx.append(idx)

                if hr_idx:
                    candles_list[i]['hr'] = len(past_highs[highest_high_index+1:]) - hr_idx[-1]
                else:
                    candles_list[i]['hr'] = len(past_highs[highest_high_index + 1:])

                lowest_low = candles_list[i]["lowest_low"]
                past_lows = [i['low'] for i in candles_list[(i+1) - depth: i+1]]
                lowest_low_index = past_lows.index(lowest_low)

                for l_idx, low_val in enumerate(past_lows[lowest_low_index+1:]):
                    if not (low_val - lowest_low) > deviation * tick_size:
                        lr_idx.append(l_idx)

                if lr_idx:
                    candles_list[i]['lr'] = int(len(past_lows[lowest_low_index + 1:]) - lr_idx[-1])
                else:
                    candles_list[i]['lr'] = int(len(past_lows[lowest_low_index + 1:]))

                print(f"hr :  {candles_list[i]['hr']} | lr : {candles_list[i]['lr']}")

                if last_direction == 0:
                    if candles_list[i]['hr'] == 8 and candles_list[i]['lr'] == 0:
                        candles_list[i]['direction'] = -1
                        last_direction = -1

                    elif candles_list[i]['hr'] == 0 and candles_list[i]['lr'] == 8:
                        candles_list[i]['direction'] = 1
                        last_direction = 1

                elif last_direction == -1 and candles_list[i]['hr'] >= 5 and candles_list[i-1]['hr'] >= 5 and candles_list[i-1]['lr'] == 0 and candles_list[i]['lr'] == 1:
                    candles_list[i]['direction'] = 1
                    last_direction = 1

                elif last_direction == 1 and candles_list[i]['lr'] >= 5 and candles_list[i-1]['lr'] >= 5 and candles_list[i-1]['hr'] == 0 and candles_list[i]['hr'] == 1:
                    candles_list[i]['direction'] = -1
                    last_direction = -1

                # elif last_direction == 1 and candles_list[i-1]['hr'] == 0 and candles_list[i]['hr'] == 1:
                #     candles_list[i]['direction'] = -1
                #     last_direction = -1
                else:
                    candles_list[i]['direction'] = last_direction

                # if not candles_list[i]["hr"] > candles_list[i]["lr"] and not candles_list[i-1]["hr"] > candles_list[i-1]["lr"]:
                #     candles_list[i]["direction"] = -1
                # else:
                #     candles_list[i]["direction"] = 1

        df = pd.DataFrame(candles_list)
        print(df)

    except Exception as e:
        print(f"Exception in signal lib : {e}")
        pass


def calculate_direction(dataframe, depth, deviation, backstep, tick_size=0.05):
    df = dataframe.copy()
    high = df['high']
    low = df['low']

    # Calculate highest high and lowest low over the specified depth
    highest_high = high.rolling(window=depth).max()
    lowest_low = low.rolling(window=depth).min()

    # Calculate hr and lr
    hr = (highest_high.shift(1) - high) > (deviation * 0.05)  # Assuming syminfo.mintick is 1e-5 for example
    lr = (low - lowest_low.shift(1)) > (deviation * 0.05)
    df["hr"] = hr
    df["lr"] = lr

    # Count bars since condition was true
    hr_bars_since = hr[::-1].cumsum()[::-1]  # Reverse cumulative sum to count bars since
    lr_bars_since = lr[::-1].cumsum()[::-1]
    df["hr_bars_since"] = hr_bars_since
    df["lr_bars_since"] = lr_bars_since

    # Calculate direction
    direction = (hr_bars_since > lr_bars_since).astype(int)  # 1 if hr > lr, else 0
    direction = direction.rolling(window=backstep).sum()  # Sum over the backstep period

    # Final direction value
    final_direction = direction.apply(lambda x: -1 if x >= backstep else 1)

    df["direction"] = final_direction
    print("final_direction :")
    print(df)


def jurik_moving_average(dataframe, length=20, phase=50, power=2, source='close', highlight_movements=True):
    """ Function to calculate Juring Moving Average """
    try:
        df = dataframe.copy()
        # Initialize necessary columns and constants
        df['src'] = df[source]  # By default, using 'close' as the source
        phase_ratio = np.where(phase < -100, 0.5, np.where(phase > 100, 2.5, phase / 100 + 1.5))

        beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
        alpha = beta ** power

        # Create placeholder columns
        df['e0'] = 0.0
        df['e1'] = 0.0
        df['e2'] = 0.0
        df['jma'] = 0.0

        # Calculate JMA
        for i in range(1, len(df)):
            df.at[i, 'e0'] = (1 - alpha) * df.at[i, 'src'] + alpha * df.at[i - 1, 'e0']
            df.at[i, 'e1'] = (df.at[i, 'src'] - df.at[i, 'e0']) * (1 - beta) + beta * df.at[i - 1, 'e1']
            df.at[i, 'e2'] = (df.at[i, 'e0'] + phase_ratio * df.at[i, 'e1'] - df.at[i - 1, 'jma']) * (1 - alpha) ** 2 + (alpha ** 2) * df.at[i - 1, 'e2']
            df.at[i, 'jma'] = df.at[i, 'e2'] + df.at[i - 1, 'jma']

        # Highlight movements by coloring JMA based on its trend
        df['jmaColor'] = np.where((df['jma'] > df['jma'].shift(1)) & highlight_movements, 'green', np.where(highlight_movements, 'red', '#6d1e7f'))

        return df[['jma', 'jmaColor']]
    except Exception as e:
        print(f"Exception in calculating Jurik Moving Average : {e}")
        pass


def heikin_ashi_smoothed(dataframe, ema_length=55):
    """ Function to calculate Heikin-Ashi Smoothed"""
    try:
        df = dataframe.copy()
        # Calculate ohlc4 and hlc3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3

        # Calculate haOpen
        df['haOpen'] = (df['ohlc4'] + df['ohlc4'].shift(1).fillna(df['ohlc4'].iloc[0])) / 2

        # Calculate haC (similar to Heikin Ashi Close)
        df['haC'] = (df['ohlc4'] + df['haOpen'] + df[['high', 'haOpen']].max(axis=1) + df[['low', 'haOpen']].min(axis=1)) / 4

        # EMA calculations for Heikin Ashi Smoothed
        df['EMA1'] = ta.ema(df['haC'], ema_length)
        df['EMA2'] = ta.ema(df['EMA1'], ema_length)
        df['EMA3'] = ta.ema(df['EMA2'], ema_length)

        # TMA1 calculations
        df['TMA1'] = 3 * df['EMA1'] - 3 * df['EMA2'] + df['EMA3']

        # Further EMA calculations for TMA2
        df['EMA4'] = ta.ema(df['TMA1'], ema_length)
        df['EMA5'] = ta.ema(df['EMA4'], ema_length)
        df['EMA6'] = ta.ema(df['EMA5'], ema_length)

        # TMA2 calculations
        df['TMA2'] = 3 * df['EMA4'] - 3 * df['EMA5'] + df['EMA6']

        # Calculate IPEK and YASIN
        df['IPEK'] = df['TMA1'] - df['TMA2']
        df['YASIN'] = df['TMA1'] + df['IPEK']

        # EMA calculations for TMA3 and TMA4
        df['EMA7'] = ta.ema(df['hlc3'], ema_length)
        df['EMA8'] = ta.ema(df['EMA7'], ema_length)
        df['EMA9'] = ta.ema(df['EMA8'], ema_length)

        # TMA3 calculations
        df['TMA3'] = 3 * df['EMA7'] - 3 * df['EMA8'] + df['EMA9']

        # Further EMA calculations for TMA4
        df['EMA10'] = ta.ema(df['TMA3'], ema_length)
        df['EMA11'] = ta.ema(df['EMA10'], ema_length)
        df['EMA12'] = ta.ema(df['EMA11'], ema_length)

        # TMA4 calculations
        df['TMA4'] = 3 * df['EMA10'] - 3 * df['EMA11'] + df['EMA12']

        # Calculate IPEK1 and YASIN1
        df['IPEK1'] = df['TMA3'] - df['TMA4']
        df['YASIN1'] = df['TMA3'] + df['IPEK1']
        return df[['YASIN', 'YASIN1']]
    except Exception as e:
        print(f"Exception in calculating HeikinAshi Smoothed : {e}")
        pass


def average_volume(volume, length):
    """ Function to calculate Average Volume on Every Candle """
    try:
        print("Calculating Average Volume")
        if length == "full":
            avg_vol = volume.expanding().mean()
        elif isinstance(length, int) and length > 0:
            avg_vol = volume.rolling(window=length).mean()
        else:
            raise ValueError("Invalid length parameter. Must be 'full' or a positive integer.")
        return avg_vol
    except Exception as e:
        print(f"Exception in calculating Average Volume: {e}")
        return None


def bb_sideways(dataframe, bb_length=50, bb_mult=4.0, bbr_len=21, bbr_std_thresh=0.05):
    """ Function to calculate Bollinger Band Percent Sideways Indicator """
    try:
        df = dataframe.copy()

        # Bollinger Bands calculation
        df['basis'] = ta.sma(df['close'], length=bb_length)
        df['dev'] = bb_mult * ta.stdev(df['close'], bb_length)

        # df['std_dev'] = df['close'].rolling(window=bb_length).std()
        df['upper'] = df['basis'] + df['dev']
        df['lower'] = df['basis'] - df['dev']

        # Bollinger Bands % (BB%) calculation
        df['bbr'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])

        # Smoothing BB% with a rolling window standard deviation
        df['bbr_std'] = ta.stdev(df['bbr'], bbr_len)

        # Sideways detection based on BB% and threshold
        df['is_sideways'] = ((df['bbr'] > 0.0) & (df['bbr'] < 1.0)) & (df['bbr_std'] <= bbr_std_thresh)

        # Background color indicator (green for sideways, red for trending)
        df['color'] = np.where(df['is_sideways'], 'green', 'red')
        return df[['is_sideways', 'color']]
    except Exception as e:
        print(f"Exception in calculating BB Percent Sideways : {e}")
        pass


def calculate_rdx(dataframe):
    """ Function to Calculate RDX by trader hari krishna """
    try:
        df = dataframe.copy()

        # Calculate RSI and DMI (14-period for each)
        df['rsi'] = ta.rsi(df['close'], timeperiod=14)
        dmi_df = calculate_dmi_2(dataframe, period=14)

        df['plus_di'] = dmi_df['plus_di']
        df['minus_di'] = dmi_df['minus_di']
        df['adx'] = dmi_df['adx']

        # Conditional ADX-based bands
        df['s1'] = np.where(df['adx'] > 20, np.nan, 45)
        df['s2'] = np.where(df['adx'] > 20, np.nan, 55)

        # Plot filling based on ADX and RSI values
        # df['str'] = np.where(df['adx'] > 20, (df['adx'] - 20) / 5, 0)
        df['str'] = np.where(df['adx'] > 20, (df['adx'] - 25) / 5, 0)
        df['shifted_rsi'] = df['str'] + df['rsi']

        # Identify trend direction
        df['color'] = np.where(df['plus_di'] > df['minus_di'], 'green', 'red')

        # Buy and Sell Signals
        df['buy'] = (df['plus_di'] > df['minus_di']) & (df['plus_di'].shift(1) <= df['minus_di'].shift(1))
        df['sell'] = (df['plus_di'] < df['minus_di']) & (df['plus_di'].shift(1) >= df['minus_di'].shift(1))

        return df[["shifted_rsi"]]

    except Exception as e:
        print(f"Exception in Calculating RDX : {e}")
        pass


def calculate_dmi_2(dataframe, period):
    """ Function to Calculate Directional Movement Index """
    try:
        df = dataframe.copy()

        # Calculate +DM and -DM
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = -df['low'].diff()
        df['plus_dm'] = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0.0)
        df['minus_dm'] = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0), df['low_diff'], 0.0)

        # Calculate True Range (TR)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = np.abs(df['high'] - df['close'].shift(1))
        df['tr3'] = np.abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Smooth +DM,-DM, and TR with an exponential moving average
        df['plus_dm_smoothed'] = df['plus_dm'].rolling(window=period).mean()
        df['minus_dm_smoothed'] = df['minus_dm'].rolling(window=period).mean()
        df['tr_smoothed'] = df['tr'].rolling(window=period).mean()

        # Calculate +DI and -DI
        df['plus_di'] = 100 * (df['plus_dm_smoothed'] / df['tr_smoothed'])
        df['minus_di'] = 100 * (df['minus_dm_smoothed'] / df['tr_smoothed'])

        # Calculate DX (Directional Movement Index)
        df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])

        # Calculate ADX by smoothing the DX values
        df['adx'] = df['dx'].rolling(window=period).mean()
        return df[['plus_di', 'minus_di', 'adx']]
    except Exception as e:
        print(f"Exception in Calculating DMI : {e}")
        pass


def calculate_mti(dataframe, bb_length=20, bb_mult=2.0, adx_length=14, rsi_length=14):
    """ Function to Calculate Market Trend Indicator by FinnoVent """
    try:
        print("Calculating MTI")
        df = dataframe.copy()

        # EMA Settings
        df['shortEma'] = ta.ema(df['close'], timeperiod=3)
        df['longEma'] = ta.ema(df['close'], timeperiod=30)

        # Bollinger Bands Settings
        df['basis'] = ta.sma(df['close'], timeperiod=bb_length)
        df['dev'] = bb_mult * ta.stdev(df['close'], timeperiod=bb_length, nbdev=1)
        df['upper'] = df['basis'] + df['dev']
        df['lower'] = df['basis'] - df['dev']

        # ADX Calculation
        plus_dm = np.where((df['high'].diff() > df['low'].diff()) & (df['high'].diff() > 0), df['high'].diff(), 0)
        minus_dm = np.where((df['low'].diff() > df['high'].diff()) & (df['low'].diff() > 0), df['low'].diff(), 0)
        tr = ta.adx(df['high'], df['low'], df['close'], timeperiod=adx_length)
        df['plusDI'] = 100 * ta.sma(plus_dm, timeperiod=adx_length) / tr
        df['minusDI'] = 100 * ta.sma(minus_dm, timeperiod=adx_length) / tr
        dx = 100 * np.abs(df['plusDI'] - df['minusDI']) / (df['plusDI'] + df['minusDI'])
        df['adx'] = ta.sma(dx, timeperiod=adx_length)

        # RSI Settings

        df['rsi'] = ta.rsi(df['close'], timeperiod=rsi_length)

        # Sideways Condition
        df['sidewaysCondition'] = (
                (df['close'] > df['lower'] + (df['upper'] - df['lower']) * 0.20) &
                (df['close'] < df['upper'] - (df['upper'] - df['lower']) * 0.20) &
                (df['adx'] < 30) &
                (df['rsi'] > 40) & (df['rsi'] < 60)
        )

        # Trend Determination
        df['uptrend'] = (df['shortEma'] > df['longEma']) & (~df['sidewaysCondition'])
        df['downtrend'] = (df['shortEma'] < df['longEma']) & (~df['sidewaysCondition'])

        # Bar color based on trend
        df['color'] = np.where(df['uptrend'], 'green', np.where(df['downtrend'], 'red', np.where(df['sidewaysCondition'], 'gray', 'na')))
        print("mti_df")
        print(df)
    except Exception as e:
        print(f"Exception in Calculating MTI : {e}")
        pass


def smoothed_adx(dataframe, adx_length=14, di_length=14, smoothing_length=9, mamode="sma"):
    """ Function to Calculated Smoothed ADX """
    try:
        print("Calculating Smoothed ADX")
        out_df = pd.DataFrame()
        adx_df = ta.adx(high=dataframe["high"], low=dataframe["low"], close=dataframe["close"], length=adx_length, lensig=di_length)
        adx = round(adx_df.iloc[:, 0], 2).fillna(0)
        out_df["adx"] = adx
        s_adx = None
        if mamode == "ema":
            s_adx = ta.ema(adx, length=smoothing_length)
        else:
            s_adx = ta.sma(adx, length=smoothing_length)
        out_df["s_adx"] = s_adx

        return out_df
    except Exception as e:
        print(f"Exception in Calculating Smoothed ADX : {e}")
        pass


def calculate_net_volume(dataframe):
    """ Function to Calculate the Net Volume indicator based on price and volume data."""
    try:
        print("Calculating Net Volume")
        df = dataframe.copy()

        # Shift the close column to get the previous close prices
        df['prev_close'] = df['close'].shift(1)

        # Calculate up and down volume based on the change in 'close' prices
        df['up_volume'] = df['volume'].where(df['close'] > df['prev_close'], 0)
        df['down_volume'] = df['volume'].where(df['close'] < df['prev_close'], 0)

        # Calculate net volume as the difference between up and down volume
        df['net_volume'] = df['up_volume'] - df['down_volume']

        # Check if volume data is available
        cum_vol = df['volume'].sum()
        if cum_vol == 0:
            raise ValueError("The data vendor doesn't provide volume data for this symbol.")

        # Drop intermediate columns if not needed
        df.drop(columns=['up_volume', 'down_volume', 'prev_close'], inplace=True)

        return df[['net_volume']]
    except Exception as e:
        print(f"Exception in calculating net volume : {e}")
        pass


def volume_oscillator(dataframe, short_length=5, long_length=10):
    """ Function to Calculate Volume Oscillator Values """
    try:
        print("Calculating Volume Oscillator")
        df = dataframe.copy()

        # Check if volume data is available
        if df['volume'].sum() == 0:
            return None

        # Calculate short and long EMAs of volume
        df['short_ema'] = ta.ema(df["volume"], short_length)
        df['long_ema'] = ta.ema(df["volume"], long_length)

        # Calculate Volume Oscillator
        df['v_osc'] = 100 * (df['short_ema'] - df['long_ema']) / df['long_ema']

        # Drop intermediate columns if not needed
        df.drop(columns=['short_ema', 'long_ema'], inplace=True)

        return df[['v_osc']]
    except Exception as e:
        print(f"Exception in calculating volume oscillator : {e}")
        pass


def volume_slope(volume, length=9, mamode="ema"):
    """ Function to calculate Volume slope """
    try:
        out_df = pd.DataFrame()
        if mamode == "sma":
            out_df["vol_ma"] = ta.sma(volume, length=length)
        elif mamode == "ema":
            out_df["vol_ma"] = ta.ema(volume, length=length)

        out_df["vol_slope"] = out_df['vol_ma'].gt(out_df['vol_ma'].shift(1).fillna(-np.inf)).map({True: '+ve', False: '-ve'})
        out_df.loc[out_df['vol_ma'].isna(), 'vol_slope'] = 'NA'
        return out_df[["vol_ma", "vol_slope"]]
    except Exception as e:
        print(f"Exception in calculating Volume Slope : {e}")
        pass


def oi_slope(oi, length=9, mamode="ema"):
    """ Function to calculate OI Slope """
    try:
        out_df = pd.DataFrame()
        if mamode == "sma":
            out_df["oi_ma"] = ta.sma(oi, length=length)
        elif mamode == "ema":
            out_df["oi_ma"] = ta.ema(oi, length=length)

        out_df["oi_slope"] = out_df['oi_ma'].gt(out_df['oi_ma'].shift(1).fillna(-np.inf)).map({True: '+ve', False: '-ve'})
        out_df.loc[out_df['oi_ma'].isna(), 'oi_slope'] = 'NA'
        return out_df[["oi_ma", "oi_slope"]]
    except Exception as e:
        print(f"Exception in calculating OI Slope : {e}")
        pass


def custom_macd(dataframe, fast_length=12, slow_length=26, signal_length=9, source='close', sma_source='EMA', sma_signal='EMA'):
    """ Function to calculate Custom MACD """
    try:
        print()
        df = dataframe.copy()
        if sma_source == 'SMA':
            fast_ma = df[source].rolling(window=fast_length).mean()
            slow_ma = df[source].rolling(window=slow_length).mean()
        elif sma_source == 'EMA':
            fast_ma = df[source].ewm(span=fast_length, adjust=False).mean()
            slow_ma = df[source].ewm(span=slow_length, adjust=False).mean()
        else:
            raise ValueError("Invalid sma_source. Choose 'SMA' or 'EMA'.")

        macd = fast_ma - slow_ma

        if sma_signal == 'SMA':
            signal = macd.rolling(window=signal_length).mean()
        elif sma_signal == 'EMA':
            signal = macd.ewm(span=signal_length, adjust=False).mean()
        else:
            raise ValueError("Invalid sma_signal. Choose 'SMA' or 'EMA'.")

        hist = macd - signal

        # Add MACD components to the DataFrame
        df['macd'] = macd
        df['macd_h'] = hist
        df['macd_s'] = signal
        return df[["macd", "macd_h", "macd_s"]]
    except Exception as e:
        print(f"Exception in Custom MACD : {e}")
        pass


def bollinger_bandwidth(dataframe, length=20, source="close", std_dev=2, he_length=125, lc_length=125):
    """ Function to calculate bollinger bandwidth"""
    try:
        basis = ta.sma(dataframe[source], length)
        dev = std_dev * ta.stdev(dataframe[source], length)
        upper = basis + dev
        lower = basis - dev
        bbw = ((upper - lower) / basis) * 100
        return bbw
    except Exception as e:
        print(f"Exception in calculating bollinger bandwidth : {e}")
        pass


def calculate_bbw_range(dataframe, length, deviation=0.02):
    """ Function to calculate Bollinger Band Range"""
    try:
        df = dataframe.copy()
        df["bbw_ma"] = round(ta.sma(df["bbw"], length), 2)
        df["bbw_diff"] = abs(round(df["bbw"] - df["bbw_ma"], 2))
        df["bbw_range"] = np.where(df["bbw_diff"] > deviation, False, True)
        return df[['bbw_range']]
    except Exception as e:
        print(f"Exception in calculating BB Range : {e}")
        pass


def calculate_tsi(close, period=14):
    """ Calculate Trend Strength Index (TSI) using Correlation. """
    try:
        bar_index = pd.Series(range(len(close)), index=close.index)

        # Calculate rolling correlation
        tsi = close.rolling(window=period).corr(bar_index)

        return tsi
    except Exception as e:
        print(f"Exception in calculating TSI : {e}")
        pass

