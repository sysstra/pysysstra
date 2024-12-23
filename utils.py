import pandas as pd


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
