from glob import glob
import os
import numpy as np
import requests
import sys
import zipfile
import sqlite3
import pandas as pd

# https://quantquote.com/historical-stock-data  Free Data tab
daily_csv_url = 'http://quantquote.com/files/quantquote_daily_sp500_83986.zip'

daily_dir = os.path.join(sys.prefix, 'share', 'fakestockdata', 'daily')


def download_daily():
    r = requests.get(daily_csv_url, stream=True)
    if not os.path.exists('data'):
        os.mkdir('data')

    with open(os.path.join('data', 'daily.zip'), 'wb') as f:
        for chunk in r.iter_content(chunk_size=2**12):
            f.write(chunk)

    f = zipfile.ZipFile(os.path.join('data', 'daily.zip'),
                        path=os.path.join('data', 'daily'))

    f.extractall()

columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

def load_file(fn):
    return pd.read_csv(fn,
                     parse_dates=['date'],
                     infer_datetime_format=True,
                     header=None, index_col='date',
                     compression='bz2' if fn.endswith('bz2') else None,
                     names=columns).drop('time', axis=1)


def generate_day(date, open, high, low, close, volume,
                 freq=pd.Timedelta(seconds=1)):
    time = pd.date_range(date + pd.Timedelta(hours=9),
                         date + pd.Timedelta(hours=5 + 12),
                         freq=freq / 5, name='timestamp')
    n = len(time)
    while True:
        values = (np.random.random(n) - 0.5).cumsum()
        values *= (high - low) / (values.max() - values.min())  # scale
        values += np.linspace(open - values[0], close - values[-1],
                              len(values))  # endpoints
        assert np.allclose(open, values[0])
        assert np.allclose(close, values[-1])

        mx = max(close, open)
        mn = min(close, open)
        ind = values > mx
        values[ind] = (values[ind] - mx) * (high - mx) / (values.max() - mx) + mx
        ind = values < mn
        values[ind] = (values[ind] - mn) * (low - mn) / (values.min() - mn) + mn
        if (np.allclose(values.max(), high) and  # The process fails if min/max
            np.allclose(values.min(), low)):     # are the same as open close
            break                                # this is pretty rare though

    s = pd.Series(values.round(8), index=time)
    rs = s.resample(freq)
    volume_distribution = np.random.dirichlet(np.ones(n), size=1).flatten()
    volumes = (volume_distribution * volume).round(8)
    volume_series = pd.Series(volumes, index=time).resample(freq).sum()

    return pd.DataFrame({'bid': rs.max(),
                         'ask': rs.min(),
                         'volume': volume_series},)


def generate_stock_csv(fn, directory=None, freq=pd.Timedelta(seconds=1),
                   start=pd.Timestamp('2000-01-01'),
                   end=pd.Timestamp('2050-01-01')):
    start = pd.Timestamp(start)
    directory = directory or os.path.join('data', 'generated')
    fn2 = os.path.split(fn)[1]
    sym = fn2[len('table_'):fn2.find('.csv')]
    if not os.path.exists(directory):
        os.mkdir(directory)
    if not os.path.exists(os.path.join(directory, sym)):
        os.mkdir(os.path.join(directory, sym))

    df = load_file(fn)
    for date, rec in df.to_dict(orient='index').items():
        if start <= pd.Timestamp(date) <= end:
            df2 = generate_day(date, freq=freq, **rec)
            fn2 = os.path.join(directory, sym, str(date).replace(' ', 'T') + '.csv')
            df2.to_csv(fn2)
    print('Finished %s' % sym)


def create_table_if_not_exists(conn, symbol, dropDates=False):
    query = ""
    if dropDates:
        query = f"""
        CREATE TABLE IF NOT EXISTS {symbol} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bid REAL,
            ask REAL,
            volume REAL
        )
        """
    else:
        query = f"""
        CREATE TABLE IF NOT EXISTS {symbol} (
            timestamp TEXT PRIMARY KEY,
            bid REAL,
            ask REAL,
            volume REAL
        )
        """
    
    conn.execute(query)
    conn.commit()


def insert_data(conn, symbol, df, dropDates=False):
    if dropDates:
        df.to_sql(symbol, conn, if_exists='append', index=False)
    else:
        df.to_sql(symbol, conn, if_exists='append', index=True, index_label='timestamp')


def generate_stock_sql(fn, db_path='stocks_data.db', freq=pd.Timedelta(seconds=1),
                   start=pd.Timestamp('2000-01-01'),
                   end=pd.Timestamp('2050-01-01'),
                   dropDates=False):
    start = pd.Timestamp(start)
    fn2 = os.path.split(fn)[1]
    sym = fn2[len('table_'):fn2.find('.csv')]

    conn = sqlite3.connect(db_path)

    create_table_if_not_exists(conn, sym, dropDates)

    df = load_file(fn)
    
    for date, rec in df.to_dict(orient='index').items():
        if start <= pd.Timestamp(date) <= end:
            if dropDates:
                df2 = generate_day(date, freq=freq, **rec).reset_index().drop('timestamp', axis=1)
                insert_data(conn, sym, df2, dropDates=True)
            else:
                df2 = generate_day(date, freq=freq, **rec)
                insert_data(conn, sym, df2)

    conn.close()
    print(f'Finished processing {sym}')


def generate_stocks_csv(freq=pd.Timedelta(seconds=1), directory=None,
                    start=pd.Timestamp('2000-01-01')):
    from concurrent.futures import ProcessPoolExecutor, wait
    e = ProcessPoolExecutor()
    if os.path.exists(os.path.join('data', 'daily')):
        glob_path = os.path.join('data', 'daily', '*')
    else:
        glob_path = os.path.join(daily_dir, '*')
    filenames = sorted(glob(glob_path))

    futures = [e.submit(generate_stock_csv, fn, directory=directory, freq=freq,
                        start=start)
                for fn in filenames]
    wait(futures)


def generate_stocks_sql(freq=pd.Timedelta(seconds=1),
                    start=pd.Timestamp('2000-01-01'),
                    db_path='stocks_data.db',
                    dropDates=False):
    if os.path.exists(os.path.join('data', 'daily')):
        glob_path = os.path.join('data', 'daily', '*')
    else:
        glob_path = os.path.join(daily_dir, '*')
    filenames = sorted(glob(glob_path))

    for file in filenames:
        generate_stock_sql(fn=file, db_path=db_path, freq=freq, start=start, dropDates=dropDates)
        print(f'Added {file} to the database.')
