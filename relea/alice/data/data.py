import pandas as pd

# https://www.alphavantage.co/documentation/
# TIME_SERIES_DAILY - 20 years of daily - HISTORICAL
# TIME_SERIES_INTRADAY - intraday time series - LIVE
# GLOBAL_QUOTE - latest price and volume
#

class Data:

    def __init__(self):
        pass

    @classmethod
    def stock_import(self, symbol, function):
        '''

        '''
        url_base = "https://www.alphavantage.co/query?"
        function = "function={}".format(function)
        symbol = "symbol={}".format(symbol)
        datatype = "datatype=csv"
        apikey = "apikey=KFUO7CP686X6PBCS"

        if function == 'function=TIME_SERIES_INTRADAY':
            interval = 'interval=1min'
            url_body = "&".join([function, symbol, interval, datatype, apikey])
        elif function == 'function=TIME_SERIES_DAILY':
            url_body = "&".join([function, symbol, datatype, apikey])


        url = url_base + url_body
        # print(url) #DEBUG
        df = pd.read_csv(url)

        return df

data = Data()
df = data.stock_import(symbol='TSLA', function='TIME_SERIES_INTRADAY')

df.tail(10)
