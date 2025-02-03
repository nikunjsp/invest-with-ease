from yahoodownloader import YahooDownloader
from datetime import date, timedelta, datetime
import config
from preprocessors import FeatureEngineer
import pandas as pd

today = date.today()
tomorrow = today + timedelta(1)


def getData():

    temp_data = pd.read_csv("1final.csv")
    ldate = temp_data.iloc[-1].date
    ldate = datetime.strptime(ldate, "%Y-%m-%d").date() + timedelta(1)
    print(ldate)
    if ldate == today:
        pass
    else:
        df2 = YahooDownloader(
            start_date=ldate, end_date=today, ticker_list=config.MY_TICKER
        ).fetch_data()

    tech_indicators = ["macd", "rsi_30", "cci_30", "dx_30"]
    # df.to_csv("raw_data2.csv")
    # df = pd.read_csv("raw_data2.csv")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=tech_indicators,
        use_turbulence=False,
        user_defined_feature=False,
    )

    df2_processed_full = fe.preprocess_data(df2)
    df1 = pd.read_csv("1final.csv", index_col=0)
    df3 = df1.append(df2_processed_full, ignore_index=True)
    df3.reset_index(inplace=True, drop=True)
    # processed_full.to_csv("processed_data.csv")
    # processed_full = pd.read_csv("processed_data.csv")
    # processed_full = fe.add_rs(processed_full)
    # temp_processed_full.fillna(temp_processed_full.mean())
    df3.to_csv("1final.csv")
    temp_df = YahooDownloader(
        start_date=today, end_date=tomorrow, ticker_list=config.MY_TICKER
    ).fetch_data()

    # df.to_csv("raw_data2.csv")
    # df = pd.read_csv("raw_data2.csv")

    processed_full = df3.append(temp_df, ignore_index=True)
    for i in processed_full.columns[
        processed_full.isnull().any(axis=0)
    ]:  # ---Applying Only on variables with NaN values
        processed_full[i].fillna(processed_full[i].mean(), inplace=True)
    # processed_full.to_csv("processed_data.csv")
    # processed_full = pd.read_csv("processed_data.csv")
    # processed_full = fe.add_rs(processed_full)
    processed_full.to_csv("ffinal.csv")


# getData()
"""temp_data = pd.read_csv("1final.csv")
ldate = temp_data.iloc[-1].date
ldate = datetime.datetime.strptime(ldate, "%Y-%m-%d").date() + timedelta(1)
print(ldate)
if ldate == tomorrow:
    pass
else:

    temp_df = YahooDownloader(
        start_date=ldate, end_date=tomorrow, ticker_list=config.MY_TICKER
    ).fetch_data()
    tech_indicators = ["macd", "rsi_30", "cci_30", "dx_30"]
    # df.to_csv("raw_data2.csv")
    # df = pd.read_csv("raw_data2.csv")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=tech_indicators,
        use_turbulence=False,
        user_defined_feature=False,
    )

    temp_processed_full = fe.preprocess_data(temp_df)
    # processed_full.to_csv("processed_data.csv")
    # processed_full = pd.read_csv("processed_data.csv")
    # processed_full = fe.add_rs(processed_full)
    # temp_processed_full.fillna(temp_processed_full.mean())
    temp_processed_full.to_csv("1final.csv", mode="a", header=False)
"""
