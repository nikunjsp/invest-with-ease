from operator import index
import pandas as pd
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
import datetime
import gym
from stable_baselines3 import A2C, SAC, PPO, TD3
import config
from yahoodownloader import YahooDownloader
from preprocessors import FeatureEngineer, data_split
from env_stocktrading import StockTradingEnv
from models import DRLAgent, DRLEnsembleAgent
from plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import data
from pprint import pprint

import sys

import itertools
from datetime import date, timedelta, datetime

today = date.today()
tomorrow = today + timedelta(1)
yesterday = today - timedelta(1)
print(yesterday)


def app():
    temp_data = pd.read_csv("finall.csv")
    ldate = temp_data.iloc[-1].date
    ldate = datetime.strptime(ldate, "%Y-%m-%d").date() + timedelta(1)
    print(ldate)
    if ldate == yesterday:
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
        df1 = pd.read_csv("finall.csv", index_col=0)
        df3 = df1.append(df2_processed_full, ignore_index=True)
        df3.reset_index(inplace=True, drop=True)
        # processed_full.to_csv("processed_data.csv")
        # processed_full = pd.read_csv("processed_data.csv")
        # processed_full = fe.add_rs(processed_full)
        # temp_processed_full.fillna(temp_processed_full.mean())
        df3.to_csv("finall.csv")
    temp_df = YahooDownloader(
        start_date=today, end_date=tomorrow, ticker_list=config.MY_TICKER
    ).fetch_data()

    # df.to_csv("raw_data2.csv")
    # df = pd.read_csv("raw_data2.csv")
    df3 = pd.read_csv("finall.csv")
    processed_full = df3.append(temp_df, ignore_index=True)
    for i in processed_full.columns[
        processed_full.isnull().any(axis=0)
    ]:  # ---Applying Only on variables with NaN values
        processed_full[i].fillna(processed_full[i].mean(), inplace=True)

    # processed_full = pd.read_csv("finall.csv")
    stock_dimension = len(processed_full.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(tech_indicators) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicators,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,  # 0.0001
        "print_verbosity": 5,
    }
    startdate = "2021-10-22"
    enddate = "2021-10-15"
    trade = data_split(processed_full, str(startdate), str(tomorrow))
    # train = data_split(processed_full, '2007-01-01','2019-09-11')
    # e_train_gym = StockTradingEnv(df = train, **env_kwargs)
    # env_train, _ = e_train_gym.get_sb_env()
    # agent = DRLAgent(env = env_train)
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
    data_turbulence = processed_full[
        (processed_full.date < "2021-01-01") & (processed_full.date >= "2019-01-01")
    ]
    insample_turbulence = data_turbulence.drop_duplicates(subset=["date"])
    turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
    trained_a2c = A2C.load("my_a2c_model", verbose=1)
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_a2c, environment=e_trade_gym
    )
    df_actions = df_actions.set_index([df_account_value.date[1:]])
    min_df_actions = df_actions.drop(df_actions.index[range(len(df_actions.index) - 1)])
    min_df_actions = min_df_actions.loc[:, (df_actions != 0).any(axis=0)]
    min_df_actions.reset_index(inplace=True, drop=True)
    st.title("Invest With Ease")
    st.markdown("Welcome to this in-depth introduction to stocks price prediction")

    st.header("See everyday actions on selected 42 stocks:")

    st.dataframe(df_actions, width=1050)
    sum_actions = df_actions.sum()
    sum_actions.to_csv("Sum.csv", index=False)
    act = pd.read_csv("Sum.csv")
    value1 = act.transpose()
    value1.astype(int)
    value1.to_csv("Sum.csv", header=None)

    old_actions = pd.read_csv("df_actions1.csv", header=0)
    old_actions.astype(int)

    value1.add(old_actions, fill_value=0)

    value1.to_csv("df_actions1.csv", mode="a", header=False, index=False)

    df_actions.to_csv("df_actions01.csv", mode="a", header=False, index=False)
    st.header("Account value increment throughout test data:")
    st.line_chart(df_account_value.account_value)

    st.header("Backtest results:")
    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    st.dataframe(perf_stats_all)


# app()
