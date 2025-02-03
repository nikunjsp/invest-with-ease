import pathlib

# import finrl

import pandas as pd
import datetime
import os

# pd.options.display.max_rows = 10
# pd.options.display.max_columns = 10


# PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
# PACKAGE_ROOT = pathlib.Path().resolve().parent

TRAINED_MODEL_DIR = f"trained_models"
# DATASET_DIR = PACKAGE_ROOT / "data"

# data
# TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
# TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
# TESTING_DATA_FILE = "test.csv"

# now = datetime.datetime.now()
# TRAINED_MODEL_DIR = f"trained_models/{now}"
DATA_SAVE_DIR = f"datasets"
TRAINED_MODEL_DIR = f"trained_models"
TENSORBOARD_LOG_DIR = f"tensorboard_log"
RESULTS_DIR = f"results"
# os.makedirs(TRAINED_MODEL_DIR)


## time_fmt = '%Y-%m-%d'
START_DATE = "2009-01-01"
END_DATE = "2021-01-01"

START_TRADE_DATE = "2019-01-01"

## dataset default columns
DEFAULT_DATA_COLUMNS = ["date", "tic", "close"]

## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
#TECHNICAL_INDICATORS_LIST = ["macd","boll_ub","boll_lb","rsi_30", "cci_30", "dx_30","close_30_sma","close_60_sma"]
TECHNICAL_INDICATORS_LIST = ["macd","rsi_30", "cci_30", "dx_30"]


## Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": "auto_0.1",
}

########################################################
############## Stock Ticker Setup starts ##############
SINGLE_TICKER = ["AAPL"]
MY_TICKER1=['ASIANPAINT.NS',
'AXISBANK.NS',
'BAJFINANCE.NS',
'BAJAJFINSV.NS',
'BHARTIARTL.NS',
]
MY_TICKER=['ASIANPAINT.NS',
'AXISBANK.NS',
'BAJFINANCE.NS',
'BAJAJFINSV.NS',
'BHARTIARTL.NS',
'BPCL.NS',
'BRITANNIA.NS',
'CIPLA.NS',
'DIVISLAB.NS',
'DRREDDY.NS',
'EICHERMOT.NS',
'GRASIM.NS',
'HCLTECH.NS',
'HDFCBANK.NS',
'HEROMOTOCO.NS',
'HINDALCO.NS',
'HINDUNILVR.NS',
'ICICIBANK.NS',
'INDUSINDBK.NS',
'INFY.NS',
'IOC.NS',
'ITC.NS',
'JSWSTEEL.NS',
'KOTAKBANK.NS',
'LT.NS',
'M&M.NS',
'MARUTI.NS',
'NTPC.NS',
'ONGC.NS',
'RELIANCE.NS',
'SBIN.NS',
'SHREECEM.NS',
'SUNPHARMA.NS',
'TATAMOTORS.NS',
'TATASTEEL.NS',
'TCS.NS',
'TATACONSUM.NS',
'TECHM.NS',
'TITAN.NS',
'ULTRACEMCO.NS',
'UPL.NS',
'WIPRO.NS']
# self defined
SRI_KEHATI_TICKER = [
		"AALI.JK",
		"ADHI.JK",
		"ASII.JK",
		"BBCA.JK", 
		"BBNI.JK",
		"BBRI.JK",
		"BBTN.JK",
		"BMRI.JK",
		"BSDE.JK",
		"INDF.JK",
		"JPFA.JK",
		"JSMR.JK",
		"KLBF.JK",
		"PGAS.JK",
		"PJAA.JK",
		"PPRO.JK",
		"SIDO.JK",
		"SMGR.JK",
		"TINS.JK",
		"TLKM.JK",
		"UNTR.JK",
		"UNVR.JK",
		"WIKA.JK",
		"WSKT.JK",
		"WTON.JK"
]

# check https://wrds-www.wharton.upenn.edu/ for U.S. index constituents
# Dow 30 constituents at 2019/01
DOW_30_TICKER = [
    "AAPL",
    "MSFT",
    "JPM",
    "V",
    "RTX",
    "PG",
    "GS",
    "NKE",
    "DIS",
    "AXP",
    "HD",
    "INTC",
    "WMT",
    "IBM",
    "MRK",
    "UNH",
    "KO",
    "CAT",
    "TRV",
    "JNJ",
    "CVX",
    "MCD",
    "VZ",
    "CSCO",
    "XOM",
    "BA",
    "MMM",
    "PFE",
    "WBA",
    "DD",
]

# Nasdaq 100 constituents at 2019/01
NAS_100_TICKER = [
    "AMGN",
    "AAPL",
    "AMAT",
    "INTC",
    "PCAR",
    "PAYX",
    "MSFT",
    "ADBE",
    "CSCO",
    "XLNX",
    "QCOM",
    "COST",
    "SBUX",
    "FISV",
    "CTXS",
    "INTU",
    "AMZN",
    "EBAY",
    "BIIB",
    "CHKP",
    "GILD",
    "NLOK",
    "CMCSA",
    "FAST",
    "ADSK",
    "CTSH",
    "NVDA",
    "GOOGL",
    "ISRG",
    "VRTX",
    "HSIC",
    "BIDU",
    "ATVI",
    "ADP",
    "ROST",
    "ORLY",
    "CERN",
    "BKNG",
    "MYL",
    "MU",
    "DLTR",
    "ALXN",
    "SIRI",
    "MNST",
    "AVGO",
    "TXN",
    "MDLZ",
    "FB",
    "ADI",
    "WDC",
    "REGN",
    "LBTYK",
    "VRSK",
    "NFLX",
    "TSLA",
    "CHTR",
    "MAR",
    "ILMN",
    "LRCX",
    "EA",
    "AAL",
    "WBA",
    "KHC",
    "BMRN",
    "JD",
    "SWKS",
    "INCY",
    "PYPL",
    "CDW",
    "FOXA",
    "MXIM",
    "TMUS",
    "EXPE",
    "TCOM",
    "ULTA",
    "CSX",
    "NTES",
    "MCHP",
    "CTAS",
    "KLAC",
    "HAS",
    "JBHT",
    "IDXX",
    "WYNN",
    "MELI",
    "ALGN",
    "CDNS",
    "WDAY",
    "SNPS",
    "ASML",
    "TTWO",
    "PEP",
    "NXPI",
    "XEL",
    "AMD",
    "NTAP",
    "VRSN",
    "LULU",
    "WLTW",
    "UAL",
]