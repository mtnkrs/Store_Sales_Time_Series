import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
import os
import plotly.express as px
from google.colab import files
import joblib
import optuna
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
plt.style.use('fivethirtyeight')
color_pal = sns.color_palette()
import warnings
warnings.filterwarnings("ignore") 
pd.set_option('display.max_columns', None)


def load_application(file):
    data = pd.read_csv(file)
    return data

train = load_application("train.csv")
test=load_application("test.csv")
store=load_application("stores.csv")
transaction=load_application("transactions.csv")
oil=load_application("oil.csv")
holiday=load_application("holidays_events.csv")

total_df=[train, test,store,transaction,oil,holiday]

def change_date(given_df):
    for df in given_df:
        for col in df.columns:
            if "date" in col:
                df[col]= pd.to_datetime(df[col])

change_date(total_df)

def oil_preprocessing(dataframe):

  dataframe = dataframe.set_index("date").dcoilwtico.resample("D").sum().reset_index()
  dataframe["dcoilwtico"] = np.where(dataframe["dcoilwtico"] == 0, np.nan, dataframe["dcoilwtico"])
  dataframe["oil_interpolated"] = dataframe.dcoilwtico.interpolate()
  dataframe.loc[0,"oil_interpolated"] = 93.126667
  return dataframe

oil=oil_preprocessing(oil)

def calculate_values(train,test):
    train_start = train.date.min()
    train_end = train.date.max()
    return train_start,train_end

train_start,train_end=calculate_values(train,test)

def interpolate_train(train, train_start, train_end):
    multi_idx = pd.MultiIndex.from_product(
        [pd.date_range(train_start, train_end), train.store_nbr.unique(), train.family.unique()],
        names=["date", "store_nbr", "family"],
    )
    train = train.set_index(["date", "store_nbr", "family"]).reindex(multi_idx).reset_index()
    train[["sales", "onpromotion"]] = train[["sales", "onpromotion"]].fillna(0)
    train["id"] = train["id"].interpolate(method="linear")
    return train

train=interpolate_train(train,train_start, train_end)

def process_transaction(train, transaction):
    num_zero_sales = (train.groupby(["date", "store_nbr"]).sales.sum().eq(0)).sum()
    store_sales = train.groupby(["date", "store_nbr"]).sales.sum().reset_index()
    transaction = transaction.merge(
        store_sales,
        on=["date", "store_nbr"],
        how="outer",
    ).sort_values(["date", "store_nbr"], ignore_index=True)

    transaction.loc[transaction.sales.eq(0), "transactions"] = 0.
    transaction = transaction.drop(columns=["sales"])

    transaction["transactions"] = transaction.groupby("store_nbr", group_keys=False)["transactions"].apply(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    return transaction

transaction= process_transaction(train,transaction)

def process_holiday(holiday):
    deleted = holiday[(holiday["type"] == "Bridge") & (holiday["description"] == "Puente Navidad")]
    holiday = holiday[~holiday.index.isin(deleted.index)]
    holiday.loc[holiday["description"].str.contains("Navidad-"), "description"] = "Navidad_prep"
    holiday["description"] = holiday["description"].replace("Navidad+1", "Navidad_prep")
    return holiday
holiday=process_holiday(holiday)

holiday["nat_terremoto"] =holiday.apply(lambda x : 1 if "Terremoto" in x["description"] else 0 , axis=1)

holiday["is_holiday"]=holiday["nat_terremoto"].apply(lambda x: 0 if x==1 else 1 )

def merged_df(train, store, oil, transaction):
    merged_df = train.merge(store, on="store_nbr", how="left")
    merged_df= merged_df.fillna(0)
    selected_col= merged_df.iloc[:, 15:].columns
    merged_df= merged_df.merge(oil[["date","oil_interpolated"]], on="date", how="left")
    merged_df= merged_df.merge(transaction, on=["date","store_nbr"], how="left")
    return merged_df

merged_df = merged_df(train, store, oil, transaction)

merged_df= merged_df.merge(holiday[["date","nat_terremoto","is_holiday"]], on="date", how="left")

selected_2= ["id","city","state","type","cluster"]

merged_df= merged_df.drop(selected_2,axis=1)

def mevsimler(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in[3,4,5]:
        return "Spring"
    elif month in [6,7,8]:
        return "Summer"
    else:
        return "Autumn"

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    #df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek+1
    df['year'] = df.date.dt.year
    df["is_wknd"] = (df.date.dt.dayofweek >= 5).astype(int)
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    df["Days_name"] = df.date.dt.day_name()
    df["quarter"]= df.date.dt.quarter
    df["season"]=df["month"].apply(mevsimler)
    df['is_year_start'] = df.date.dt.is_year_start.astype("int8")
    df['is_year_end'] = df.date.dt.is_year_end.astype("int8")
    return df

merged_df= create_date_features(merged_df)

def oil_lag(dataframe):
  dataframe['lag15_sales'] = dataframe["oil_interpolated"].shift(15)
  dataframe['lag30_sales'] = dataframe["oil_interpolated"].shift(30)
  dataframe['lag60_sales'] = dataframe["oil_interpolated"].shift(60)
  dataframe['lag90_sales'] = dataframe["oil_interpolated"].shift(90)

  dataframe['lag15_sales'].fillna(method='bfill', inplace=True)
  dataframe['lag30_sales'].fillna(method='bfill', inplace=True)
  dataframe['lag60_sales'].fillna(method='bfill', inplace=True)
  dataframe['lag90_sales'].fillna(method='bfill', inplace=True)
  return dataframe

merged_df= oil_lag(merged_df)

def new_features(merged_df):
  merged_df["promotion_level"]=pd.cut(merged_df["onpromotion"], bins=[-1,0,10,120,float('inf')] ,labels=["no_prom","rare_prom","minumum_prom","yok_artık"])
  merged_df["promotiondays"]= merged_df["promotion_level"].astype("object") +"_"+ merged_df["Days_name"].astype("object")
  merged_df["NEW_transaction_onpromotion"]= merged_df["onpromotion"]/ merged_df["transactions"]
  merged_df['time_to_next_promotion'] = merged_df.groupby(['store_nbr', 'family'])['onpromotion'].transform(lambda x: x[::-1].cumsum()[::-1])
  merged_df['transaction_oil_interaction'] = merged_df['transactions'] * merged_df['oil_interpolated']
  merged_df["NEW_transaction_onpromotion"]= merged_df["NEW_transaction_onpromotion"].fillna(0)
  return merged_df

merged_df= new_features(merged_df)

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=len(dataframe))

def lag_features(dataframe, fh):
    df_copy = dataframe.copy()
    lags = [
        fh + 2, fh + 6, fh + 13, fh + 16, fh + 30, fh + 46, fh + 76, fh + 77, fh + 83, fh + 85,
        fh + 90, fh + 97, fh + 104, fh + 111, fh + 135, fh + 167, fh + 185,
        fh + 205, fh + 235, fh + 285, fh + 335, fh + 350
    ]
    for lag in lags:
        df_copy['sales_lag_' + str(lag)] = dataframe.groupby(["store_nbr", "family"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return df_copy

def roll_mean_features(dataframe, fh):
    df_copy = dataframe.copy()
    windows = [181, 365, 546]
    for window in windows:
        df_copy['sales_roll_mean_' + str(window)] = dataframe.groupby(["store_nbr", "family"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return df_copy

def ewm_features(dataframe, alphas, fh):
    df_copy = dataframe.copy()
    lags = [fh + 2, fh + 6, fh + 13, fh + 16, fh + 30, fh + 46, fh + 76, fh + 77, fh + 83, fh +75 , fh + 180, fh + 270, fh + 365]
    alphas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    for alpha in alphas:
        for lag in lags:
            df_copy['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store_nbr", "family"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return df_copy

merged_df = lag_features(merged_df, 15)

merged_df = roll_mean_features(merged_df, 15)

alphas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
merged_df = ewm_features(merged_df, alphas, 15)

def selected_df(dataframe, strore_name, family_name):
    for i, data in enumerate(dataframe):
        if data.store_nbr.values[1] == strore_name:
            data=data[data.family==family_name]
            return data

def one_hot_encode_dataframe(dataframe):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    cat_cols = [col for col in cat_cols if col not in ["family"]]

    ohe = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoded = ohe.fit_transform(dataframe[cat_cols]).toarray()
    X_train_ohe = pd.DataFrame(one_hot_encoded, columns=ohe.get_feature_names_out(cat_cols))
    df_merged_train = pd.concat([dataframe.reset_index(drop=True), X_train_ohe.reset_index(drop=True)], axis=1)
    df_merged_train = df_merged_train.drop(cat_cols, axis=1)
    return df_merged_train

merged_df= one_hot_encode_dataframe(merged_df)

def target_log_transmission(dataframe,target):
    dataframe[target]=np.log1p(dataframe[target].values)
    dataframe = dataframe.sort_values("date").reset_index(drop = True)
    dataframe.index=dataframe["date"]
    dataframe= dataframe.fillna(0)
    return dataframe

merged_df= target_log_transmission(merged_df,"sales")
