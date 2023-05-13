# %% [markdown]
# # Python Setup

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %matplotlib inline

# import numpy as np
import pandas as pd
# from natsort import natsorted
# from pprint import pprint
# import sys
import os
import glob
import datetime
from zoneinfo import ZoneInfo

# import statsmodels.api as sm
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.ar_model import ar_select_order
# from statsmodels.graphics.api import qqplot

# %%
# plotting code from plotting.py
# from plotting import plot_func
# marker_styles = ['o', '^', 'v', '+']

# %%
# # https://stackoverflow.com/a/59866006
# from IPython.display import display, HTML

# def force_show_all(dfp):
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
#         display(HTML(dfp.to_html()))

# %%
# force_show_all(dfp_data)

# %% [markdown]
# ***
# # Load and Explore the Data

# %%
data_path = '../saved_data'

# %%
date_fmt = "%Y-%m-%d"
time_fmt = "%H:%M:%S"
# time_hm_fmt = "%H:%M"
datetime_fmt = f"{date_fmt} {time_fmt}"

# %%
# when t_start = t_start.replace(minute=t_start_minute, second=0, microsecond=0) fixed the issue of drifting seconds
dt_end_of_drifting_seconds = datetime.datetime.strptime('2023-05-02 01:22:08', datetime_fmt).replace(tzinfo=ZoneInfo('UTC'))


# %%
def load_data(data_path, with_fname=False):
    dfp_list = []
    for f in glob.glob(os.path.join(data_path, "*.csv")):
        try:
            dfp = pd.read_csv(f)
            # dfp['fname'] = f.split("/")[-1]
            dfp_list.append(dfp)
        except:
            raise ValueError(f'Error loading file {f}')

    dfp = pd.concat(dfp_list)
    dfp['datetime_utc'] = pd.to_datetime(dfp['datetime_utc'], utc=True, format=datetime_fmt)
    dfp['datetime_est'] = dfp['datetime_utc'].dt.tz_convert('US/Eastern')
    dfp['day_of_week_int'] = dfp['datetime_est'].dt.dayofweek
    dfp['day_of_week_str'] = dfp['datetime_est'].dt.day_name()
    # dfp['time'] = dfp['datetime_est'].dt.strftime(time_hm_fmt)
    
    dfp = dfp.sort_values(['datetime_utc'], ascending=[True]).reset_index(drop=True)

    dfp_drift_seconds_records = dfp.loc[( (dt_end_of_drifting_seconds < dfp['datetime_utc']) & (dfp['datetime_utc'].dt.second != 0) )]
    if 0 < dfp_drift_seconds_records.size:
        display(dfp_drift_seconds_records)
        raise ValueError(f'Found {dfp_drift_seconds_records.size} after {dt_end_of_drifting_seconds.strftime(datetime_fmt)} UTC!')

    return dfp


# %%
dfp_data = load_data(data_path)

# %%
dfp_data

# %%
dfp_data.dtypes

# %%
dfp_data['mean_pressure_value'].describe()
