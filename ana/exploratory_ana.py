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
import plotly.graph_objects as go

# %%
# plotting code from plotting.py
from plotting import plot_func
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
# TODO use to draw reference hlines, get from config yaml shared between all scripts
# full_pressure_value = 14000

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
    # Add more date columns
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
# dfp_data

# %%
dfp_data.dtypes

# %%
dfp_data['mean_pressure_value'].describe()

# %% [markdown]
# ### Minute Time Series

# %%
dfp_data.index = pd.DatetimeIndex(dfp_data['datetime_est']).tz_localize(None).to_period('T')

# %%
# TODO
# convert to DatetimeIndex, with T (minute) fequency
# create null rows between min and max datetime if they do not exist
# dfp.index = pd.DatetimeIndex(dfp['datetime_est']).tz_localize(None)
# dfp = dfp.asfreq('T')

# %%
dfp_data

# %%
dfp_data.loc[dfp_data['had_flow'].isnull()]

# %%
plot_objs_ts = {}
plot_objs_ts['minutes'] = {'type': 'scatter',
    'x': dfp_data['datetime_est'], 'y': dfp_data['mean_pressure_value'],
    'c': f'C0', 'ms': '.', 'ls': '',
    'label': None}

# %%
plot_func(plot_objs_ts, 'Minute', 'Mean Pressure Value', fig_size=(12, 8))

# %%
fig = go.Figure()

fig.add_trace(go.Scatter(x=dfp_data['datetime_est'], y=dfp_data['mean_pressure_value']))

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1h",
                     step="hour",
                     stepmode="todate"),
                dict(count=12,
                     label="12h",
                     step="hour",
                     stepmode="todate"),
                dict(count=1,
                     label="1d",
                     step="day",
                     stepmode="backward"),
                dict(count=7,
                     label="1w",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)



fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Pressure DAQ Value",
)

fig.update_xaxes(minor=dict(ticks="inside", showgrid=True))


fig.show()
