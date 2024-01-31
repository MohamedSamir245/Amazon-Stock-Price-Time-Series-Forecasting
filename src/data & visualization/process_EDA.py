import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import create_tensorboard_callback, check_tf_gpu,group_by_year_month,create_sequences,get_train_valid_test_split,get_train_valid_test_datasets
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

check_tf_gpu()

df = pd.read_csv('data/external/AmazonStockPrice.csv',
                 parse_dates=['Date'], index_col='Date')

df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

df = df.drop_duplicates()

df["2023-10-01":"2024-01-26"].drop(['Volume'], axis=1).plot(
    grid=True, marker='.', figsize=(18, 6), title='Amazon Stock Price [2023/10/01:2024/01/26]')
plt.savefig('reports/figures/Stock Price [2023-2024].png', dpi=300)


df.groupby(df.index.year).size().plot(kind="bar", figsize=(
    18, 6), grid=True, title="Number of days per year")
plt.savefig('reports/figures/Number of days per year.png', dpi=300)

df = df["1998":"2023"]

group_by_year_month(df.drop(['Volume'], axis=1)).plot(
    figsize=(18, 6), grid=True, title="Average price per month")
plt.savefig('reports/figures/Average price per month.png', dpi=300)

group_by_year_month(df.drop(['Volume'], axis=1)["2015":"2023"]).plot(
    figsize=(18, 6), grid=True, title="Average price per month [2015:2023]")
plt.savefig('reports/figures/Average price per month [2015-2023].png', dpi=300)


diff_weekly = df[['Open', 'High', 'Low', 'Close', 'Adj Close']].diff(7)[
    "2023-10-01":"2024-01-26"]

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(18, 6))
df[['Open', 'High', 'Low', 'Close', 'Adj Close']
   ]["2023-10-01":"2024-01-26"].plot(ax=axs[0], marker=".")  # original time series
df[['Open', 'High', 'Low', 'Close', 'Adj Close']]["2023-10-01":"2024-01-26"].shift(
    7).plot(ax=axs[0], grid=True, legend=False, linestyle=":")  # lagged
# 7-day difference time series
diff_weekly.plot(ax=axs[1], grid=True, marker=".")

legend_elements = [Line2D([0], [0], color='b', lw=1, linestyle=":", label='Lagged'),
                   Line2D([0], [0], color='b', lw=1, label='Original')]
axs[0].legend(handles=legend_elements, loc='best')
axs[0].set_title('Original and Lagged One Week')
axs[1].set_title('7-Day Difference')

plt.show()
fig.savefig('reports/figures/Weekly Seasonality.png', dpi=300)


diff_mothly = df[['Open', 'High', 'Low', 'Close', 'Adj Close']].diff(30)[
    "2023-01-01":"2024-01-26"]

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(18, 6))
df[['Open', 'High', 'Low', 'Close', 'Adj Close']]["2023-01-01":"2024-01-26"].plot(
    ax=axs[0], legend=False, marker=".")  # original time series
df[['Open', 'High', 'Low', 'Close', 'Adj Close']]["2023-01-01":"2024-01-26"].shift(
    30).plot(ax=axs[0], grid=True, legend=False, linestyle=":")  # lagged
diff_mothly.plot(ax=axs[1], grid=True, marker=".")

legend_elements = [Line2D([0], [0], color='b', lw=1, linestyle=":", label='Lagged'),
                   Line2D([0], [0], color='b', lw=1, label='Original')]
axs[0].legend(handles=legend_elements, loc='best')
axs[0].set_title('Original and Lagged One Month')
axs[1].set_title('30-Days Difference')

plt.show()
fig.savefig('reports/figures/Monthly Seasonality.png', dpi=300)

diff_yearly = df[['Open', 'High', 'Low', 'Close', 'Adj Close']].diff(252)[
    "2020-10-01":"2024-01-26"]

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(18, 6))
df[['Open', 'High', 'Low', 'Close', 'Adj Close']]["2020-10-01":"2024-01-26"].plot(
    ax=axs[0], legend=False, marker=".")  # original time series
df[['Open', 'High', 'Low', 'Close', 'Adj Close']]["2020-10-01":"2024-01-26"].shift(
    252).plot(ax=axs[0], grid=True, legend=False, linestyle=":")  # lagged
diff_yearly.plot(ax=axs[1], grid=True, marker=".")

legend_elements = [Line2D([0], [0], color='b', lw=1, linestyle=":", label='Lagged'),
                   Line2D([0], [0], color='b', lw=1, label='Original')]
axs[0].legend(handles=legend_elements, loc='best')
axs[0].set_title('Original and Lagged One Year')
axs[1].set_title('1-Year Difference')

plt.show()

fig.savefig('reports/figures/Yearly Seasonality.png', dpi=300)

df = df.dropna()

df.drop(['Volume'], axis=1).boxplot(figsize=(18, 6))
plt.savefig('reports/figures/Boxplot.png', dpi=300)

df[['Volume']].boxplot(figsize=(8, 10))
plt.savefig('reports/figures/Boxplot Volume.png', dpi=300)

fig, axs = plt.subplots(2, 3, sharex=True, figsize=(18, 12))

# Iterate over the columns
for i, column in enumerate(df.drop(['Volume'], axis=1).columns):
    # Calculate the subplot position
    row = i // 3
    col = i % 3

    sns.distplot(df[column], ax=axs[row, col], kde=True)

plt.tight_layout()
plt.savefig('reports/figures/Distribution.png', dpi=300)


f=sns.distplot(df['Volume'], kde=True)
f.figure.savefig('reports/figures/Distribution Volume.png', dpi=300)

close_df = df[['Close']]

close_df.to_csv('data/processed/close_df.csv')





