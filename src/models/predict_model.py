import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import  create_sequences, get_train_valid_test_split, get_train_valid_test_datasets
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model


close_model = load_model('models/Close_model.h5')
close_model_LSTM=load_model('models/close_LSTM_model.h5')
close_model_GRU=load_model('models/close_GRU_model.h5')


def get_data(dataset):
    data = []
    for element in dataset:
        data.append(element[1].numpy())
    return data


def flatten_data(data):
    flattened = []

    for arr in data[0]:
        for val in arr:
            if val is not None:
                flattened.append(val)

    return np.array(flattened)


def get_real_pred_df(scalar, model, sequencies, labels, indices):
    preds = scalar.inverse_transform(model.predict(sequencies))
    labels = scalar.inverse_transform(labels)
    dataframe = pd.DataFrame(
        {'Real': labels.flatten(), 'Predictions': preds.flatten()})
    dataframe.index = indices
    return dataframe

close_df = pd.read_csv('data/processed/close_df.csv',
                       parse_dates=['Date'], index_col='Date')

close_scaler = MinMaxScaler()
close_scaled_data = close_scaler.fit_transform(close_df)

close_sequencies, close_labels = create_sequences(close_scaled_data, 56)

close_X_train, close_X_valid, close_X_test, close_y_train, close_y_valid, close_y_test = get_train_valid_test_split(
    close_sequencies, close_labels)


close_train_ds, close_valid_ds, close_test_ds = get_train_valid_test_datasets(
    close_X_train, close_X_valid, close_X_test, close_y_train, close_y_valid, close_y_test)

close_preds = close_scaler.inverse_transform(
    close_model.predict(close_test_ds))

close_df_predictions = get_real_pred_df(close_scaler, close_model,close_sequencies[-200:],close_labels[-200:],close_df[-200:].index)

close_df_predictions.plot(grid=True, marker='.', figsize=(18, 6), title='Deep RNN Predictions')
plt.savefig('reports/figures/DeepRNN_Predictions.png', dpi=300)

close_preds_LSTM = close_scaler.inverse_transform(
    close_model_LSTM.predict(close_test_ds))

close_df_predictions_LSTM = get_real_pred_df(close_scaler, close_model_LSTM,close_sequencies[-200:],close_labels[-200:],close_df[-200:].index)

close_df_predictions_LSTM.plot(grid=True, marker='.', figsize=(18, 6), title='LSTM Predictions')
plt.savefig('reports/figures/LSTM_Predictions.png', dpi=300)

close_preds_GRU = close_scaler.inverse_transform(
    close_model_GRU.predict(close_test_ds))

close_df_predictions_GRU = get_real_pred_df(close_scaler, close_model_GRU,close_sequencies[-200:],close_labels[-200:],close_df[-200:].index)

close_df_predictions_GRU.plot(grid=True, marker='.', figsize=(18, 6), title='GRU Predictions')
plt.savefig('reports/figures/GRU_Predictions.png', dpi=300)

mse_RNN = mean_squared_error(
    close_df_predictions['Real'].values, close_df_predictions['Predictions'].values)
mse_LSTM = mean_squared_error(
    close_df_predictions_LSTM['Real'].values, close_df_predictions_LSTM['Predictions'].values)
mse_GRU = mean_squared_error(
    close_df_predictions_GRU['Real'].values, close_df_predictions_GRU['Predictions'].values)

rmse_RNN = np.sqrt(mse_RNN)
rmse_LSTM = np.sqrt(mse_LSTM)
rmse_GRU = np.sqrt(mse_GRU)

mae_RNN = mean_absolute_error(
    close_df_predictions['Real'].values, close_df_predictions['Predictions'].values)
mae_LSTM = mean_absolute_error(
    close_df_predictions_LSTM['Real'].values, close_df_predictions_LSTM['Predictions'].values)
mae_GRU = mean_absolute_error(
    close_df_predictions_GRU['Real'].values, close_df_predictions_GRU['Predictions'].values)

models = ['Deep RNN', 'LSTM', 'GRU']

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.bar(models, [mse_RNN, mse_LSTM, mse_GRU], color='blue')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')

plt.subplot(3, 1, 2)
plt.bar(models, [rmse_RNN, rmse_LSTM, rmse_GRU], color='green')
plt.title('Root Mean Squared Error (RMSE)')
plt.ylabel('RMSE')

plt.subplot(3, 1, 3)
plt.bar(models, [mae_RNN, mae_LSTM, mae_GRU], color='orange')
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('MAE')

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig('reports/figures/Model Comparison.png', dpi=300)