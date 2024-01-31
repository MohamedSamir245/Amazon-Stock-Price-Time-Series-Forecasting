from helper_functions import create_tensorboard_callback, check_tf_gpu, group_by_year_month, create_sequences, get_train_valid_test_split, get_train_valid_test_datasets
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
import numpy as np

close_df = pd.read_csv('data/processed/close_df.csv',
                       parse_dates=['Date'], index_col='Date')

close_scaler = MinMaxScaler()
close_scaled_data = close_scaler.fit_transform(close_df)

close_sequencies, close_labels = create_sequences(close_scaled_data, 56)

close_X_train, close_X_valid, close_X_test, close_y_train, close_y_valid, close_y_test = get_train_valid_test_split(
    close_sequencies, close_labels)


close_train_ds, close_valid_ds, close_test_ds = get_train_valid_test_datasets(
    close_X_train, close_X_valid, close_X_test, close_y_train, close_y_valid, close_y_test)

# callbacks

checkpoint_path = "models/model_checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         monitor="val_mae",
                                                         save_best_only=True)

# Setup EarlyStopping callback to stop training if model's val_mae doesn't improve for 50 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_mae",  # watch the val mae metric
                                                  patience=50,
                                                  restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=30, min_lr=1e-3)


def fit_and_evaluate(model, train_set, valid_set, test_set, learning_rate, tensorborad_name, epochs=200):
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
    history = model.fit(train_set, validation_data=valid_set, epochs=epochs,
                        callbacks=[
                            early_stopping,  # Uncomment this line if you want to use early stopping
                            create_tensorboard_callback("reports/training_logs",
                                                        tensorborad_name),
                            checkpoint_callback,
                            # reduce_lr
                        ])
    valid_loss, valid_mae = model.evaluate(valid_set)
    test_loss, test_mae = model.evaluate(test_set)
    model.save(f"models/{tensorborad_name}_model.h5")
    print(f"=Valid mae: {valid_mae * 1e6}")
    print(f"Test mae: {test_mae * 1e6}")
    print(model.summary())
    return valid_mae * 1e6


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


close_data = get_data(close_test_ds)

close_flattened = close_scaler.inverse_transform([flatten_data(close_data)])

close_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(
        100, input_shape=[None, 1], return_sequences=True),
    tf.keras.layers.SimpleRNN(50, return_sequences=True),
    tf.keras.layers.SimpleRNN(1)
])

fit_and_evaluate(close_model, close_train_ds, close_valid_ds, close_test_ds, learning_rate=0.02, tensorborad_name="Close",epochs=200)

close_model_LSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, input_shape=[None, 1], return_sequences=True),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.SimpleRNN(1)
])

fit_and_evaluate(close_model_LSTM, close_train_ds, close_valid_ds, close_test_ds, learning_rate=0.02, tensorborad_name="close_LSTM",epochs=200)

close_model_GRU = tf.keras.Sequential([
    tf.keras.layers.GRU(64, input_shape=[None, 1], return_sequences=True),
    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(1)
])

fit_and_evaluate(close_model_GRU, close_train_ds, close_valid_ds, close_test_ds, learning_rate=0.02, tensorborad_name="close_GRU",epochs=200)