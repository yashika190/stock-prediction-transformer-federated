# ======================================
#  IMPORTS
# ======================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tensorflow.keras.layers import (
    Input,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Dropout
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from scipy.signal import savgol_filter


# ======================================
# 🔒 REPRODUCIBILITY
# ======================================
np.random.seed(42)
tf.random.set_seed(42)


# ======================================
# 📥 LOAD DATA
# ======================================
def load_data(filepath):

    df = pd.read_csv(filepath)

    # remove extra spaces
    df.columns = df.columns.str.strip()

    print("\nDataset Columns:")
    print(df.columns)

    # ==================================
    # DATE FORMATTING
    # ==================================
    df['Date'] = pd.to_datetime(df['Date'])

    # ==================================
    # SORT BY DATE
    # ==================================
    df = df.sort_values('Date').reset_index(drop=True)

    # ==================================
    # HANDLE MISSING VALUES
    # ==================================
    df = df.ffill()

    # ==================================
    # REMOVE DUPLICATES
    # ==================================
    df = df.drop_duplicates()

    # ==================================
    # DROP NULL VALUES
    # ==================================
    df = df.dropna()

    return df


# ======================================
# 🎯 FEATURE ENGINEERING
# ======================================
def add_features(df):

    df = df.copy()

    # ==================================
    # RETURNS
    # ==================================
    df['Return'] = df['Close'].pct_change()

    # ==================================
    # ENGINEERED FEATURES
    # ==================================
    df['HL_Spread'] = (
        df['High'] - df['Low']
    )

    df['OC_Spread'] = (
        df['Close'] - df['Open']
    )

    df['MA5'] = (
        df['Close']
        .rolling(5)
        .mean()
    )

    # ==================================
    # TARGET
    # ==================================
    df['Target'] = (
        df['Return']
        .rolling(3)
        .mean()
        .shift(-1)
    )

    df = df.dropna().reset_index(drop=True)

    return df


# ======================================
# 📊 FEATURES
# ======================================
features = [
    'Open',
    'High',
    'Low',
    'Close',
    'HL_Spread',
    'OC_Spread',
    'MA5'
]

time_step = 50


# ======================================
# 📊 CREATE DATASET
# ======================================
def create_dataset(data, target):

    X, Y = [], []

    for i in range(len(data) - time_step):

        X.append(
            data[i:i + time_step]
        )

        Y.append(
            target[i + time_step]
        )

    return np.array(X), np.array(Y)


# ======================================
# 🔄 PREPARE DATA
# ======================================
def prepare_data(df):

    # ==================================
    # FEATURE ENGINEERING
    # ==================================
    df = add_features(df)

    # ==================================
    # TRAIN TEST SPLIT
    # ==================================
    split = int(len(df) * 0.8)

    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    # ==================================
    # SMOOTH TRAIN DATA
    # ==================================
    smooth_cols = [
        'Open',
        'High',
        'Low',
        'Close'
    ]

    for col in smooth_cols:

        if len(train_df[col]) >= 11:

            train_df[col] = savgol_filter(
                train_df[col],
                window_length=11,
                polyorder=2
            )

    # ==================================
    # SCALING
    # ==================================
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_df[features]
    )

    test_scaled = scaler.transform(
        test_df[features]
    )

    # ==================================
    # CREATE SEQUENCES
    # ==================================
    X_train, y_train = create_dataset(
        train_scaled,
        train_df['Target'].values
    )

    X_test, y_test = create_dataset(
        test_scaled,
        test_df['Target'].values
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        test_df
    )


# ======================================
# 🤖 TRANSFORMER MODEL
# ======================================
def create_model(input_shape):

    inputs = Input(shape=input_shape)

    # ==================================
    # INPUT PROJECTION
    # ==================================
    x = Dense(256)(inputs)

    # ==================================
    # POSITIONAL EMBEDDING
    # ==================================
    positions = tf.range(
        start=0,
        limit=input_shape[0],
        delta=1
    )

    pos_embedding = tf.keras.layers.Embedding(
        input_dim=input_shape[0],
        output_dim=256
    )(positions)

    pos_embedding = tf.expand_dims(
        pos_embedding,
        axis=0
    )

    x = x + pos_embedding

    # ==================================
    # TRANSFORMER BLOCKS
    # ==================================
    for _ in range(3):

        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=64
        )(x, x)

        attention = Dropout(0.2)(
            attention
        )

        x = LayerNormalization(
            epsilon=1e-6
        )(x + attention)

        ffn = Dense(
            512,
            activation='relu'
        )(x)

        ffn = Dropout(0.2)(ffn)

        ffn = Dense(256)(ffn)

        x = LayerNormalization(
            epsilon=1e-6
        )(x + ffn)

    # ==================================
    # OUTPUT
    # ==================================
    x = GlobalAveragePooling1D()(x)

    x = Dropout(0.2)(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(
            learning_rate=0.0001,
            clipnorm=1.0
        ),
        loss='mse',
        metrics=['mae']
    )

    return model


# ======================================
# 📂 LOAD DATASETS
# ======================================
df1 = load_data(
    r"C:\Users\ASUS\OneDrive\Desktop\python files\infosys.csv"
)

df2 = load_data(
    r"C:\Users\ASUS\OneDrive\Desktop\python files\reliance2.csv"
)

df3 = load_data(
    r"C:\Users\ASUS\OneDrive\Desktop\python files\tcs1.csv"
)


# ======================================
# 📊 PREPARE DATA
# ======================================
X_train1, y_train1, X_test1, y_test1, test1 = prepare_data(df1)

X_train2, y_train2, X_test2, y_test2, test2 = prepare_data(df2)

X_train3, y_train3, X_test3, y_test3, test3 = prepare_data(df3)

input_shape = (
    time_step,
    len(features)
)


# ======================================
# 🏢 CENTRALIZED TRAINING
# ======================================
print("\n==============================")
print("CENTRALIZED TRAINING")
print("==============================")

X_train_all = np.concatenate([
    X_train1,
    X_train2,
    X_train3
])

y_train_all = np.concatenate([
    y_train1,
    y_train2,
    y_train3
])

# ======================================
# SHUFFLE DATA
# ======================================
idx = np.random.permutation(
    len(X_train_all)
)

X_train_all = X_train_all[idx]
y_train_all = y_train_all[idx]

# ======================================
# VALIDATION SPLIT
# ======================================
split = int(
    len(X_train_all) * 0.9
)

X_train_c = X_train_all[:split]
X_val = X_train_all[split:]

y_train_c = y_train_all[:split]
y_val = y_train_all[split:]

# ======================================
# BUILD MODEL
# ======================================
central_model = create_model(
    input_shape
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ======================================
# TRAIN MODEL
# ======================================
central_model.fit(
    X_train_c,
    y_train_c,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)


# ======================================
# 🌐 FEDERATED TRAINING
# ======================================
def federated_training(
    datasets,
    rounds=25,
    local_epochs=2
):

    print("\n==============================")
    print("FEDERATED TRAINING")
    print("==============================")

    global_model = create_model(
        input_shape
    )

    for r in range(rounds):

        print(f"\nRound {r+1}/{rounds}")

        local_weights = []
        client_sizes = []

        for X_client, y_client in datasets:

            local_model = create_model(
                input_shape
            )

            local_model.set_weights(
                global_model.get_weights()
            )

            local_model.fit(
                X_client,
                y_client,
                epochs=local_epochs,
                batch_size=32,
                verbose=0,
                shuffle=True
            )

            local_weights.append(
                local_model.get_weights()
            )

            client_sizes.append(
                len(X_client)
            )

        # ==================================
        # FEDERATED AVERAGING
        # ==================================
        total_size = sum(client_sizes)

        new_weights = []

        for weights in zip(*local_weights):

            weighted_avg = np.zeros_like(
                weights[0]
            )

            for w, size in zip(
                weights,
                client_sizes
            ):

                weighted_avg += (
                    w * (size / total_size)
                )

            new_weights.append(
                weighted_avg
            )

        global_model.set_weights(
            new_weights
        )

    print("\nFederated Training Complete!")

    return global_model


# ======================================
# TRAIN FEDERATED MODEL
# ======================================
fed_model = federated_training([
    (X_train1, y_train1),
    (X_train2, y_train2),
    (X_train3, y_train3)
])


# ======================================
# 📈 EVALUATION
# ======================================
def evaluate(
    model,
    X_test,
    y_test,
    test_df,
    name
):

    pred = model.predict(
        X_test
    ).flatten()

    # ==================================
    # STABILITY CLIPPING
    # ==================================
    pred = np.clip(
        pred,
        -0.2,
        0.2
    )

    # ==================================
    # LAST CLOSE VALUES
    # ==================================
    last_close = test_df[
        'Close'
    ].values[
        time_step:
        time_step + len(pred)
    ]

    # ==================================
    # RECONSTRUCT PRICES
    # ==================================
    pred_price = (
        last_close * (1 + pred)
    )

    true_price = (
        last_close * (1 + y_test)
    )

    # ==================================
    # METRICS
    # ==================================
    rmse = math.sqrt(
        mean_squared_error(
            true_price,
            pred_price
        )
    )

    mae = mean_absolute_error(
        true_price,
        pred_price
    )

    r2 = r2_score(
        true_price,
        pred_price
    )

    # ==================================
    # DIRECTIONAL ACCURACY
    # ==================================
    pred_dir = (
        pred > 0
    )

    true_dir = (
        y_test > 0
    )

    da = np.mean(
        pred_dir == true_dir
    ) * 100

    print(f"\n{name}")

    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")
    print(f"DA   : {da:.2f}%")

    return true_price, pred_price


# ======================================
# 📊 CENTRALIZED RESULTS
# ======================================
print("\n==============================")
print("CENTRALIZED RESULTS")
print("==============================")

tr1, pr1 = evaluate(
    central_model,
    X_test1,
    y_test1,
    test1,
    "Infosys"
)

tr2, pr2 = evaluate(
    central_model,
    X_test2,
    y_test2,
    test2,
    "Reliance"
)

tr3, pr3 = evaluate(
    central_model,
    X_test3,
    y_test3,
    test3,
    "TCS"
)


# ======================================
# 📊 FEDERATED RESULTS
# ======================================
print("\n==============================")
print("FEDERATED RESULTS")
print("==============================")

_, fr1 = evaluate(
    fed_model,
    X_test1,
    y_test1,
    test1,
    "Infosys"
)

_, fr2 = evaluate(
    fed_model,
    X_test2,
    y_test2,
    test2,
    "Reliance"
)

_, fr3 = evaluate(
    fed_model,
    X_test3,
    y_test3,
    test3,
    "TCS"
)


# ======================================
# 📉 VISUALIZATION
# ======================================
plt.figure(figsize=(14, 12))


# ======================================
# INFOSYS
# ======================================
plt.subplot(3,1,1)

plt.plot(
    tr1,
    label='Actual',
    linewidth=2
)

plt.plot(
    pr1,
    label='Centralized',
    linestyle='dashed'
)

plt.plot(
    fr1,
    label='Federated',
    linestyle='dotted'
)

plt.title('Infosys')
plt.legend()
plt.grid(alpha=0.3)


# ======================================
# RELIANCE
# ======================================
plt.subplot(3,1,2)

plt.plot(
    tr2,
    label='Actual',
    linewidth=2
)

plt.plot(
    pr2,
    label='Centralized',
    linestyle='dashed'
)

plt.plot(
    fr2,
    label='Federated',
    linestyle='dotted'
)

plt.title('Reliance')
plt.legend()
plt.grid(alpha=0.3)


# ======================================
# TCS
# ======================================
plt.subplot(3,1,3)

plt.plot(
    tr3,
    label='Actual',
    linewidth=2
)

plt.plot(
    pr3,
    label='Centralized',
    linestyle='dashed'
)

plt.plot(
    fr3,
    label='Federated',
    linestyle='dotted'
)

plt.title('TCS')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

