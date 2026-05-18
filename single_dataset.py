import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import (Input, Dense, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D,
                                     Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.signal import savgol_filter

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("sampledataset.csv")
df.columns = df.columns.str.strip()

# ==============================
# 2. NOISE REMOVAL
# ==============================
df['Close'] = savgol_filter(df['Close'], 11, 2)
df['Open'] = savgol_filter(df['Open'], 11, 2)
df['High'] = savgol_filter(df['High'], 11, 2)
df['Low'] = savgol_filter(df['Low'], 11, 2)

# ==============================
# 3. FEATURE ENGINEERING
# ==============================
df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
df['MA10'] = df['Close'].rolling(10).mean()
df['Volatility'] = df['Return'].rolling(10).std()
df['Momentum'] = df['Close'] - df['Close'].shift(5)
df['Target'] = df['Return'].shift(-1)

df = df.dropna()

features = ['Close','High','Low','Open','MA10','Volatility','Momentum']
data = df[features].values
target = df['Target'].values

# ==============================
# CORRELATION MATRIX (TABLE)
# ==============================
corr_matrix = df[features + ['Target']].corr()

print("\nCorrelation Matrix:\n")
print(corr_matrix)

# ==============================
# HEATMAP (VISUALIZATION)
# ==============================


plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")

plt.show(block=False)   # show without stopping code
plt.pause(3)            # display for 3 seconds
plt.close()             # then continue

# ==============================
# 4. TRAIN TEST SPLIT
# ==============================
train_size = int(len(data) * 0.8)

train_data = data[:train_size]
test_data = data[train_size:]

train_target = target[:train_size]
test_target = target[train_size:]

# ==============================
# 5. SCALING
# ==============================
scaler = MinMaxScaler()

train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# ==============================
# 6. CREATE SEQUENCES
# ==============================
def create_dataset(dataset, target_array, time_step=60):

    X, Y = [], []

    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i+time_step])
        Y.append(target_array[i+time_step])

    return np.array(X), np.array(Y)

time_step = 50

X_train_full, y_train_full = create_dataset(train_data, train_target, time_step)
X_test, y_test = create_dataset(test_data, test_target, time_step)

# ==============================
# 7. TRAIN VALIDATION SPLIT
# ==============================
val_split = int(len(X_train_full) * 0.9)

X_val = X_train_full[val_split:]
y_val = y_train_full[val_split:]

X_train = X_train_full[:val_split]
y_train = y_train_full[:val_split]

# ==============================
# 8. TRANSFORMER MODEL
# ==============================
def create_transformer_model(input_shape,
                             d_model=128,
                             num_heads=4,
                             num_layers=2,
                             dropout_rate=0.2):

    inputs = Input(shape=input_shape)

    x = Dense(d_model)(inputs)

    # Positional Encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)

    pos_embedding_layer = tf.keras.layers.Embedding(
        input_dim=input_shape[0],
        output_dim=d_model
    )

    pos_embeddings = pos_embedding_layer(positions)

    pos_embeddings = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x, axis=0)
    )(pos_embeddings)

    x = x + pos_embeddings

    # Transformer blocks
    for _ in range(num_layers):

        attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )(x, x)

        attention = Dropout(dropout_rate)(attention)

        x = LayerNormalization(epsilon=1e-6)(x + attention)

        ffn = Dense(d_model*2, activation='relu')(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)

        x = LayerNormalization(epsilon=1e-6)(x + ffn)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate= 0.00025),
        loss='mse',
        metrics=['mae']
    )

    return model

# ==============================
# 9. CENTRALIZED TRAINING
# ==============================
central_model = create_transformer_model(
    (time_step, X_train.shape[2])
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True
)

central_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=2
)

# ==============================
# 10. FEDERATED TRAINING
# ==============================
def federated_training(X_train, y_train, input_shape,
                       rounds=150,
                       num_clients=5,
                       local_epochs=1):

    print("\nStarting Federated Training")

    global_model = create_transformer_model(input_shape)

    # Create clients
    X_clients = np.array_split(X_train, num_clients)
    y_clients = np.array_split(y_train, num_clients)

    clients = list(zip(X_clients, y_clients))

    for r in range(rounds):

        print(f"Federated Round {r+1}/{rounds}")

        local_weights = []
        client_sizes = []

        for X_c, y_c in clients:

            local_model = create_transformer_model(input_shape)

            local_model.set_weights(global_model.get_weights())

            local_model.fit(
                X_c,
                y_c,
                epochs=local_epochs,
                batch_size=32,
                verbose=0
            )


            local_weights.append(local_model.get_weights())
            client_sizes.append(len(X_c))

        total_size = sum(client_sizes)

        new_weights = []

        for weights in zip(*local_weights):

            weighted_avg = np.zeros_like(weights[0])

            for w, size in zip(weights, client_sizes):
                weighted_avg += w * (size / total_size)

            new_weights.append(weighted_avg)

        global_model.set_weights(new_weights)

    print("Federated Training Complete")

    return global_model

# ==============================
# RUN FEDERATED
# ==============================
fed_model = federated_training(
    X_train,
    y_train,
    (time_step, X_train.shape[2]),
    rounds=70,
    num_clients=5,
    local_epochs=3
)

# ==============================
# 11. EVALUATION
# ==============================
def evaluate_model(model, name):

    pred_return = model.predict(X_test).flatten()

    last_close = df['Close'].values[train_size + time_step:]

    pred_price = last_close * np.exp(pred_return)
    true_price = last_close * np.exp(y_test)

    mse = mean_squared_error(true_price, pred_price)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(true_price, pred_price)
    mape = np.mean(np.abs((true_price - pred_price) / (true_price + 1e-8))) * 100
    r2 = r2_score(true_price, pred_price)

    true_direction = (y_test > 0).astype(int)
    pred_direction = (pred_return >  0.001).astype(int)

    directional_accuracy = np.mean(true_direction == pred_direction) * 100

    print(f"\n{name} Results")
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("MAPE:", mape,"%")
    print("R2 :", r2)
    print("Directional Accuracy:", directional_accuracy,"%")

    return true_price, pred_price

# ==============================
# 12. COMPARE
# ==============================
# Get outputs correctly
true_price, central_pred = evaluate_model(central_model, "Centralized Transformer")
_, fed_pred = evaluate_model(fed_model, "Federated Transformer")  # ignore true_price again

# Plot everything
plt.figure(figsize=(12,6))

plt.plot(true_price, label="Actual Price", linewidth=2)
plt.plot(central_pred, label="Centralized", linewidth=2)
plt.plot(fed_pred, label="Federated", linewidth=2)

plt.title("Centralized vs Federated Transformer")
plt.xlabel("Time Step")
plt.ylabel("Price")

plt.legend()
plt.grid(alpha=0.3)

plt.show()
