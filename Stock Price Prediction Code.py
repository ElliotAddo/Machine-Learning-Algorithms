import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error

# Load the dataset
data = pd.read_csv("C:/Users/hp/OneDrive/Desktop/Final GW/code/work/Dataset.csv")  # Replace 'data.csv' with the actual filename


# Preprocess the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close']])

# Split the data into independent and dependent variables
X = scaled_data[:, :-1]  # Independent variables (open, high, low)
y = scaled_data[:, -1]   # Dependent variable (close)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
# KNN
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# SVM
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

# LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Reshape for LSTM
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)

# ANN
ann_model = Sequential()
ann_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
ann_model.add(Dense(32, activation='relu'))
ann_model.add(Dense(1))
ann_model.compile(optimizer='adam', loss='mse')
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Make predictions
# KNN
knn_pred = knn_model.predict(X_test)

# SVM
svm_pred = svm_model.predict(X_test)

# LSTM
lstm_pred = lstm_model.predict(X_test_lstm)
lstm_pred = lstm_pred.reshape((lstm_pred.shape[0],))

# ANN
ann_pred = ann_model.predict(X_test)
ann_pred = ann_pred.reshape((ann_pred.shape[0],))

# Inverse transform the predictions to get actual prices
knn_pred_actual = scaler.inverse_transform(np.hstack((X_test, knn_pred.reshape(-1, 1))))[:, -1]
svm_pred_actual = scaler.inverse_transform(np.hstack((X_test, svm_pred.reshape(-1, 1))))[:, -1]
lstm_pred_actual = scaler.inverse_transform(np.hstack((X_test, lstm_pred.reshape(-1, 1))))[:, -1]
ann_pred_actual = scaler.inverse_transform(np.hstack((X_test, ann_pred.reshape(-1, 1))))[:, -1]
y_test_actual = scaler.inverse_transform(np.hstack((X_test, y_test.reshape(-1, 1))))[:, -1]

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(y_test_actual, label='Actual Close Price', color='black')
plt.plot(knn_pred_actual, label='KNN Predicted Close Price', color='#ff7f0e')
plt.plot(svm_pred_actual, label='SVM Predicted Close Price', color='#1f77b4')
plt.plot(lstm_pred_actual, label='LSTM Predicted Close Price', color='#2ca02c')
plt.plot(ann_pred_actual, label='ANN Predicted Close Price', color='#d62728')
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price')
plt.legend()
plt.show()

# Calculate RMSE for each model
knn_rmse = np.sqrt(mean_squared_error(y_test_actual, knn_pred_actual))
svm_rmse = np.sqrt(mean_squared_error(y_test_actual, svm_pred_actual))
lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_pred_actual))
ann_rmse = np.sqrt(mean_squared_error(y_test_actual, ann_pred_actual))
'''
print("KNN RMSE:", knn_rmse)
print("SVM RMSE:", svm_rmse)
print("LSTM RMSE:", lstm_rmse)
print("ANN RMSE:", ann_rmse)
'''
# Plot the results for each model separately
plt.figure(figsize=(14, 20))

# Plot for KNN
plt.subplot(4, 1, 1)
plt.plot(y_test_actual, label='Actual Close Price', color='black')
plt.plot(knn_pred_actual, label='KNN Predicted Close Price', color='#ff7f0e')
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('KNN: Actual vs Predicted Close Price')
plt.legend()

plt.figure(figsize=(14, 20))
# Plot for SVM
plt.subplot(4, 1, 2)
plt.plot(y_test_actual, label='Actual Close Price', color='black')
plt.plot(svm_pred_actual, label='SVM Predicted Close Price', color='#1f77b4')
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('SVM: Actual vs Predicted Close Price')
plt.legend()

plt.figure(figsize=(14, 20))
# Plot for LSTM
plt.subplot(4, 1, 3)
plt.plot(y_test_actual, label='Actual Close Price',color='black')
plt.plot(lstm_pred_actual, label='LSTM Predicted Close Price', color='#2ca02c')
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('LSTM: Actual vs Predicted Close Price')
plt.legend()

plt.figure(figsize=(14, 20))
# Plot for ANN
plt.subplot(4, 1, 4)
plt.plot(y_test_actual, label='Actual Close Price', color='black')
plt.plot(ann_pred_actual, label='ANN Predicted Close Price', color='#d62728')
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('ANN: Actual vs Predicted Close Price')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate MAPE (Mean Absolute Percentage Error)
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Calculate MBE (Mean Bias Error)
def calculate_mbe(actual, predicted):
    return np.mean(actual - predicted)
'''
# Calculate metrics for each model
print("____________________________________")
metrics = {
    'Model': ['KNN', 'SVM', 'LSTM', 'ANN'],
    'RMSE': [knn_rmse, svm_rmse, lstm_rmse, ann_rmse],
    'MAPE': [calculate_mape(y_test_actual, knn_pred_actual),
             calculate_mape(y_test_actual, svm_pred_actual),
             calculate_mape(y_test_actual, lstm_pred_actual),
             calculate_mape(y_test_actual, ann_pred_actual)],
    'MBE': [calculate_mbe(y_test_actual, knn_pred_actual),
            calculate_mbe(y_test_actual, svm_pred_actual),
            calculate_mbe(y_test_actual, lstm_pred_actual),
            calculate_mbe(y_test_actual, ann_pred_actual)]
}
print("____________________________________")

# Create DataFrame
metrics_df = pd.DataFrame(metrics)

# Format MAPE column as percentage
metrics_df['MAPE'] = metrics_df['MAPE'].map('{:.2f}%'.format)

# Print the DataFrame
print(metrics_df)
print("____________________________________")


'''


# Calculate MSE (Mean Squared Error)
def calculate_mse(actual, predicted):
    return mean_squared_error(actual, predicted)

# Calculate MAE (Mean Absolute Error)
def calculate_mae(actual, predicted):
    return mean_absolute_error(actual, predicted)

# Calculate metrics for each model
print("_________________________________________________________")
metrics = {
    'Model': ['KNN', 'SVM', 'LSTM', 'ANN'],
    'RMSE': [knn_rmse, svm_rmse, lstm_rmse, ann_rmse],
    'MSE': [calculate_mse(y_test_actual, knn_pred_actual),
            calculate_mse(y_test_actual, svm_pred_actual),
            calculate_mse(y_test_actual, lstm_pred_actual),
            calculate_mse(y_test_actual, ann_pred_actual)],
    'MAE': [calculate_mae(y_test_actual, knn_pred_actual),
            calculate_mae(y_test_actual, svm_pred_actual),
            calculate_mae(y_test_actual, lstm_pred_actual),
            calculate_mae(y_test_actual, ann_pred_actual)],
    'MAPE': [calculate_mape(y_test_actual, knn_pred_actual),
             calculate_mape(y_test_actual, svm_pred_actual),
             calculate_mape(y_test_actual, lstm_pred_actual),
             calculate_mape(y_test_actual, ann_pred_actual)],
    'MBE': [calculate_mbe(y_test_actual, knn_pred_actual),
            calculate_mbe(y_test_actual, svm_pred_actual),
            calculate_mbe(y_test_actual, lstm_pred_actual),
            calculate_mbe(y_test_actual, ann_pred_actual)]
}

# Create DataFrame
metrics_df = pd.DataFrame(metrics)

# Format MAPE column as percentage
metrics_df['MAPE'] = metrics_df['MAPE'].map('{:.2f}%'.format)

# Print the DataFrame
print("_________________________________________________________")
print(metrics_df)
print("_________________________________________________________")


# Calculate MAPE (Mean Absolute Percentage Error)
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Calculate MBE (Mean Bias Error)
def calculate_mbe(actual, predicted):
    return np.mean(actual - predicted)
