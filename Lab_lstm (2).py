# Lab: Stock price prediction
# Réalisé par: IMAD MANNI EMSI 2025/2026
# Réf: https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

# Step 1: Dataset
# url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
url = 'dataset/train_dataset.csv'
dataset_train = pd.read_csv(url)
training_set = dataset_train.iloc[:, 1:2].values
print(dataset_train)
# Data transformation
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled)
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Step 2: Model
model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2)) # 20% dropout Overfitting
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2)) # 20% dropout Overfitting
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2)) # 20% dropout Overfitting
model.add(LSTM(units=50))
model.add(Dropout(0.2)) # 20% dropout Overfitting = forget 20% of ex data
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error') # optimizer: is adam
# Step 3: train
model.fit(X_train,y_train,epochs=10,batch_size=32) #batch_size= home many set traited at once , epochs= how many time the model will see the data
# Step 4: test
# url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
url = 'dataset/test_dataset.csv'
dataset_test = pd.read_csv(url)
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 60 + len(dataset_test)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Prediction
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
plt.figure(figsize=(12, 6))
plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('prediction_plot.png')
print("Plot saved as 'prediction_plot.png'")
plt.show()
#  save the model
model.save('tata_model.h5')