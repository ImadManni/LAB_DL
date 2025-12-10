from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
import os

app = FastAPI(title="LSTM Stock Price Prediction API", version="1.0.0")

class TrainingRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    sequence_length: int = 60
    train_path: str = "dataset/train_dataset.csv"
    test_path: Optional[str] = None

class PredictionRequest(BaseModel):
    model_path: str = "tata_model.h5"
    test_path: str = "dataset/test_dataset.csv"
    train_path: str = "dataset/train_dataset.csv"

class SinglePredictionRequest(BaseModel):
    model_path: str = "tata_model.h5"
    sequence: List[float]
    train_path: str = "dataset/train_dataset.csv"

def load_and_prepare_data(train_path: str, sequence_length: int = 60):
    dataset_train = pd.read_csv(train_path)
    training_set = dataset_train.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    X_train = []
    y_train = []
    for i in range(sequence_length, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-sequence_length:i, 0])
        y_train.append(training_set_scaled[i, 0])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, sc, dataset_train

def build_lstm_model(input_shape, units: int = 50, dropout: float = 0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.get("/")
async def root():
    return {"message": "LSTM Stock Price Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/train")
async def train_model(request: TrainingRequest):
    try:
        if not os.path.exists(request.train_path):
            raise HTTPException(status_code=404, detail=f"Training file not found: {request.train_path}")
        
        X_train, y_train, sc, dataset_train = load_and_prepare_data(
            request.train_path, 
            request.sequence_length
        )
        
        model = build_lstm_model((X_train.shape[1], 1))
        
        history = model.fit(
            X_train, 
            y_train, 
            epochs=request.epochs, 
            batch_size=request.batch_size,
            verbose=0
        )
        
        model_path = "tata_model.h5"
        model.save(model_path)
        
        final_loss = float(history.history['loss'][-1])
        
        return {
            "status": "success",
            "model_path": model_path,
            "training_samples": int(X_train.shape[0]),
            "final_loss": final_loss,
            "epochs": request.epochs,
            "batch_size": request.batch_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
        
        if not os.path.exists(request.test_path):
            raise HTTPException(status_code=404, detail=f"Test file not found: {request.test_path}")
        
        if not os.path.exists(request.train_path):
            raise HTTPException(status_code=404, detail=f"Train file not found: {request.train_path}")
        
        model = load_model(request.model_path)
        
        dataset_train = pd.read_csv(request.train_path)
        dataset_test = pd.read_csv(request.test_path)
        real_stock_price = dataset_test.iloc[:, 1:2].values
        
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set = dataset_train.iloc[:, 1:2].values
        sc.fit_transform(training_set)
        
        dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)
        
        X_test = []
        for i in range(60, 60 + len(dataset_test)):
            X_test.append(inputs[i-60:i, 0])
        
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predicted_stock_price = model.predict(X_test, verbose=0)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        
        predictions = predicted_stock_price.flatten().tolist()
        actual = real_stock_price.flatten().tolist()
        
        mse = float(np.mean((predicted_stock_price - real_stock_price) ** 2))
        mae = float(np.mean(np.abs(predicted_stock_price - real_stock_price)))
        
        return {
            "status": "success",
            "predictions": predictions,
            "actual": actual,
            "mse": mse,
            "mae": mae,
            "num_predictions": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/single")
async def predict_single(request: SinglePredictionRequest):
    try:
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
        
        if len(request.sequence) != 60:
            raise HTTPException(status_code=400, detail="Sequence must contain exactly 60 values")
        
        model = load_model(request.model_path)
        
        dataset_train = pd.read_csv(request.train_path)
        training_set = dataset_train.iloc[:, 1:2].values
        
        sc = MinMaxScaler(feature_range=(0, 1))
        sc.fit_transform(training_set)
        
        sequence_array = np.array(request.sequence).reshape(-1, 1)
        sequence_scaled = sc.transform(sequence_array)
        
        X_pred = sequence_scaled.reshape(1, 60, 1)
        prediction_scaled = model.predict(X_pred, verbose=0)
        prediction = sc.inverse_transform(prediction_scaled)
        
        return {
            "status": "success",
            "prediction": float(prediction[0][0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info(model_path: str = "tata_model.h5", 
                     train_path: str = "dataset/train_dataset.csv",
                     test_path: str = "dataset/test_dataset.csv"):
    exists = os.path.exists(model_path)
    train_exists = os.path.exists(train_path)
    test_exists = os.path.exists(test_path)
    
    info = {
        "model_path": model_path,
        "model_exists": exists,
        "train_path": train_path,
        "train_exists": train_exists,
        "test_path": test_path,
        "test_exists": test_exists
    }
    
    if exists:
        try:
            model = load_model(model_path)
            info["model_summary"] = {
                "layers": len(model.layers),
                "trainable_params": int(model.count_params())
            }
        except:
            pass
    
    return info

@app.get("/datasets/info")
async def datasets_info():
    train_path = "dataset/train_dataset.csv"
    test_path = "dataset/test_dataset.csv"
    
    info = {}
    
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
        info["train"] = {
            "exists": True,
            "rows": int(len(df_train)),
            "columns": list(df_train.columns),
            "shape": list(df_train.shape)
        }
    else:
        info["train"] = {"exists": False}
    
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        info["test"] = {
            "exists": True,
            "rows": int(len(df_test)),
            "columns": list(df_test.columns),
            "shape": list(df_test.shape)
        }
    else:
        info["test"] = {"exists": False}
    
    return info

