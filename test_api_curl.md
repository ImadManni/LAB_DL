# API Test Examples

## Base URL
```
http://localhost:8000
```

## 1. Health Check
```bash
curl http://localhost:8000/health
```

## 2. Root Endpoint
```bash
curl http://localhost:8000/
```

## 3. Datasets Info
```bash
curl http://localhost:8000/datasets/info
```

## 4. Model Info
```bash
curl http://localhost:8000/model/info
```

## 5. Train Model
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d "{
    \"epochs\": 10,
    \"batch_size\": 32,
    \"sequence_length\": 60,
    \"train_path\": \"dataset/train_dataset.csv\"
  }"
```

## 6. Predict (Full Test Dataset)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_path\": \"tata_model.h5\",
    \"test_path\": \"dataset/test_dataset.csv\",
    \"train_path\": \"dataset/train_dataset.csv\"
  }"
```

## 7. Single Prediction
```bash
curl -X POST "http://localhost:8000/predict/single" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_path\": \"tata_model.h5\",
    \"sequence\": [100.0, 101.0, 102.0, ...],
    \"train_path\": \"dataset/train_dataset.csv\"
  }"
```

## Python Requests Examples

### Health Check
```python
import requests
response = requests.get("http://localhost:8000/health")
print(response.json())
```

### Train Model
```python
import requests
payload = {
    "epochs": 10,
    "batch_size": 32,
    "sequence_length": 60,
    "train_path": "dataset/train_dataset.csv"
}
response = requests.post("http://localhost:8000/train", json=payload)
print(response.json())
```

### Predict
```python
import requests
payload = {
    "model_path": "tata_model.h5",
    "test_path": "dataset/test_dataset.csv",
    "train_path": "dataset/train_dataset.csv"
}
response = requests.post("http://localhost:8000/predict", json=payload)
result = response.json()
print(f"MSE: {result['mse']}")
print(f"MAE: {result['mae']}")
print(f"Predictions: {result['predictions'][:5]}")
```

### Single Prediction
```python
import requests
import pandas as pd

df = pd.read_csv("dataset/train_dataset.csv")
sequence = df.iloc[-60:, 1].tolist()

payload = {
    "model_path": "tata_model.h5",
    "sequence": sequence,
    "train_path": "dataset/train_dataset.csv"
}
response = requests.post("http://localhost:8000/predict/single", json=payload)
print(response.json())
```

