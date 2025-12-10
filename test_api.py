import requests
import json
import pandas as pd
import numpy as np

BASE_URL = "http://localhost:8000"

def test_health():
    print("=" * 50)
    print("Testing /health endpoint")
    print("=" * 50)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_root():
    print("=" * 50)
    print("Testing / endpoint")
    print("=" * 50)
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_datasets_info():
    print("=" * 50)
    print("Testing /datasets/info endpoint")
    print("=" * 50)
    response = requests.get(f"{BASE_URL}/datasets/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_model_info():
    print("=" * 50)
    print("Testing /model/info endpoint")
    print("=" * 50)
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_train():
    print("=" * 50)
    print("Testing /train endpoint")
    print("=" * 50)
    payload = {
        "epochs": 5,
        "batch_size": 32,
        "sequence_length": 60,
        "train_path": "dataset/train_dataset.csv"
    }
    print(f"Sending request: {json.dumps(payload, indent=2)}")
    print("Training may take a while...")
    response = requests.post(f"{BASE_URL}/train", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()

def test_predict():
    print("=" * 50)
    print("Testing /predict endpoint")
    print("=" * 50)
    payload = {
        "model_path": "tata_model.h5",
        "test_path": "dataset/test_dataset.csv",
        "train_path": "dataset/train_dataset.csv"
    }
    print(f"Sending request: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Number of predictions: {result['num_predictions']}")
        print(f"MSE: {result['mse']:.4f}")
        print(f"MAE: {result['mae']:.4f}")
        print(f"First 5 predictions: {result['predictions'][:5]}")
        print(f"First 5 actual: {result['actual'][:5]}")
    else:
        print(f"Error: {response.text}")
    print()

def test_predict_single():
    print("=" * 50)
    print("Testing /predict/single endpoint")
    print("=" * 50)
    
    try:
        df = pd.read_csv("dataset/train_dataset.csv")
        training_set = df.iloc[:, 1:2].values
        last_60 = training_set[-60:].flatten().tolist()
        
        payload = {
            "model_path": "tata_model.h5",
            "sequence": last_60,
            "train_path": "dataset/train_dataset.csv"
        }
        print(f"Sending request with sequence length: {len(payload['sequence'])}")
        response = requests.post(f"{BASE_URL}/predict/single", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result['status']}")
            print(f"Prediction: {result['prediction']:.4f}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error preparing test data: {e}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("LSTM Stock Price Prediction API - Test Suite")
    print("=" * 50 + "\n")
    
    test_health()
    test_root()
    test_datasets_info()
    test_model_info()
    
    print("\n" + "=" * 50)
    print("NOTE: The following tests require model training first")
    print("=" * 50 + "\n")
    
    user_input = input("Do you want to test training? This may take a while. (y/n): ")
    if user_input.lower() == 'y':
        test_train()
        test_model_info()
        
        user_input = input("Do you want to test predictions? (y/n): ")
        if user_input.lower() == 'y':
            test_predict()
            test_predict_single()
    else:
        print("Skipping training and prediction tests.")
        print("You can test predictions if a model already exists (tata_model.h5)")

