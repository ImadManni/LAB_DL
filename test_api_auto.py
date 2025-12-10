import requests
import json
import pandas as pd
import time

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_endpoint(method, endpoint, payload=None, description=""):
    try:
        url = f"{BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, json=payload, timeout=300)
        
        print(f"\n✓ {description}")
        print(f"  URL: {method} {url}")
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict):
                if "status" in result:
                    print(f"  Status: {result['status']}")
                if "message" in result:
                    print(f"  Message: {result['message']}")
                if "training_samples" in result:
                    print(f"  Training Samples: {result['training_samples']}")
                    print(f"  Final Loss: {result['final_loss']:.6f}")
                if "num_predictions" in result:
                    print(f"  Predictions: {result['num_predictions']}")
                    print(f"  MSE: {result['mse']:.4f}")
                    print(f"  MAE: {result['mae']:.4f}")
                if "prediction" in result:
                    print(f"  Prediction: {result['prediction']:.4f}")
                if "model_exists" in result:
                    print(f"  Model Exists: {result['model_exists']}")
                if "train" in result:
                    if result['train'].get('exists'):
                        print(f"  Train Dataset: {result['train']['rows']} rows")
                if "test" in result:
                    if result['test'].get('exists'):
                        print(f"  Test Dataset: {result['test']['rows']} rows")
            return True
        else:
            print(f"  Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"\n✗ {description}")
        print(f"  Error: {str(e)}")
        return False

print("\n" + "=" * 60)
print("  LSTM Stock Price Prediction API - Automated Test Suite")
print("=" * 60)

print_section("Basic Endpoints")
test_endpoint("GET", "/health", description="Health Check")
test_endpoint("GET", "/", description="Root Endpoint")
test_endpoint("GET", "/datasets/info", description="Datasets Information")
test_endpoint("GET", "/model/info", description="Model Information")

print_section("Training Endpoint")
train_payload = {
    "epochs": 3,
    "batch_size": 32,
    "sequence_length": 60,
    "train_path": "dataset/train_dataset.csv"
}
train_success = test_endpoint("POST", "/train", train_payload, "Train LSTM Model (3 epochs)")

if train_success:
    time.sleep(1)
    print_section("Model Info After Training")
    test_endpoint("GET", "/model/info", description="Updated Model Information")
    
    print_section("Prediction Endpoints")
    predict_payload = {
        "model_path": "tata_model.h5",
        "test_path": "dataset/test_dataset.csv",
        "train_path": "dataset/train_dataset.csv"
    }
    test_endpoint("POST", "/predict", predict_payload, "Predict on Test Dataset")
    
    try:
        df = pd.read_csv("dataset/train_dataset.csv")
        training_set = df.iloc[:, 1:2].values
        last_60 = training_set[-60:].flatten().tolist()
        
        single_payload = {
            "model_path": "tata_model.h5",
            "sequence": last_60,
            "train_path": "dataset/train_dataset.csv"
        }
        test_endpoint("POST", "/predict/single", single_payload, "Single Prediction")
    except Exception as e:
        print(f"\n✗ Single Prediction")
        print(f"  Error preparing sequence: {str(e)}")
else:
    print("\n⚠ Skipping prediction tests - training failed or model not found")

print("\n" + "=" * 60)
print("  Test Suite Complete")
print("=" * 60 + "\n")

