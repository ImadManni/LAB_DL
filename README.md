# LAB_DL - Stock Price Prediction with LSTM

Deep Learning Lab project for stock price prediction using LSTM neural networks.

## 📋 Project Overview

This project implements an advanced stock price prediction system using Long Short-Term Memory (LSTM) neural networks. It includes both a command-line script and an interactive Streamlit web application.

## 🚀 Features

- **LSTM-based Stock Price Prediction**: Multi-layer LSTM model with configurable architecture
- **Interactive Streamlit Interface**: User-friendly web app with multiple tabs for training, visualization, and analysis
- **Advanced Configuration**: Customizable model parameters, optimizers, and preprocessing options
- **Comprehensive Analysis**: Training history, error distribution, correlation matrices, and performance metrics
- **Data Exploration**: Statistical analysis and time series visualization

## 📁 Project Structure

```
LAB_DL/
├── stock_predictor.py          # Main Streamlit application
├── Lab_lstm (2).py            # Command-line LSTM training script
├── app.py                      # Alternative Streamlit app with yfinance
├── Lab_ANN.py                  # ANN implementation
├── dataset/                    # Training and test datasets
│   ├── train_dataset.csv
│   └── test_dataset.csv
└── README.md                   # This file
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/ImadManni/LAB_DL.git
cd LAB_DL
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
- **Windows**:
```bash
venv\Scripts\activate
```
- **Linux/Mac**:
```bash
source venv/bin/activate
```

4. Install required packages:
```bash
pip install numpy pandas matplotlib scikit-learn keras tensorflow streamlit seaborn
```

## 📖 Usage

### Streamlit Application

Run the interactive web application:
```bash
streamlit run stock_predictor.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- Configure model architecture (layers, units, dropout)
- Adjust training parameters (epochs, batch size, learning rate)
- Select different features (Open, High, Low, Close, Volume)
- Choose preprocessing scalers
- View training history and predictions
- Export results as CSV

### Command-Line Script

Run the basic LSTM training script:
```bash
python "Lab_lstm (2).py"
```

This will:
- Load training and test datasets
- Train an LSTM model
- Generate predictions
- Save the model as `tata_model.h5`
- Display prediction plot

## ⚙️ Configuration

### Model Architecture
- **LSTM Layers**: 1-5 layers (default: 4)
- **LSTM Units**: 10-200 per layer (default: 50)
- **Dropout Rate**: 0.0-0.5 (default: 0.2)
- **Sequence Length**: 10-120 timesteps (default: 60)

### Training Parameters
- **Optimizers**: Adam, RMSprop, SGD, Adagrad
- **Learning Rate**: 0.0001-0.1 (default: 0.001)
- **Epochs**: 1-100 (default: 10)
- **Batch Size**: 8-256 (default: 32)
- **Validation Split**: 0.0-0.3

### Preprocessing
- **Scalers**: MinMaxScaler, StandardScaler, RobustScaler

## 📊 Model Performance Metrics

The application calculates:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R² Score** (Coefficient of Determination)

## 🎯 Why LSTM for Stock Prediction?

LSTM (Long Short-Term Memory) networks are ideal for stock price prediction because:

1. **Sequential Memory**: Maintains long-term dependencies in time series data
2. **Temporal Patterns**: Captures both short-term and long-term trends
3. **Non-linear Relationships**: Learns complex patterns in price movements
4. **Sequence Processing**: Naturally handles sequential data structure
5. **Adaptability**: Adjusts to changing market conditions

## 📝 Dataset

The project uses stock price data with the following columns:
- Date
- Open
- High
- Low
- Close
- Total Trade Quantity
- Turnover (Lacs)

If datasets are not found locally, the application will automatically download sample data from the repository.

## 💾 Model Saving

Trained models are saved as:
- `stock_price_predictor.h5` (from Streamlit app)
- `tata_model.h5` (from command-line script)

## 📚 References

- [Predicting Stock Prices using Keras LSTM Model](https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233)

## 👤 Author

**Imad Manni** - EMSI 2025/2026

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## ⚠️ Disclaimer

This project is for educational purposes only. Stock price prediction is inherently uncertain, and past performance does not guarantee future results. Always do your own research before making investment decisions.

