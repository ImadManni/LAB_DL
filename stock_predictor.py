import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam, RMSprop, SGD, Adagrad
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import streamlit as st
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Advanced Stock Price Predictor", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 Advanced Stock Price Prediction with LSTM</h1>', unsafe_allow_html=True)

if 'model_history' not in st.session_state:
    st.session_state.model_history = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'actual' not in st.session_state:
    st.session_state.actual = None

def load_data(train_path, test_path):
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    try:
        dataset_train = pd.read_csv(train_path)
        dataset_test = pd.read_csv(test_path)
        return dataset_train, dataset_test, False
    except FileNotFoundError:
        st.info("Local dataset not found. Downloading sample data...")
        dataset_train = pd.read_csv('https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv')
        dataset_test = pd.read_csv('https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv')
        dataset_train.to_csv(train_path, index=False)
        dataset_test.to_csv(test_path, index=False)
        return dataset_train, dataset_test, True

def prepare_data(dataset_train, dataset_test, sequence_length, feature_col, scaler_type):
    training_set = dataset_train.iloc[:, feature_col:feature_col+1].values
    
    if scaler_type == "MinMaxScaler":
        sc = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == "StandardScaler":
        sc = StandardScaler()
    else:
        sc = RobustScaler()
    
    training_set_scaled = sc.fit_transform(training_set)
    
    X_train = []
    y_train = []
    for i in range(sequence_length, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-sequence_length:i, 0])
        y_train.append(training_set_scaled[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, sc, dataset_train, dataset_test

def build_model(input_shape, lstm_units, num_layers, dropout_rate, dense_units, optimizer_name, learning_rate):
    model = Sequential()
    
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        if i == 0:
            model.add(LSTM(units=lstm_units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units=lstm_units, return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))
    
    if dense_units > 0:
        model.add(Dense(units=dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(units=1))
    
    if optimizer_name == "Adam":
        opt = Adam(learning_rate=learning_rate)
    elif optimizer_name == "RMSprop":
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == "SGD":
        opt = SGD(learning_rate=learning_rate)
    else:
        opt = Adagrad(learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
    return model

st.sidebar.header("📊 Data Configuration")
train_path = st.sidebar.text_input("Training Data Path", "dataset/train_dataset.csv")
test_path = st.sidebar.text_input("Test Data Path", "dataset/test_dataset.csv")

feature_options = {
    "Open": 1,
    "High": 2,
    "Low": 3,
    "Close": 4,
    "Volume": 5
}
selected_feature = st.sidebar.selectbox("Feature to Predict", list(feature_options.keys()), index=0)
feature_col = feature_options[selected_feature]

st.sidebar.header("🔧 Model Architecture")
num_layers = st.sidebar.slider("Number of LSTM Layers", 1, 5, 4)
lstm_units = st.sidebar.slider("LSTM Units per Layer", 10, 200, 50)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
dense_units = st.sidebar.slider("Dense Layer Units (0 to disable)", 0, 100, 0)
sequence_length = st.sidebar.slider("Sequence Length (Timesteps)", 10, 120, 60)

st.sidebar.header("⚙️ Training Parameters")
optimizer_name = st.sidebar.selectbox("Optimizer", ["Adam", "RMSprop", "SGD", "Adagrad"], index=0)
learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
epochs = st.sidebar.slider("Epochs", 1, 100, 10)
batch_size = st.sidebar.slider("Batch Size", 8, 256, 32)
validation_split = st.sidebar.slider("Validation Split", 0.0, 0.3, 0.0, 0.05)

use_early_stopping = st.sidebar.checkbox("Use Early Stopping", False)
if use_early_stopping:
    patience = st.sidebar.slider("Early Stopping Patience", 1, 20, 5)
    min_delta = st.sidebar.number_input("Min Delta", min_value=0.0, max_value=0.01, value=0.0001, step=0.0001, format="%.4f")

use_lr_reduction = st.sidebar.checkbox("Use Learning Rate Reduction", False)
if use_lr_reduction:
    lr_patience = st.sidebar.slider("LR Reduction Patience", 1, 10, 3)
    lr_factor = st.sidebar.slider("LR Reduction Factor", 0.1, 0.9, 0.5, 0.1)

st.sidebar.header("📈 Preprocessing")
scaler_type = st.sidebar.selectbox("Scaler Type", ["MinMaxScaler", "StandardScaler", "RobustScaler"], index=0)

st.sidebar.header("💾 Model Management")
model_save_name = st.sidebar.text_input("Model Save Name", "stock_price_predictor.h5")
load_existing_model = st.sidebar.checkbox("Load Existing Model", False)
if load_existing_model:
    model_file = st.sidebar.file_uploader("Upload Model File (.h5)", type=['h5'])

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Training & Prediction", "📊 Data Exploration", "📈 Visualizations", "📋 Model Analysis"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Train Model and Predict", type="primary", use_container_width=True):
            with st.spinner("Loading data..."):
                dataset_train, dataset_test, downloaded = load_data(train_path, test_path)
                if downloaded:
                    st.success("Sample data downloaded and saved locally.")
            
            st.subheader("📁 Dataset Information")
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            with info_col1:
                st.metric("Training Samples", len(dataset_train))
            with info_col2:
                st.metric("Test Samples", len(dataset_test))
            with info_col3:
                st.metric("Training Features", len(dataset_train.columns))
            with info_col4:
                st.metric("Selected Feature", selected_feature)
            
            st.subheader("Training Data Preview")
            st.dataframe(dataset_train.head(10), use_container_width=True)
            
            st.subheader("Test Data Preview")
            st.dataframe(dataset_test.head(10), use_container_width=True)
            
            with st.spinner("Preparing data..."):
                X_train, y_train, sc, dataset_train, dataset_test = prepare_data(
                    dataset_train, dataset_test, sequence_length, feature_col, scaler_type
                )
                st.session_state.scaler = sc
            
            st.info(f"Training shape: {X_train.shape}, Target shape: {y_train.shape}")
            
            with st.spinner("Building model..."):
                model = build_model(
                    (X_train.shape[1], 1), lstm_units, num_layers, 
                    dropout_rate, dense_units, optimizer_name, learning_rate
                )
            
            st.subheader("Model Architecture")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))
            
            callbacks = []
            if use_early_stopping:
                callbacks.append(EarlyStopping(monitor='loss', patience=patience, min_delta=min_delta, restore_best_weights=True))
            if use_lr_reduction:
                callbacks.append(ReduceLROnPlateau(monitor='loss', patience=lr_patience, factor=lr_factor, min_lr=1e-7))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Training the model..."):
                status_text.text("Training in progress...")
                history = model.fit(
                    X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_split=validation_split if validation_split > 0 else None,
                    callbacks=callbacks,
                    verbose=0
                )
                progress_bar.progress(100)
                status_text.text("Training completed!")
                st.session_state.model_history = history.history
                st.session_state.trained_model = model
            
            st.success(f"✅ Model trained successfully!")
            
            if validation_split > 0:
                col_loss1, col_loss2 = st.columns(2)
                with col_loss1:
                    st.line_chart(pd.DataFrame({'Loss': history.history['loss'], 'Val Loss': history.history.get('val_loss', [])}))
                with col_loss2:
                    st.line_chart(pd.DataFrame({'MAE': history.history['mae'], 'Val MAE': history.history.get('val_mae', [])}))
            
            with st.spinner("Making predictions..."):
                real_stock_price = dataset_test.iloc[:, feature_col:feature_col+1].values
                dataset_total = pd.concat((dataset_train.iloc[:, feature_col], dataset_test.iloc[:, feature_col]), axis=0)
                inputs = dataset_total[len(dataset_total) - len(dataset_test) - sequence_length:].values
                inputs = inputs.reshape(-1, 1)
                inputs = sc.transform(inputs)
                
                X_test = []
                for i in range(sequence_length, sequence_length + len(dataset_test)):
                    X_test.append(inputs[i-sequence_length:i, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                
                predicted_stock_price = model.predict(X_test, verbose=0)
                predicted_stock_price = sc.inverse_transform(predicted_stock_price)
                
                st.session_state.predictions = predicted_stock_price
                st.session_state.actual = real_stock_price
            
            st.subheader("📊 Prediction Results")
            
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(real_stock_price, color='#1f77b4', label='Actual Stock Price', linewidth=2.5, alpha=0.8)
            ax.plot(predicted_stock_price, color='#2ca02c', label='Predicted Stock Price', linewidth=2.5, alpha=0.8)
            ax.fill_between(range(len(real_stock_price)), real_stock_price.flatten(), predicted_stock_price.flatten(), 
                           alpha=0.3, color='gray', label='Error Region')
            ax.set_title(f'{selected_feature} Stock Price Prediction', fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Time Step', fontsize=14)
            ax.set_ylabel('Stock Price', fontsize=14)
            ax.legend(fontsize=12, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Actual Price (Last)", f"${real_stock_price[-1][0]:.2f}")
            with metric_col2:
                st.metric("Predicted Price (Last)", f"${predicted_stock_price[-1][0]:.2f}")
            with metric_col3:
                error_pct = abs((predicted_stock_price[-1][0] - real_stock_price[-1][0]) / real_stock_price[-1][0] * 100)
                st.metric("Error %", f"{error_pct:.2f}%")
            with metric_col4:
                direction = "↑" if predicted_stock_price[-1][0] > real_stock_price[-1][0] else "↓"
                st.metric("Direction", direction)
            
            comparison_df = pd.DataFrame({
                'Actual': real_stock_price.flatten(),
                'Predicted': predicted_stock_price.flatten(),
                'Difference': (predicted_stock_price.flatten() - real_stock_price.flatten()),
                'Error %': (abs(predicted_stock_price.flatten() - real_stock_price.flatten()) / real_stock_price.flatten() * 100)
            })
            
            st.subheader("📋 Detailed Comparison Table")
            st.dataframe(
                comparison_df.style.format({
                    'Actual': '${:.2f}', 
                    'Predicted': '${:.2f}', 
                    'Difference': '${:.2f}',
                    'Error %': '{:.2f}%'
                }),
                use_container_width=True
            )
            
            mse = np.mean((real_stock_price - predicted_stock_price) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(real_stock_price - predicted_stock_price))
            mape = np.mean(np.abs((real_stock_price - predicted_stock_price) / real_stock_price)) * 100
            r2 = 1 - (np.sum((real_stock_price - predicted_stock_price) ** 2) / np.sum((real_stock_price - np.mean(real_stock_price)) ** 2))
            
            st.subheader("📈 Model Performance Metrics")
            perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
            with perf_col1:
                st.metric("MSE", f"{mse:.4f}")
            with perf_col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with perf_col3:
                st.metric("MAE", f"{mae:.4f}")
            with perf_col4:
                st.metric("MAPE", f"{mape:.2f}%")
            with perf_col5:
                st.metric("R² Score", f"{r2:.4f}")
            
            model.save(model_save_name)
            st.success(f"💾 Model saved as '{model_save_name}'")
            
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
    
    with col2:
        st.subheader("Quick Actions")
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.model_history = None
            st.session_state.trained_model = None
            st.session_state.scaler = None
            st.session_state.predictions = None
            st.session_state.actual = None
            st.rerun()
        
        if st.session_state.trained_model is not None:
            st.success("✅ Model Trained")
            if st.button("💾 Save Model", use_container_width=True):
                st.session_state.trained_model.save(model_save_name)
                st.success(f"Model saved as {model_save_name}")

with tab2:
    st.subheader("📊 Data Exploration")
    
    if st.button("Load and Explore Data"):
        with st.spinner("Loading data..."):
            dataset_train, dataset_test, _ = load_data(train_path, test_path)
        
        st.subheader("Dataset Statistics")
        st.dataframe(dataset_train.describe(), use_container_width=True)
        
        st.subheader("Data Types and Info")
        info_buffer = BytesIO()
        dataset_train.info(buf=info_buffer)
        st.text(info_buffer.getvalue().decode())
        
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
            'Column': dataset_train.columns,
            'Missing Count': dataset_train.isnull().sum().values,
            'Missing %': (dataset_train.isnull().sum().values / len(dataset_train) * 100)
        })
        st.dataframe(missing_df, use_container_width=True)
        
        st.subheader("Correlation Matrix")
        numeric_cols = dataset_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            corr_matrix = dataset_train[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.subheader("Time Series Visualization")
        if len(dataset_train.columns) > 1:
            selected_cols = st.multiselect("Select columns to plot", dataset_train.columns.tolist(), 
                                          default=dataset_train.columns[:5].tolist() if len(dataset_train.columns) >= 5 else dataset_train.columns.tolist())
            if selected_cols:
                fig, ax = plt.subplots(figsize=(14, 6))
                for col in selected_cols:
                    if dataset_train[col].dtype in [np.float64, np.int64]:
                        ax.plot(dataset_train[col].values, label=col, alpha=0.7)
                ax.set_title('Time Series Data', fontsize=16, fontweight='bold')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

with tab3:
    st.subheader("📈 Advanced Visualizations")
    
    if st.session_state.model_history is not None and st.session_state.trained_model is not None:
        history = st.session_state.model_history
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.subheader("Training History")
            if 'val_loss' in history:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                ax1.plot(history['loss'], label='Training Loss', linewidth=2)
                ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
                ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(history['mae'], label='Training MAE', linewidth=2)
                ax2.plot(history['val_mae'], label='Validation MAE', linewidth=2)
                ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MAE')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                ax1.plot(history['loss'], label='Training Loss', linewidth=2, color='#1f77b4')
                ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(history['mae'], label='Training MAE', linewidth=2, color='#2ca02c')
                ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MAE')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
        
        with viz_col2:
            st.subheader("Error Distribution")
            try:
                if st.session_state.predictions is not None and st.session_state.actual is not None:
                    errors = (st.session_state.actual - st.session_state.predictions).flatten()
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    ax1.hist(errors, bins=20, color='#ff7f0e', alpha=0.7, edgecolor='black')
                    ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Error')
                    ax1.set_ylabel('Frequency')
                    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.scatter(st.session_state.actual.flatten(), st.session_state.predictions.flatten(), alpha=0.6, color='#9467bd')
                    min_val = min(st.session_state.actual.min(), st.session_state.predictions.min())
                    max_val = max(st.session_state.actual.max(), st.session_state.predictions.max())
                    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                    ax2.set_title('Actual vs Predicted Scatter', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Actual Price')
                    ax2.set_ylabel('Predicted Price')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")
    else:
        st.info("Please train a model first to see visualizations.")

with tab4:
    st.subheader("📋 Model Analysis")
    
    if st.session_state.trained_model is not None:
        st.subheader("Model Summary")
        model_summary = []
        st.session_state.trained_model.summary(print_fn=lambda x: model_summary.append(x))
        st.text("\n".join(model_summary))
        
        total_params = st.session_state.trained_model.count_params()
        trainable_params = sum([np.prod(v.get_shape()) for v in st.session_state.trained_model.trainable_variables])
        non_trainable_params = total_params - trainable_params
        
        st.subheader("Model Parameters")
        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            st.metric("Total Parameters", f"{total_params:,}")
        with param_col2:
            st.metric("Trainable Parameters", f"{trainable_params:,}")
        with param_col3:
            st.metric("Non-Trainable Parameters", f"{non_trainable_params:,}")
        
        if st.session_state.model_history is not None:
            history = st.session_state.model_history
            final_loss = history['loss'][-1]
            final_mae = history['mae'][-1]
            
            st.subheader("Final Training Metrics")
            final_col1, final_col2 = st.columns(2)
            with final_col1:
                st.metric("Final Loss", f"{final_loss:.6f}")
            with final_col2:
                st.metric("Final MAE", f"{final_mae:.6f}")
            
            if 'val_loss' in history:
                final_val_loss = history['val_loss'][-1]
                final_val_mae = history['val_mae'][-1]
                val_col1, val_col2 = st.columns(2)
                with val_col1:
                    st.metric("Final Val Loss", f"{final_val_loss:.6f}")
                with val_col2:
                    st.metric("Final Val MAE", f"{final_val_mae:.6f}")
    else:
        st.info("Please train a model first to see analysis.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Instructions")
st.sidebar.markdown("""
1. Configure data paths and feature selection
2. Adjust model architecture parameters
3. Set training hyperparameters
4. Choose preprocessing options
5. Click 'Train Model and Predict' to start
6. Explore results in different tabs
""")
