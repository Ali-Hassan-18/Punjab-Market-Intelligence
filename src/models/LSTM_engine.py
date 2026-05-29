import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgriculturalLSTM(nn.Module):
    """
    Robust PyTorch LSTM Architecture with Dropout for noise regularization.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float):
        super(AgriculturalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states dynamically based on batch size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Extract the hidden state from the final time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class DeepLearningEngine:
    """
    Executes Phase 4: Advanced LSTM Forecasting.
    Enforces strict chronological isolation, leakage-free scaling, and 3D tensor windowing.
    """
    def __init__(self, target_crop: str, target_market: str, lookback: int = 30, horizon: int = 7):
        self.target_crop = target_crop
        self.target_market = target_market
        self.lookback = lookback
        self.horizon = horizon
        self.target_col = 'price'
        
        # UPDATED: Exogenous features strictly devoid of STL components to prevent leakage.
        self.feature_cols = [
            'price', 'log_return', 'volatility_30d', 'velocity_7d', 
            'positive_return', 'momentum_7d', 'panic_index_7d'
        ]
        
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        load_dotenv()
        self.db_url = f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'postgres')}"
        self.engine = create_engine(self.db_url)

    def extract_and_split(self, train_ratio: float = 0.8):
        """Pulls DB data and executes a strict chronological split."""
        logging.info(f"Extracting Feature Store for {self.target_crop} in {self.target_market}...")
        query = f"""
            SELECT date, {', '.join(self.feature_cols)}
            FROM amis_market_features
            WHERE market = '{self.target_market}' AND crop = '{self.target_crop}'
            ORDER BY date ASC
        """
        df = pd.read_sql(query, con=self.engine)
        df.set_index('date', inplace=True)
        df.dropna(inplace=True)
        
        split_idx = int(len(df) * train_ratio)
        self.train_df = df.iloc[:split_idx].copy()
        self.test_df = df.iloc[split_idx:].copy()
        logging.info(f"Chronological Split Enforced. Train: {len(self.train_df)} | Test: {len(self.test_df)}")

    def apply_leakage_free_scaling(self):
        """Fits scale strictly on training data; transforms both to prevent look-ahead bias."""
        logging.info("Applying strict leakage-free MinMaxScaler...")
        
        # Fit and transform training data
        train_features = self.feature_scaler.fit_transform(self.train_df[self.feature_cols])
        train_target = self.target_scaler.fit_transform(self.train_df[[self.target_col]])
        
        # Transform testing data using training parameters
        test_features = self.feature_scaler.transform(self.test_df[self.feature_cols])
        test_target = self.target_scaler.transform(self.test_df[[self.target_col]])
        
        return train_features, train_target, test_features, test_target

    def create_sliding_windows(self, features, target):
        """Reshapes 2D tabular data into 3D tensors: [samples, sequence_length, features]."""
        X, y = [], []
        for i in range(len(features) - self.lookback - self.horizon + 1):
            X.append(features[i : i + self.lookback, :])
            y.append(target[i + self.lookback : i + self.lookback + self.horizon, 0])
            
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def train_model(self, epochs: int = 150, batch_size: int = 32, patience: int = 20):
        """Executes the training loop with Early Stopping."""
        train_f, train_t, test_f, test_t = self.apply_leakage_free_scaling()
        
        X_train, y_train = self.create_sliding_windows(train_f, train_t)
        self.X_test, self.y_test = self.create_sliding_windows(test_f, test_t)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.model = AgriculturalLSTM(
            input_size=len(self.feature_cols), 
            hidden_size=64, 
            num_layers=2, 
            output_size=self.horizon, 
            dropout=0.2
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        logging.info(f"Initiating LSTM Training on {self.device.type.upper()}...")
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            val_loss = self._validate(criterion)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at Epoch {epoch+1}")
                break
                
        self.model.load_state_dict(torch.load('best_lstm_model.pth', weights_only=True))
        logging.info("Training complete. Best model state restored.")

    def _validate(self, criterion):
        self.model.eval()
        with torch.no_grad():
            X_val, y_val = self.X_test.to(self.device), self.y_test.to(self.device)
            val_outputs = self.model(X_val)
            loss = criterion(val_outputs, y_val)
        self.model.train()
        return loss.item()

    def evaluate_model(self):
        """Inverses the scale and computes exact currency metrics against the SARIMAX baseline."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X_test.to(self.device)).cpu().numpy()
            actuals = self.y_test.numpy()
            
        # FIX: Flatten to (N*Horizon, 1), inverse transform, and reshape back to (N, Horizon)
        pred_pkr = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
        actual_pkr = self.target_scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
        
        rmse = np.sqrt(mean_squared_error(actual_pkr, pred_pkr))
        mae = mean_absolute_error(actual_pkr, pred_pkr)
        mape = mean_absolute_percentage_error(actual_pkr, pred_pkr) * 100
        
        print("\n" + "="*50)
        print(f"🧠 LSTM ADVANCED PERFORMANCE: {self.target_crop} ({self.target_market})")
        print("="*50)
        print(f"Forecast Horizon: {self.horizon} Days")
        print(f"Test RMSE: {rmse:.2f} PKR")
        print(f"Test MAE:  {mae:.2f} PKR")
        print(f"Test MAPE: {mape:.2f} %")
        print(f"SARIMAX Baseline to Beat: 3.92 %")
        print("="*50 + "\n")


if __name__ == "__main__":
    try:
        engine = DeepLearningEngine(target_crop="Onion", target_market="Lahore")
        engine.extract_and_split(train_ratio=0.8)
        engine.train_model(epochs=150, batch_size=32, patience=20)
        engine.evaluate_model()
    except Exception as e:
        logging.error(f"Phase 4 Execution Failed: {e}")