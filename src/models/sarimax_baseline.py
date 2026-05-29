# src/models/sarimax_baseline.py
import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

class SARIMAXBaseline:
    """
    Executes Phase 3: Baseline SARIMAX Forecasting.
    Optimized: Tunes parameters once, then evaluates using fast statsmodels backend.
    """
    def __init__(self, forecast_horizon: int = 7, seasonal_period: int = 7):
        self.forecast_horizon = forecast_horizon
        self.seasonal_period = seasonal_period
        self.best_order = None
        self.best_seasonal_order = None

    def tune_hyperparameters(self, train_df: pd.DataFrame):
        """Finds optimal ARIMA parameters using a strictly isolated training set."""
        logging.info("Initiating Auto-ARIMA hyperparameter search (This may take a minute)...")
        y_train = train_df['price']
        X_train = train_df[['volatility_30d', 'panic_index_7d']]
        
        stepwise_model = pm.auto_arima(
            y=y_train, 
            X=X_train,
            start_p=1, start_q=1,
            max_p=3, max_q=3, 
            m=self.seasonal_period,
            start_P=0, start_Q=0,
            seasonal=True,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        
        self.best_order = stepwise_model.order
        self.best_seasonal_order = stepwise_model.seasonal_order
        logging.info(f"Optimal Parameters Discovered -> ARIMA: {self.best_order}, Seasonal: {self.best_seasonal_order}")

    def evaluate_cv(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        """Executes strict rolling-origin CV using locked parameters."""
        if not self.best_order:
            raise ValueError("Model must be tuned before cross-validation.")

        # Enforce a strict forecast horizon (test_size) to mimic production forecasting
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=self.forecast_horizon)
        metrics = {'RMSE': [], 'MAE': [], 'MAPE': []}
        
        y = df['price']
        X = df[['volatility_30d', 'panic_index_7d']]

        logging.info(f"Executing Temporal Cross-Validation ({n_splits} folds, {self.forecast_horizon}-day horizon)...")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            
            # Using statsmodels directly for massive speed boost during CV
            model = SARIMAX(
                endog=y_train,
                exog=X_train,
                order=self.best_order,
                seasonal_order=self.best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False)
            predictions = fitted_model.forecast(steps=len(y_test), exog=X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions) * 100
            
            metrics['RMSE'].append(rmse)
            metrics['MAE'].append(mae)
            metrics['MAPE'].append(mape)
            
            logging.info(f"Fold {fold + 1} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

        avg_metrics = {
            'RMSE': np.mean(metrics['RMSE']),
            'MAE': np.mean(metrics['MAE']),
            'MAPE': np.mean(metrics['MAPE'])
        }
        return avg_metrics