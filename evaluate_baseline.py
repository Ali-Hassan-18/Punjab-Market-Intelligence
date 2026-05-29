# evaluate_baseline.py
import pandas as pd
import logging
from src.database.loader import PostgresLoader
from src.models.sarimax_baseline import SARIMAXBaseline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_evaluation(target_market="Lahore", target_crop="Onion"):
    db_loader = PostgresLoader()
    
    logging.info(f"Extracting Feature Store for {target_crop} in {target_market}...")
    query = f"""
        SELECT date, price, volatility_30d, panic_index_7d 
        FROM amis_market_features 
        WHERE market='{target_market}' AND crop='{target_crop}'
        ORDER BY date
    """
    
    df = pd.read_sql(query, db_loader.engine)
    
    if df.empty:
        logging.error("No data found. Ensure Phase 2 completed successfully.")
        return

    # Use the last 3 years to ensure relevance
    df = df.tail(1095).reset_index(drop=True)

    # STRICT LEAKAGE PREVENTION: 
    # Tune on the first 70%, Cross-Validate on the remaining 30%.
    split_point = int(len(df) * 0.7)
    tuning_df = df.iloc[:split_point]
    cv_df = df.iloc[split_point:]

    evaluator = SARIMAXBaseline(forecast_horizon=7, seasonal_period=7)
    
    try:
        # Step 1: Tune
        evaluator.tune_hyperparameters(tuning_df)
        
        # Step 2: Evaluate
        avg_metrics = evaluator.evaluate_cv(cv_df, n_splits=5)
        
        print("\n" + "="*50)
        print(f"📊 SARIMAX BASELINE PERFORMANCE: {target_crop} ({target_market})")
        print("="*50)
        print(f"Forecast Horizon: 7 Days")
        print(f"Average RMSE: {avg_metrics['RMSE']:.2f} PKR")
        print(f"Average MAE:  {avg_metrics['MAE']:.2f} PKR")
        print(f"Average MAPE: {avg_metrics['MAPE']:.2f} %")
        print("="*50 + "\n")
        
    except Exception as e:
        logging.error(f"Evaluation Failed: {e}")

if __name__ == "__main__":
    run_evaluation(target_market="Lahore", target_crop="Onion")