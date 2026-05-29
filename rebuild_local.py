# rebuild_local.py
import pandas as pd
import logging
from src.database.loader import PostgresLoader
from src.preprocessing.transformer import TimeAwarePreprocessor
from src.features.regime import TemporalFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fast_local_rebuild():
    logging.info("INITIATING FAST LOCAL PIPELINE REBUILD (Bypassing Network Extraction)...")
    db_loader = PostgresLoader()
    db_loader.verify_connection()
    
    # 1. Pull the 7-year RAW data directly from local PostgreSQL
    logging.info("Extracting 7-year raw dataset from 'amis_raw_prices'...")
    raw_df = pd.read_sql("SELECT * FROM amis_raw_prices ORDER BY market, crop, date", db_loader.engine)
    
    if raw_df.empty:
        logging.error("CRITICAL: 'amis_raw_prices' is empty. You must run backfill.py.")
        return
        
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    
    # 2. Re-run Phase 1: STRICT LEAKAGE-FREE Preprocessing
    logging.info("Executing Phase 1 (Strict Backward-Looking MAD & Linear Interpolation)...")
    preprocessor = TimeAwarePreprocessor(window_size=14, k_threshold=3.0)
    clean_df = preprocessor.process(raw_df)
    
    logging.info("Overwriting 'amis_clean_prices' in database...")
    clean_df.to_sql('amis_clean_prices', db_loader.engine, if_exists='replace', index=False)
    
    # 3. Re-run Phase 2: STRICT LEAKAGE-FREE Feature Engineering
    logging.info("Executing Phase 2 (Strict Historical Regime Extraction, No STL)...")
    engineer = TemporalFeatureEngineer(vol_window=30, panic_window=7)
    features_df = engineer.process(clean_df)
    
    logging.info("Overwriting 'amis_market_features' in database...")
    features_df.to_sql('amis_market_features', db_loader.engine, if_exists='replace', index=False)
    
    logging.info("✅ LOCAL REBUILD SUCCESSFUL. Compute time saved: ~4 hours.")

if __name__ == "__main__":
    fast_local_rebuild()