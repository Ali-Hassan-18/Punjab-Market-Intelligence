# build_features.py
import pandas as pd
import logging
from src.database.loader import PostgresLoader
from src.features.regime import TemporalFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_feature_store():
    db_loader = PostgresLoader()
    db_loader.verify_connection()
    
    logging.info("Extracting cleaned baseline data from PostgreSQL...")
    query = "SELECT * FROM amis_clean_prices ORDER BY market, crop, date"
    
    try:
        # Load the clean data we generated in Phase 1
        df_clean = pd.read_sql(query, db_loader.engine)
        
        if df_clean.empty:
            logging.error("amis_clean_prices table is empty. Run backfill.py first.")
            return
            
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # Initialize and run Phase 2
        engineer = TemporalFeatureEngineer(vol_window=30, panic_window=7)
        features_df = engineer.process(df_clean)
        
        # We use if_exists='replace' here because if we run this script again, 
        # we want to recalculate the entire history cleanly.
        logging.info("Loading enriched dataset into 'amis_market_features' table...")
        features_df.to_sql('amis_market_features', db_loader.engine, if_exists='replace', index=False)
        logging.info("✅ FEATURE STORE BUILD SUCCESSFUL.")
        
    except Exception as e:
        logging.error(f"Failed to build feature store: {e}")

if __name__ == "__main__":
    build_feature_store()