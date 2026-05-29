import pandas as pd
import logging
from datetime import datetime, timedelta
import time

from src.ingestion.extractor import AMISExtractor
from src.preprocessing.transformer import DataIngestion, TimeAwarePreprocessor
from src.database.loader import PostgresLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_backfill(days_to_backfill=2555):
    """Orchestrates a historical backfill of the AMIS database."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_backfill)
    
    date_list = pd.date_range(start=start_date, end=end_date)
    
    extractor = AMISExtractor(max_workers=10)
    ingestion = DataIngestion()
    preprocessor = TimeAwarePreprocessor(window_size=14, k_threshold=3.0)
    db_loader = PostgresLoader()
    
    db_loader.verify_connection()
    all_raw_data = []
    
    logging.info(f"INITIATING MASS BACKFILL FROM {start_date.strftime('%m/%d/%Y')} TO {end_date.strftime('%m/%d/%Y')}")
    
    for current_date in date_list:
        date_str = current_date.strftime("%m/%d/%Y")
        try:
            raw_data = extractor.run_daily_scrape(date_str)
            if not raw_data.empty:
                all_raw_data.append(raw_data)
            # Throttle to prevent government server from blocking us
            time.sleep(0.5) 
        except Exception as e:
            logging.error(f"Failed to scrape {date_str}: {e}")

    if all_raw_data:
        master_raw_df = pd.concat(all_raw_data, ignore_index=True)
        logging.info(f"Backfill Extraction Complete. Total Raw Rows: {len(master_raw_df)}")
        
        # 1. Format the raw data
        filtered_data = ingestion.filter_and_format(master_raw_df)
        
        # 2. SAVE THE RAW DATA FOR EDA
        db_loader.load_data(filtered_data, table_name="amis_raw_prices")
        logging.info("Raw data safely loaded to database for EDA purposes.")
        
        # 3. Clean the data
        clean_data = preprocessor.process(filtered_data)
        
        # 4. SAVE THE CLEAN DATA FOR MODELING
        db_loader.load_data(clean_data, table_name="amis_clean_prices")
        
        logging.info("BACKFILL COMPLETION SUCCESSFUL.")
    else:
        logging.error("Backfill failed. No data extracted.")

if __name__ == "__main__":
    # If 7 years takes too long, switch this to 365. 
    run_backfill(days_to_backfill=2555)