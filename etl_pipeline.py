import logging
from datetime import datetime

# Import all modules
from src.ingestion.extractor import AMISExtractor
from src.preprocessing.transformer import DataIngestion, TimeAwarePreprocessor
from src.database.loader import PostgresLoader  # <--- NEW IMPORT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    try:
        target_date = datetime.now().strftime("%m/%d/%Y")
        
        # 1. EXTRACT
        extractor = AMISExtractor(max_workers=10)
        raw_data = extractor.run_daily_scrape(target_date)
        
        # 2. TRANSFORM
        ingestion = DataIngestion()
        filtered_data = ingestion.filter_and_format(raw_data)
        
        preprocessor = TimeAwarePreprocessor(window_size=14, k_threshold=3.0)
        clean_data = preprocessor.process(filtered_data)
        
        print("\n=== TRANSFORM SUCCESS ===")
        print(clean_data.head())
        
        # 3. LOAD (NEW)
        print("\n=== INITIATING LOAD PHASE ===")
        db_loader = PostgresLoader()
        db_loader.verify_connection()
        db_loader.load_data(clean_data, table_name="amis_clean_prices")
        
        logging.info("ETL PIPELINE EXECUTION COMPLETED SUCCESSFULLY.")
        
    except Exception as e:
        logging.error(f"Pipeline Failed: {e}")