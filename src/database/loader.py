# src/database/loader.py
import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
from dotenv import load_dotenv

# Load credentials from the .env file we created earlier
load_dotenv()

class PostgresLoader:
    """
    Handles the Load phase of the ETL pipeline. 
    Pushes clean time-series data into a PostgreSQL database.
    """
    def __init__(self):
        # Construct the database URL securely from environment variables
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "")
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "5432")
        self.db_name = os.getenv("DB_NAME", "postgres")
        
        # Format: postgresql://user:password@host:port/database
        self.connection_string = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self.engine = create_engine(self.connection_string)

    def verify_connection(self):
        """Tests if the database is alive and accessible."""
        try:
            with self.engine.connect() as connection:
                # Use sqlalchemy text() for literal SQL execution
                result = connection.execute(text("SELECT 1"))
                if result.scalar() == 1:
                    logging.info("Successfully connected to PostgreSQL database.")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Database connection failed. Is PostgreSQL running? Error: {e}")

    def load_data(self, df: pd.DataFrame, table_name: str = "amis_clean_prices"):
        """
        Loads the pandas DataFrame into the SQL table.
        Uses 'append' to safely add new daily data without deleting history.
        """
        if df.empty:
            logging.warning("DataFrame is empty. Nothing to load into the database.")
            return

        logging.info(f"Loading {len(df)} records into table '{table_name}'...")
        try:
            # Pushes the dataframe to SQL. 
            # if_exists='append' ensures we just add today's data to the existing table.
            df.to_sql(name=table_name, con=self.engine, if_exists='append', index=False)
            logging.info(f"Successfully loaded data into '{table_name}'.")
        except Exception as e:
            logging.error(f"Failed to load data into database: {e}")
            raise