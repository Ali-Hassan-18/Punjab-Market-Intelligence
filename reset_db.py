# reset_db.py
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()

db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "postgres")

engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

def wipe_database():
    logging.warning("⚠️ INITIATING DATABASE WIPE ⚠️")
    with engine.connect() as connection:
        # Using CASCADE to handle any potential foreign key locks
        connection.execute(text("DROP TABLE IF EXISTS amis_raw_prices CASCADE;"))
        connection.execute(text("DROP TABLE IF EXISTS amis_clean_prices CASCADE;"))
        connection.commit()  # Required for SQLAlchemy 2.0+
    logging.info("✅ Tables dropped successfully. You may now run backfill.py.")

if __name__ == "__main__":
    wipe_database()