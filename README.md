# Punjab Agricultural Market Intelligence System 🌾📈

An end-to-end, production-ready Machine Learning and Data Engineering pipeline designed to extract, clean, and forecast agricultural commodity prices across the Punjab province using data from AMIS.pk.

## 🏗️ System Architecture
This project implements a strict ETL (Extract, Transform, Load) architecture:
1. **Extract:** Threaded web scraper bypassing ASP.NET security tokens.
2. **Transform:** Dynamic anomaly purging using a Rolling Median Absolute Deviation (MAD) filter, followed by Time-Aware Cubic Spline Interpolation to bridge missing dates.
3. **Load:** In-memory batch loading to a local PostgreSQL database.
4. **Analysis:** Automated generation of high-resolution Data Forensic Dashboards.

*Current Target Commodities:* Onion, Tomato, Potato Fresh, Green Chilli.

## ⚙️ Prerequisites
* Python 3.10+
* PostgreSQL 14+

## 🚀 Setup & Installation (For Collaborators)

**1. Clone the repository**
```bash
git clone <your_github_repo_url>
cd punjab_market_intelligence
```

**2. Initialize the virtual environment**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. DataBase Configuration**

1. Install PostgreSQL on your local machine and remember your postgres user password.
2. Copy the .env.example file and rename it to .env.
3. Open .env and replace your_local_db_password_here with your actual password.

**5. 📊 Execution Protocol**
1. The Historical Backfill
To initialize your local database, you must run the backfill script. This will extract historical data, clean it, and load it into your local Postgres instance.

run these on the terminal after activating the virtual environment
```bash
python reset_db.py
python backfill.py
```

2. Generate Exploratory Data Analysis (EDA)
To verify the statistical integrity of the data and view the Before/After anomaly purging dashboard:

```bash
python notebooks/temporal_eda.py
```
3. Daily Cron Job (Pipeline Execution)
To scrape and append today's market data to the database:

```bash
python etl_pipeline.py
```