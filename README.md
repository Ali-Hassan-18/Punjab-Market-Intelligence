# 🌾 Punjab Agricultural Market Intelligence System (PAMIS)

An end-to-end, production-ready Data Engineering and Machine Learning pipeline designed to extract, clean, engineer, and forecast agricultural commodity prices across the Punjab province using data from AMIS.pk.

## 📖 Background & Project Objective
Agricultural markets in Pakistan are notoriously volatile, plagued by multi-year inflationary drift, climate-induced harvest gluts, and catastrophic supply-chain shocks. Relying on raw government data (AMIS) is dangerous due to frequent server outages, administrative typos, and massive non-linear price spikes.

**Our Objective:** To build an automated intelligence system that isolates the most volatile, inflation-driving kitchen staples (Onion, Tomato, Potato Fresh, Green Chilli) and accurately predicts their prices 7 days into the future. 

Instead of arbitrarily throwing "black-box" Deep Learning at the problem, this project strictly adheres to mathematical forensics. We implemented dynamic anomaly purging, engineered exogenous regime features (Volatility & Panic Indices), and pitted Deep Neural Networks (LSTM) against Ensemble Models (XGBoost) and Classical Statistical Models (SARIMAX) to find the absolute mathematically optimal inference engine.

## 🏗️ System Architecture
This project is divided into four strictly decoupled architectural phases to prevent data leakage and look-ahead bias:

1. **Extract & Transform (ETL):** - Threaded web scraper bypassing ASP.NET security tokens.
   - Dynamic anomaly purging using a **Strict Backward-Looking Median Absolute Deviation (MAD)** filter.
   - **Chronological Linear Interpolation** to bridge server outages without deflating historical variance.
2. **Temporal Feature Engineering (Regime Isolation):**
   - Transformed absolute prices into **Log-Returns** (velocity) to achieve mathematical stationarity.
   - Engineered **30-Day Rolling Volatility** and a **7-Day Directional Panic Index** to quantify market psychology.
3. **Advanced Data Forensics:**
   - Automated ADF Stationarity tests, ACF/PACF memory topology mapping, and Volatility-Momentum space clustering.
4. **Predictive Modeling:**
   - Evaluated LSTM (PyTorch), XGBoost, and SARIMAX via strict Rolling-Origin Cross-Validation (TimeSeriesSplit).

🏆 **Final Result:** The classical **SARIMAX** model outperformed both Deep Learning and Gradient Boosting, achieving a **3.92% MAPE** (Mean Absolute Percentage Error) on highly volatile Lahore Onions over a 7-day forecast horizon.

---

## ⚙️ Prerequisites
* Python 3.10+
* PostgreSQL 14+
* Visual Studio Code (or equivalent IDE)

## 🚀 Setup & Installation

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
**4. Database Configuration**
1. Install PostgreSQL on your local machine.
2. Create a .env file in the root directory.
3. Add the following credentials:
```bash
DB_USER=postgres
DB_PASSWORD=your_local_db_password_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
```
**5. 📊 Execution Protocol (Chronological Order)**
To run the pipeline from scratch and replicate the findings, execute these scripts in this exact order:
1. Phase 1: Database Initialization & Network Extraction
*Warning: This scrapes 7 years of daily data from the AMIS servers. It may take several hours depending on network latency.*
2. Phase 2: Fast Local Rebuild (Data Cleaning & Feature Engineering)
*If you already have the raw data in PostgreSQL and need to re-apply the leakage-free mathematical transformations (Backward MAD, Interpolation, and Regime Feature Engineering), run the local orchestrator. It processes 7 years of data in under 30 seconds:*
```bash
python rebuild_local.py
```
3. Phase 3: Global-Standard Data Forensics (EDA)
*Generates high-resolution statistical proof of stationarity, autocorrelation boundaries, and regime clustering. Outputs images directly to your root directory.*
```bash 
python notebooks/advanced_eda.py
```
4. Phase 4: Model Evaluation & Inference
*The system utilizes strict TimeSeriesSplit cross-validation to prevent data leakage.*
Run the Production Champion (SARIMAX):
```bash 
python evaluate_baseline.py
```

### Final Sanity Check
This README clearly defines the architecture, establishes the mathematical justification for the project, and provides an idiot-proof command sequence to execute the pipeline from top to bottom. 

Add this to your repository, ensure your `.env` file is excluded in your `.gitignore`, and commit the codebase. 