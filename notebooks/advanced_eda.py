# notebooks/advanced_eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from src.database.loader import PostgresLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedDataForensics:
    """
    Executes global-standard time-series diagnostics and exploratory data analysis.
    Provides mathematical validation for down-stream predictive models.
    """
    def __init__(self, market: str = "Lahore", crop: str = "Onion"):
        self.market = market
        self.crop = crop
        self.db_loader = PostgresLoader()
        self.df = pd.DataFrame()

    def extract_clean_store(self):
        """Pulls the strictly isolated feature matrix from the database."""
        logging.info(f"Extracting enriched regime data for {self.crop} in {self.market}...")
        query = f"""
            SELECT date, price, log_return, volatility_30d, velocity_7d, momentum_7d, panic_index_7d
            FROM amis_market_features
            WHERE market='{self.market}' AND crop='{self.crop}'
            ORDER BY date ASC
        """
        self.df = pd.read_sql(query, self.db_loader.engine)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)
        logging.info(f"Data successfully pulled. Diagnostic matrix shape: {self.df.shape}")

    def plot_stationarity_analysis(self):
        """Visually and statistically tests for unit-roots and stationarity."""
        logging.info("Executing Augmented Dickey-Fuller (ADF) diagnostic test...")
        
        # Run statistical ADF test
        raw_result = adfuller(self.df['price'].dropna())
        diff_result = adfuller(self.df['log_return'].dropna())

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
        sns.despine()

        # Top Plot: Absolute Inflationary Drift
        axes[0].plot(self.df.index, self.df['price'], color='darkblue', label=f"Raw Clean Price (ADF p-val: {raw_result[1]:.4f})")
        axes[0].set_title(f"Macro Stationarity Diagnostic: {self.crop} Price Levels ({self.market})", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Price (PKR)")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].legend(loc="upper left")

        # Bottom Plot: First-Differenced Log-Returns
        axes[1].plot(self.df.index, self.df['log_return'], color='crimson', alpha=0.7, label=f"Log-Return Series (ADF p-val: {diff_result[1]:.4f})")
        axes[1].set_title(r"First-Differenced Stationary Transformation ($\ln(P_t / P_{t-1})$)", fontsize=12)
        axes[1].set_ylabel("Log Return")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(loc="upper left")

        plt.tight_layout()
        output_path = f"eda_stationarity_{self.market}_{self.crop}.png"
        plt.savefig(output_path, dpi=300)
        logging.info(f"Saved stationarity analysis to {output_path}")
        plt.close()

    def plot_temporal_dependencies(self):
        """Generates memory boundary layouts using ACF and PACF."""
        logging.info("Constructing Autocorrelation and Partial Autocorrelation topologies...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # Autocorrelation (ACF) - Measures global memory decay
        plot_acf(self.df['log_return'].dropna(), lags=28, ax=axes[0], color='darkblue', vlines_kwargs={"colors": 'darkblue'})
        axes[0].set_title("Autocorrelation (ACF): Memory Decay Boundary", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Lags (Days)")
        axes[0].grid(True, linestyle='--', alpha=0.3)

        # Partial Autocorrelation (PACF) - Measures direct lag dependencies
        plot_pacf(self.df['log_return'].dropna(), lags=28, ax=axes[1], color='crimson', vlines_kwargs={"colors": 'crimson'}, method='ywm')
        axes[1].set_title("Partial Autocorrelation (PACF): Direct Temporal Links", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Lags (Days)")
        axes[1].grid(True, linestyle='--', alpha=0.3)

        plt.suptitle(f"Temporal Dependancy Topologies for {self.crop} in {self.market}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        output_path = f"eda_memory_lags_{self.market}_{self.crop}.png"
        plt.savefig(output_path, dpi=300)
        logging.info(f"Saved dependency architecture to {output_path}")
        plt.close()

    def plot_regime_clustering_space(self):
        """Clusters market psychology dynamically without look-ahead metrics."""
        logging.info("Mapping Volatility-Momentum Space and behavioral clusters...")
        
        plt.figure(figsize=(10, 8))
        sns.despine()

        # Categorize the data points into explicit behavioral regimes
        # Based on statistical standard deviations of our Engineered Feature Store
        vol_thresh = self.df['volatility_30d'].median() + self.df['volatility_30d'].std()
        panic_thresh = 0.05 # 5% upward panic velocity boundary

        conditions = [
            (self.df['panic_index_7d'] > panic_thresh) & (self.df['volatility_30d'] > vol_thresh),
            (self.df['volatility_30d'] > vol_thresh) & (self.df['panic_index_7d'] <= panic_thresh)
        ]
        # Only two choices to match the two explicit conditions above.
        choices = ['Supply Shock (Panic Zone)', 'High Volatility Drift']
        
        self.df['Regime Cluster'] = np.select(conditions, choices, default='Normal Equilibrium Trading')

        sns.scatterplot(
            data=self.df, 
            x='volatility_30d', 
            y='panic_index_7d', 
            hue='Regime Cluster',
            palette={'Supply Shock (Panic Zone)': 'crimson', 'High Volatility Drift': 'gold', 'Normal Equilibrium Trading': 'darkblue'},
            alpha=0.6,
            s=40
        )

        plt.title(f"Behavioral Market Clustering Space: {self.crop} ({self.market})", fontsize=14, fontweight='bold')
        plt.xlabel("30-Day Rolling Volatility (Market Instability Scale)")
        plt.ylabel("7-Day Bounded Panic Index (Momentum Vector)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(frameon=True, facecolor='white', edgecolor='none')

        plt.tight_layout()
        output_path = f"eda_market_regimes_{self.market}_{self.crop}.png"
        plt.savefig(output_path, dpi=300)
        logging.info(f"Saved regime spaces to {output_path}")
        plt.close()

    def execute_complete_forensics(self):
        """Orchestrates the entire advanced visual diagnostic layer."""
        self.extract_clean_store()
        self.plot_stationarity_analysis()
        self.plot_temporal_dependencies()
        self.plot_regime_clustering_space()
        logging.info("🏆 ADVANCED DATA FORENSICS COMPLETE. DIAGNOSTIC IMAGES EXPORTED SUCCESSFULLY.")

if __name__ == "__main__":
    # Target Lahore Onions for worst-case system validation
    forensics = AdvancedDataForensics(market="Lahore", crop="Onion")
    forensics.execute_complete_forensics()