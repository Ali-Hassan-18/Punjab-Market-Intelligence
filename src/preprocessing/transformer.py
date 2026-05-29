import pandas as pd
import numpy as np
import logging

# =============================================================================
# TRANSFORM: Ingestion & Filtering
# =============================================================================
class DataIngestion:
    """
    Handles the standardisation of scraped data and filters for target crops.
    """
    def __init__(self):
        self.target_crops = ['Onion', 'Tomato', 'Potato Fresh', 'Green Chilli']
        
    def filter_and_format(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting schema and filtering target commodities...")
        if df.empty:
            raise ValueError("Ingestion received an empty DataFrame from the Extractor.")

        df['date'] = pd.to_datetime(df['date'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        df = df[df['crop'].isin(self.target_crops)].copy()
        df = df.sort_values(by=['market', 'crop', 'date']).reset_index(drop=True)
        
        return df

# =============================================================================
# TRANSFORM: Anomaly Purging & Interpolation
# =============================================================================
class TimeAwarePreprocessor:
    """
    Executes mathematically sound anomaly detection and temporal gap-filling.
    Structurally enforced against Pandas versioning flaws and SciPy matrix singularities.
    """
    def __init__(self, window_size: int = 14, k_threshold: float = 3.0):
        self.window = window_size
        self.k = k_threshold
        self.scale_factor = 1.4826 

    def _hampel_filter_series(self, series: pd.Series) -> pd.Series:
        """Applies Rolling MAD to a single series mathematically."""
        rolling_median = series.rolling(window=self.window, center=True, min_periods=1).median()
        deviation = np.abs(series - rolling_median)
        rolling_mad = deviation.rolling(window=self.window, center=True, min_periods=1).median() * self.scale_factor
        
        outlier_mask = deviation > (self.k * rolling_mad)
        
        purged_count = outlier_mask.sum()
        if purged_count > 0:
            logging.info(f"Purged {purged_count} local outliers in current temporal window.")
            
        series_clean = series.copy()
        series_clean.loc[outlier_mask] = np.nan
        return series_clean

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing Dynamic Outlier Purging (Rolling MAD)...")
        # Step 1: Safe localized transformation
        df['price'] = df.groupby(['market', 'crop'])['price'].transform(self._hampel_filter_series)
        
        logging.info("Executing Time-Aware Spline Interpolation (with Flatline Protection)...")
        
        # Step 2: Bulletproof iteration to bypass Pandas .apply() column stripping
        processed_chunks = []
        
        for (market, crop), group in df.groupby(['market', 'crop']):
            
            # Isolate variables and protect against AMIS duplicate date uploads
            g = group[['date', 'price']].copy()
            g = g.groupby('date').mean() 
            
            # Enforce strict daily frequency to expose implicit missing dates
            g = g.asfreq('D')
            
            valid_points = g['price'].notna().sum()
            
            # Calculate variance to detect flatlines (which break cubic splines)
            price_variance = g['price'].var() if valid_points > 1 else 0
            
            if valid_points > 3 and price_variance > 1e-5:
                # High variance and enough points: Use Spline to respect momentum
                g['price'] = g['price'].interpolate(method='spline', order=3)
            elif valid_points > 1:
                # Low variance (flatline) or sparse initial subsets: Fallback to linear
                g['price'] = g['price'].interpolate(method='linear')
                
            # Extrapolate edges only if interpolation leaves boundary NaNs
            g['price'] = g['price'].bfill().ffill()
            
            # Reconstruct the structural integrity explicitly
            g = g.reset_index()
            g['market'] = market
            g['crop'] = crop
            
            processed_chunks.append(g)
            
        # Recombine the universe
        clean_df = pd.concat(processed_chunks, ignore_index=True)
            
        if clean_df['price'].isna().sum() > 0:
            raise ValueError("Pipeline Failure: NaN values remain after temporal interpolation.")
            
        logging.info("Phase 1 Complete. Temporal continuity established.")
        return clean_df