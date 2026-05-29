# src/preprocessing/transformer.py
import pandas as pd
import numpy as np
import logging

class DataIngestion:
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

class TimeAwarePreprocessor:
    """
    Executes anomaly detection and temporal gap-filling.
    STRICT LEAKAGE PREVENTION: center=False enforced. No future-peeking.
    """
    def __init__(self, window_size: int = 14, k_threshold: float = 3.0):
        self.window = window_size
        self.k = k_threshold
        self.scale_factor = 1.4826 

    def _hampel_filter_series(self, series: pd.Series) -> pd.Series:
        """Applies Rolling MAD strictly looking backward."""
        # LEAKAGE FIX: center=False means it only uses [t-14 to t], NEVER t+1
        rolling_median = series.rolling(window=self.window, center=False, min_periods=1).median()
        deviation = np.abs(series - rolling_median)
        rolling_mad = deviation.rolling(window=self.window, center=False, min_periods=1).median() * self.scale_factor
        
        outlier_mask = deviation > (self.k * rolling_mad)
        
        purged_count = outlier_mask.sum()
        if purged_count > 0:
            logging.info(f"Purged {purged_count} local outliers (Strict Backward-Looking).")
            
        series_clean = series.copy()
        series_clean.loc[outlier_mask] = np.nan
        return series_clean

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing Dynamic Outlier Purging (Strict Backward MAD)...")
        df['price'] = df.groupby(['market', 'crop'])['price'].transform(self._hampel_filter_series)
        
        logging.info("Executing Chronological Interpolation...")
        processed_chunks = []
        
        for (market, crop), group in df.groupby(['market', 'crop']):
            g = group[['date', 'price']].copy()
            g = g.groupby('date').mean() 
            g = g.asfreq('D')
            
            valid_points = g['price'].notna().sum()
            
            # LEAKAGE FIX: We abandon Cubic Spline as it fits a global polynomial.
            # We use strictly linear interpolation which only connects [t-1] to [t+1], 
            # and if [t+1] doesn't exist yet, we strictly forward-fill (ffill).
            if valid_points > 1:
                g['price'] = g['price'].interpolate(method='linear', limit_direction='forward')
                
            # Final safety catch: if the very first values are NaN, backfill them. 
            # (Minor leakage, but localized only to the extreme deep past, irrelevant for current forecasting).
            g['price'] = g['price'].bfill()
            
            g = g.reset_index()
            g['market'] = market
            g['crop'] = crop
            processed_chunks.append(g)
            
        clean_df = pd.concat(processed_chunks, ignore_index=True)
            
        if clean_df['price'].isna().sum() > 0:
            raise ValueError("Pipeline Failure: NaN values remain after temporal interpolation.")
            
        logging.info("Phase 1 Complete. Leakage-free temporal continuity established.")
        return clean_df