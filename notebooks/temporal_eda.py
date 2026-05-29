import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import numpy as np
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore') # Suppress seaborn warnings for clean output

load_dotenv()
engine = create_engine(f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")

def generate_advanced_eda(target_market="Lahore"):
    print(f"📊 Querying Database for Advanced EDA in {target_market}...")
    
    # 1. Fetch Data
    df_raw = pd.read_sql(f"SELECT date, crop, price as raw_price FROM amis_raw_prices WHERE market='{target_market}'", engine)
    df_clean = pd.read_sql(f"SELECT date, crop, price as clean_price FROM amis_clean_prices WHERE market='{target_market}'", engine)
    
    if df_raw.empty or df_clean.empty:
        print("❌ ERROR: Database returned empty. Ensure the backfill script completed successfully.")
        return

    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    core_crops = ['Onion', 'Tomato', 'Potato Fresh', 'Green Chilli']
    
    # Initialize the Figure
    fig = plt.figure(figsize=(20, 18))
    sns.set_theme(style="whitegrid")
    fig.suptitle(f"Comprehensive Data Forensics & Quality Assurance Report ({target_market})", fontsize=24, fontweight='bold', y=0.98)
    
    # ==============================================================================
    # PANEL 1: Volatility Proof (Why these 4 crops?)
    # ==============================================================================
    ax1 = plt.subplot(4, 1, 1)
    
    # Calculate Coefficient of Variation (StdDev / Mean) to compare volatility fairly across different price scales
    stats = df_raw.groupby('crop')['raw_price'].agg(['std', 'mean'])
    stats['cov'] = stats['std'] / stats['mean']
    stats = stats.sort_values('cov', ascending=False).head(20) # Top 20 most volatile
    
    # Color mapping: Red for our targets, grey for the rest
    colors = ['crimson' if crop in core_crops else 'lightgray' for crop in stats.index]
    
    sns.barplot(x=stats.index, y=stats['cov'], ax=ax1, palette=colors)
    ax1.set_title("Selection Rationale: Coefficient of Variation (Volatility normalized by Price Scale)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Volatility Score", fontsize=12)
    ax1.set_xlabel("")
    ax1.tick_params(axis='x', rotation=45)
    
    ax1.text(0.98, 0.85, "Highlight: Chosen Target Crops\nMetric: Coefficient of Variation proves these crops suffer the most extreme relative price shocks.", 
             transform=ax1.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # ==============================================================================
    # PREPARE DATA FOR PANELS 2 & 3 (Missing Data Matrix)
    # ==============================================================================
    # We create a perfect continuous daily calendar to expose the missing days
    min_date = df_raw['date'].min()
    max_date = df_raw['date'].max()
    full_calendar = pd.date_range(start=min_date, end=max_date, freq='D')
    
    raw_pivot = df_raw[df_raw['crop'].isin(core_crops)].pivot(index='date', columns='crop', values='raw_price')
    raw_pivot = raw_pivot.reindex(full_calendar)
    
    clean_pivot = df_clean[df_clean['crop'].isin(core_crops)].pivot(index='date', columns='crop', values='clean_price')
    clean_pivot = clean_pivot.reindex(full_calendar)

    # ==============================================================================
    # PANEL 2: Missing Data Matrix (Before)
    # ==============================================================================
    ax2 = plt.subplot(4, 1, 2)
    # Visualizing NaNs as black lines
    sns.heatmap(raw_pivot.isna().T, cmap=['lightgreen', 'black'], cbar=False, ax=ax2, xticklabels=False)
    ax2.set_title("Missing Data Topology (Before: Raw AMIS Extraction)", fontsize=16, fontweight='bold')
    ax2.set_ylabel("Core Crops", fontsize=12)
    ax2.text(0.01, 0.90, "Black = Missing Data Points", transform=ax2.transAxes, color='white', fontweight='bold')

    # ==============================================================================
    # PANEL 3: Missing Data Matrix (After)
    # ==============================================================================
    ax3 = plt.subplot(4, 1, 3)
    sns.heatmap(clean_pivot.isna().T, cmap=['lightgreen', 'black'], cbar=False, ax=ax3, xticklabels=False)
    ax3.set_title("Continuity Restored (After: Time-Aware Spline Interpolation)", fontsize=16, fontweight='bold')
    ax3.set_ylabel("Core Crops", fontsize=12)
    
    # ==============================================================================
    # PANEL 4: Anomaly Purging Visualization
    # ==============================================================================
    ax4 = plt.subplot(4, 1, 4)
    target_crop = "Onion" # We focus the time-series on one highly volatile crop
    
    df_merged = pd.merge(raw_pivot[[target_crop]], clean_pivot[[target_crop]], left_index=True, right_index=True)
    df_merged.columns = ['raw_price', 'clean_price']
    
    ax4.plot(df_merged.index, df_merged['raw_price'], color='red', alpha=0.4, linewidth=1.5, label='Raw Observations')
    ax4.plot(df_merged.index, df_merged['clean_price'], color='blue', alpha=1.0, linewidth=2, label='Clean Trajectory (Rolling MAD)')
    
    anomalies = df_merged[df_merged['raw_price'] != df_merged['clean_price']]
    if not anomalies.empty:
         ax4.scatter(anomalies.index, anomalies['raw_price'], color='darkred', marker='x', s=40, label='Identified & Purged Anomalies')
         
    ax4.set_title(f"Rolling MAD Anomaly Purging: {target_crop} Trajectory Analysis", fontsize=16, fontweight='bold')
    ax4.set_ylabel("Price (PKR)", fontsize=12)
    ax4.set_xlabel("Date", fontsize=12)
    ax4.legend(loc="upper left")

    # ==============================================================================
    plt.tight_layout()
    plt.subplots_adjust(top=0.94) 
    
    output_filename = f"Advanced_EDA_Report_{target_market}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Advanced EDA Dashboard successfully generated: {output_filename}")
    plt.show()

if __name__ == "__main__":
    generate_advanced_eda(target_market="Lahore")