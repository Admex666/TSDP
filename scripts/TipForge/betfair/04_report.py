import os
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "output"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

def create_plots(df):
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. BSP vs Morningwap scatter (sample a subset if too large)
    sample_df = df.sample(n=min(len(df), 50000), random_state=42) if len(df) > 50000 else df
    
    plt.figure(figsize=(8, 6))
    plt.scatter(sample_df['morningwap'], sample_df['bsp'], alpha=0.1, s=2)
    plt.plot([0, sample_df['bsp'].max()], [0, sample_df['bsp'].max()], 'r--') # y=x line
    plt.xlabel("Morning WAP")
    plt.ylabel("Betfair SP")
    plt.title("BSP vs Morning WAP (Log Scale Optional)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "bsp_vs_morningwap.png"), dpi=300)
    plt.close()
    
    # 2. Odds change distribution
    plt.figure(figsize=(8, 6))
    plt.hist(df['odds_change'], bins=100, range=(-10, 10), color='salmon', edgecolor='black')
    plt.xlabel("Odds Change (BSP - Morning WAP)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Odds Change")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "odds_change_dist.png"), dpi=300)
    plt.close()
    
    # 3. CLV distribution
    plt.figure(figsize=(8, 6))
    plt.hist(df['clv'], bins=100, range=(-0.25, 0.25), color='skyblue', edgecolor='black')
    plt.xlabel("Closing Line Value (CLV)")
    plt.ylabel("Frequency")
    plt.title("Distribution of CLV")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "clv_dist.png"), dpi=300)
    plt.close()
    
    # 4. Volume distribution (log1p scale to handle large variance)
    plt.figure(figsize=(8, 6))
    import numpy as np
    plt.hist(np.log1p(df['early_volume']), bins=50, alpha=0.5, label='Early')
    plt.hist(np.log1p(df['preplay_volume']), bins=50, alpha=0.5, label='Pre-off')
    plt.hist(np.log1p(df['inplay_volume']), bins=50, alpha=0.5, label='In-play')
    plt.xlabel("Log(1 + Volume)")
    plt.ylabel("Frequency")
    plt.title("Volume Distributions")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "volume_dist.png"), dpi=300)
    plt.close()

def generate_markdown_report(results_df):
    report_path = os.path.join(OUTPUT_DIR, "final_report.md")
    
    # Sort strategies by Net ROI desc
    results_df = results_df.sort_values(by="Net ROI %", ascending=False)
    
    md = "# Betfair Pricing & Strategy Analysis Report\n\n"
    md += "## Strategy Performance\n\n"
    md += "The following table summarizes the performance of the backtested strategies:\n\n"
    md += results_df.to_markdown(index=False) + "\n\n"
    
    md += "### Top Strategies by Net ROI:\n"
    for i, row in results_df.head(3).iterrows():
        md += f"- **{row['Strategy']}**: {row['Net ROI %']}% Net ROI over {row['Bets']} bets.\n"
        
    md += "\n## Data Visualizations\n\n"
    md += "### Odds Change & BSP vs Morning WAP\n"
    md += "![BSP vs Morning WAP](plots/bsp_vs_morningwap.png)\n"
    md += "![Odds Change Dist](plots/odds_change_dist.png)\n\n"
    
    md += "### Closing Line Value (CLV)\n"
    md += "![CLV Dist](plots/clv_dist.png)\n\n"
    
    md += "### Volume Dynamics\n"
    md += "![Volume Dist](plots/volume_dist.png)\n"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
        
    logging.info(f"Generated markdown report at {report_path}")

def main():
    df_path = os.path.join(PROCESSED_DIR, "master_dataset.csv")
    results_path = os.path.join(OUTPUT_DIR, "backtest_results.csv")
    
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        logging.info("Generating plots...")
        create_plots(df)
    else:
        logging.warning("Master dataset not found. Skipping plotting.")
        
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        logging.info("Generating report...")
        generate_markdown_report(results_df)
    else:
        logging.warning("Backtest results not found. Skipping report.")

if __name__ == "__main__":
    main()
