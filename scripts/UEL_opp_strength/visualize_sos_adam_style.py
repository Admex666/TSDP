import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from adam_style import ADAM_STYLE

try:
    from adjust_text import adjust_text
except ImportError:
    try:
        from adjustText import adjust_text
    except ImportError:
        print("adjust_text not found, falling back to basic labels.")
        def adjust_text(texts, **kwargs):
            return texts

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    # Switching back to the original SOS (Including head-to-head) as requested
    sos_file = os.path.join(directory, "uel_2025_26_sos_analysis.csv")
    
    if not os.path.exists(sos_file):
        print("SOS Analysis file not found.")
        return
        
    df = pd.read_csv(sos_file)
    
    # Using 'avg_opp_points' (simple average of opponent final points)
    x = df['avg_opp_points'].values
    y = df['points'].values
    
    # Linear Regression
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y_mean)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate Signed Residuals
    df['residual'] = y - y_pred
    
    # --- PLOTTING ---
    plt.rcParams.update({
        'text.color': ADAM_STYLE['text_color'],
        'axes.labelcolor': ADAM_STYLE['text_color'],
        'xtick.color': ADAM_STYLE['text_color'],
        'ytick.color': ADAM_STYLE['text_color'],
        'axes.edgecolor': ADAM_STYLE['text_color'],
        'grid.color': '#555555',
        'grid.alpha': 0.3
    })
    
    fig = plt.figure(figsize=(15, 10), facecolor=ADAM_STYLE['bg_color'])
    ax = plt.gca()
    ax.set_facecolor(ADAM_STYLE['bg_color'])
    
    # Scatter plot
    plt.scatter(x, y, color=ADAM_STYLE['acc_color'], alpha=0.9, edgecolors='white', s=160, zorder=3, label='Teams')
    
    # Regression line
    line_x = np.array([min(x), max(x)])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color=ADAM_STYLE['line_color'], linewidth=3.5, linestyle='-', zorder=2, label='Regression')
    
    # Y-axis ticks
    max_y = int(np.ceil(max(y) / 3) * 3)
    plt.yticks(np.arange(0, max_y + 3, 3))
    plt.ylim(-2, max_y + 2)
    
    # Identify Top n Outliers
    n = 6
    topn_pos_ids = set(df.sort_values(by='residual', ascending=False).head(n).index)
    topn_neg_ids = set(df.sort_values(by='residual', ascending=True).head(n).index)
    labeled_ids = topn_pos_ids | topn_neg_ids
    
    # Labels with adjustText
    texts = []
    for i, row in df.iterrows():
        if i in labeled_ids:
            label_color = ADAM_STYLE['acc_color'] if row['residual'] > 0 else '#FF9999'
            t = plt.text(x[i], y[i], f"{row['team_name']} ({'+' if row['residual']>0 else ''}{row['residual']:.1f})", 
                         fontsize=12, color=label_color)
            texts.append(t)
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='lightgray', lw=1.0),
                expand_points=(2.0, 2.0), force_points=(0.5, 0.5))
    
    # Axes Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    # Header
    plt.text(0.5, 1.06, 'Strength of Schedule and Performance', transform=ax.transAxes, 
             fontsize=22, fontweight='bold', ha='center', color='white')
    plt.text(0.5, 1.02, f'UEFA Europa League 2025/26 | League Phase', 
             transform=ax.transAxes, fontsize=12, ha='center', color='#cccccc')
    
    plt.xlabel('Average Opponent Points', fontsize=14, labelpad=15)
    plt.ylabel('Points Earned', fontsize=14, labelpad=15)
    plt.grid(True, zorder=1)
    
    # Stats box
    eq_str = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.4f}"
    props = dict(boxstyle='round', facecolor=ADAM_STYLE['bg_color'], alpha=0.9, edgecolor='white')
    plt.text(0.04, 0.04, eq_str, transform=ax.transAxes, fontsize=14, verticalalignment='bottom', bbox=props, color='white')
    
    # SIGNATURE (with specified font)
    sig_font_props = None
    if ADAM_STYLE.get('font_path') and os.path.exists(ADAM_STYLE['font_path']):
        sig_font_props = fm.FontProperties(fname=ADAM_STYLE['font_path'])
    
    plt.text(1.0, 1.08, ADAM_STYLE['signature'], color=ADAM_STYLE['acc_color'], 
             fontsize=28, fontweight='normal', fontproperties=sig_font_props, 
             transform=ax.transAxes, ha='right', va='top', alpha=0.8)
    
    plt.legend(facecolor=ADAM_STYLE['bg_color'], edgecolor='white', labelcolor='white', loc='upper right', fontsize=12)
    plt.tight_layout()
    
    output_image = os.path.join(directory, "sos_regression_adam_style_minimal.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_image}")
    
    # --- CONSOLE STATS ---
    print("\n" + "="*90)
    print(f"{'Pos.':<5} {'Team Name':<30} {'Pts':<6} {'SOS (Raw)':<12} {'Diff (+/-)':<10}")
    print("-" * 90)
    
    df_sorted = df.sort_values(by='pos')
    for _, row in df_sorted.iterrows():
        prefix = "+" if row['residual'] > 0 else ""
        print(f"{row['pos']:<5} {row['team_name']:<30} {row['points']:<6} {row['avg_opp_points']:<12.2f} {prefix}{row['residual']:<10.2f}")
    
    print("-" * 90)
    print(f"\nTOP {n} OVERPERFORMERS (Better than predicted by raw SOS):")
    top_pos = df.sort_values(by='residual', ascending=False).head(n)
    for i, (_, row) in enumerate(top_pos.iterrows(), 1):
        print(f"{i}. {row['team_name']:<30} +{row['residual']:.2f} pts")
        
    print(f"\nTOP {n} UNDERPERFORMERS (Worse than predicted by raw SOS):")
    top_neg = df.sort_values(by='residual', ascending=True).head(n)
    for i, (_, row) in enumerate(top_neg.iterrows(), 1):
        print(f"{i}. {row['team_name']:<30} {row['residual']:.2f} pts")
    
    print("="*90)

if __name__ == "__main__":
    main()
