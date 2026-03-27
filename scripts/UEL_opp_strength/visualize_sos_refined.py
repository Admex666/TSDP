import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    sos_file = os.path.join(directory, "uel_2025_26_sos_refined_analysis.csv")
    
    if not os.path.exists(sos_file):
        print("Refined SOS Analysis file not found.")
        return
        
    df = pd.read_csv(sos_file)
    
    # Use adjusted SOS
    x = df['avg_opp_points_adj'].values
    y = df['points'].values
    
    # Linear Regression (degree 1 polyfit)
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y_mean)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='#2ecc71', alpha=0.7, edgecolors='k', s=120, label='Csapatok')
    
    # Regression line
    line_x = np.array([min(x), max(x)])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='#e74c3c', linewidth=2.5, linestyle='-', label='Regressziós vonal')
    
    # Set Y-axis ticks to increments of 3
    max_y = int(np.ceil(max(y) / 3) * 3)
    plt.yticks(np.arange(0, max_y + 3, 3))
    plt.ylim(-1, max_y + 1)
    
    # Labels
    labeled_indices = set(df.head(3).index) | set(df.tail(3).index)
    labeled_indices.add(df[df['team_name'] == 'Ferencváros TC'].index[0])
    
    for i, row in df.iterrows():
        if i in labeled_indices or i % 4 == 0:
            plt.annotate(row['team_name'], (x[i], y[i]), xytext=(7, 7), 
                         textcoords='offset points', fontsize=9, alpha=0.9)

    plt.title('Finomított Sorsolás Erőssége vs. Megszerzett pontok (UEL 25/26)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Ellenfelek átlagpontszáma (Közvetlen meccs nélkül)', fontsize=13)
    plt.ylabel('Megszerzett pontok', fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.8)
    
    equation = f"y = {slope:.2f}x + {intercept:.2f}"
    r2_text = f"R² = {r_squared:.4f}"
    
    textstr = '\n'.join((
        f"Regressziós egyenlet: {equation}",
        f"Determinációs együttható: {r2_text}",
        "(A pontokból levontuk az aktuális csapat elleni meccs eredményét)"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.legend(frameon=True, shadow=True, borderpad=1)
    plt.tight_layout()
    
    output_image = os.path.join(directory, "sos_refined_vs_points.png")
    plt.savefig(output_image, dpi=300)
    print(f"Refined visualization saved to {output_image}")

if __name__ == "__main__":
    main()
