import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# -----------------------------
# INPUT (az általad adott tábla)
# -----------------------------
data = [
    ("HUN", "2023-2024", 12, 7),
    ("HUN", "2024-2025", 12, 7),
    ("AUT", "2022-2023", 12, 8),
    ("ITA", "2024-2025", 20, 8),
    ("GER", "2022-2023", 18, 10),
    ("ESP", "2024-2025", 20, 11),
    ("ENG", "2024-2025", 20, 11),
    ("ENG", "2022-2023", 20, 11),
    ("HUN", "2022-2023", 12, 11),
    ("ENG", "2023-2024", 20, 11),
    ("ITA", "2023-2024", 20, 13),
    ("AUT", "2024-2025", 12, 13),
    ("GER", "2024-2025", 18, 14),
    ("GER", "2023-2024", 18, 14),
    ("AUT", "2023-2024", 12, 18),
    ("ESP", "2022-2023", 20, 20),
    ("ITA", "2022-2023", 20, 24),
    ("ESP", "2023-2024", 20, 29),
]

df = pd.DataFrame(data, columns=["Country", "Season", "Nodes", "FirstCycleWeek"])


# -----------------------------
# STATISZTIKA
# -----------------------------
x = df["Nodes"].values
y = df["FirstCycleWeek"].values

corr, p_value = pearsonr(x, y)

# lineáris regresszió (y = ax + b)
a, b = np.polyfit(x, y, 1)

print("\n📊 KORRELÁCIÓS ELEMZÉS")
print("------------------------")
print(f"Pearson r: {corr:.4f}")
print(f"p-value:   {p_value:.6f}")
print(f"Egyenlet:  y = {a:.4f}x + {b:.4f}")


# -----------------------------
# PLOT
# -----------------------------
plt.figure()

plt.scatter(x, y)

# regressziós egyenes
x_line = np.linspace(min(x), max(x), 100)
y_line = a * x_line + b

plt.plot(x_line, y_line)

plt.xlabel("Csapatok száma (Nodes)")
plt.ylabel("Első Hamilton-kör megjelenése (forduló)")
plt.title("Circle of Parity: méret vs kialakulás ideje")

plt.grid(True)

plt.show()