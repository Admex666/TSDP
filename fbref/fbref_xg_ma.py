import pandas as pd
import numpy as np
from fbref import fbref_module as fbref

URL = 'https://fbref.com/en/squads/7213da33/2024-2025/matchlogs/c11/schedule/Lazio-Scores-and-Fixtures-Serie-A'
table_id = 'matchlogs_for'
df_matchlogs = fbref.scrape(URL, table_id)
df_matchlogs = df_matchlogs[df_matchlogs.Time !='Time'].reset_index(drop=True)
teamname = URL.split('/')[-1].split('-Scores')[0]
league = URL.split('/')[-1].split('-Fixtures-')[-1].replace('-', ' ')

#%% Transform data
df.dropna(inplace=True)
df = df_matchlogs[['Round', 'Opponent', 'GF', 'GA', 'xG', 'xGA']]
df[['GF', 'GA', 'xG', 'xGA']] = df[['GF', 'GA', 'xG', 'xGA']].astype(float)
df['Round_nr'] = df['Round'].str.replace('Matchweek ', '').astype(int)

# Calculate moving averages
window = 4

df['xG_ma'] = df['xG'].rolling(window=window).mean()
df['xGA_ma'] = df['xGA'].rolling(window=window).mean()
df['xG_diff_ma'] = df['xG_ma'] - df['xGA_ma']

#%% Line chart
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
# Colors and styling
mycolor = '#5ECB43'
background_color = '#3c3d3d'
my_font_path = os.getcwd()+ r'\Athletic\Nexa-ExtraLight.ttf'
my_font_props = font_manager.FontProperties(fname=my_font_path)

plt.figure(figsize=(10, 6), facecolor=background_color)
ax = plt.gca()
ax.set_facecolor(background_color)
# Draw lines
plt.plot(df['Round_nr'], df['xG_ma'], label='xG For (Moving Avg)', color='green')
plt.plot(df['Round_nr'], df['xGA_ma'], label='xG Against (Moving Avg)', color='red')
# Fill area between
plt.fill_between(df['Round_nr'], df['xG_ma'], df['xGA_ma'],
                 where=(df['xG_ma'] > df['xGA_ma']),
                 interpolate=True, color='green', alpha=0.3)

plt.fill_between(df['Round_nr'], df['xG_ma'], df['xGA_ma'],
                 where=(df['xG_ma'] < df['xGA_ma']),
                 interpolate=True, color='red', alpha=0.3)

# Find max and min of xG difference
max_idx = df['xG_diff_ma'].idxmax()
min_idx = df['xG_diff_ma'].idxmin()

# MAX arrow
x_max = df.loc[max_idx, 'Round_nr']
y1_max = df.loc[max_idx, 'xG_ma']
y2_max = df.loc[max_idx, 'xGA_ma']
diff_max = y1_max - y2_max
y_top = max(y1_max, y2_max)
y_bottom = min(y1_max, y2_max)

plt.annotate("",
             xy=(x_max, y_bottom), xytext=(x_max, y_top),
             arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))

plt.text(x_max + 0.3, (y_top + y_bottom)/2,
         f"{diff_max:.2f}",
         color='white', fontsize=10, va='center')

# MIN arrow
x_min = df.loc[min_idx, 'Round_nr']
y1_min = df.loc[min_idx, 'xG_ma']
y2_min = df.loc[min_idx, 'xGA_ma']
diff_min = y1_min - y2_min
y_top = max(y1_min, y2_min)
y_bottom = min(y1_min, y2_min)

plt.annotate("",
             xy=(x_min, y_bottom), xytext=(x_min, y_top),
             arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))

plt.text(x_min + 0.3, (y_top + y_bottom)/2,
         f"{diff_min:.2f}",
         color='white', fontsize=10, va='center')

plt.xlabel('Match number', color='white')
plt.ylabel('Expected Goals', color='white')
plt.title(f'{teamname} xG moving avg. in {league}', fontsize=17, color='white')
plt.legend()
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
plt.grid(True)
plt.tick_params(colors='white')
xmax = 35
plt.xticks(ticks=range(window, xmax, int(window/2)))
ymax = 2.5
plt.yticks(ticks=np.arange(0.4, ymax, 0.2))
y_max_value = np.array([*[df['xG_ma'].dropna()], *[df['xGA_ma'].dropna()]]).max()
plt.text(xmax-4, y_max_value*1.07,
         'ADAM JAKUS', color=mycolor,
         fontsize = 15, fontproperties=my_font_props,
         ha='center', va='center')
plt.tight_layout()
#plt.show()

# Save plot
from datetime import time
today_str = datetime.today().strftime('%Y.%m.%d.')
file_name = f'{today_str}, {teamname}, xg moving, {league}'
pic_path = r'C:\Users\Adam\Dropbox\TSDP_output\fbref\2025.04\{}.png'.format(file_name)
plt.savefig(pic_path, bbox_inches="tight", dpi=300)