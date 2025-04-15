import pandas as pd
import fbref.fbref_module as fbref

URL = "https://fbref.com/en/squads/8602292d/2024-2025/matchlogs/all_comps/shooting/Aston-Villa-Match-Logs-All-Competitions"
table_id = 'matchlogs_for'

df = fbref.scrape(URL, table_id)
df = df[(df.Expected_xG != 'Expected') & (df.Expected_xG != 'xG')]
df.dropna(inplace=True)

# Calculate moving averages
df['xG_mavg'] = df['Expected_xG'].rolling(window=window).mean()
df['xGA_ma'] = df['xGA'].rolling(window=window).mean()
df.rename(columns={[col for col in df.columns if '_Date' in col][0]:'Date'}, inplace=True)
df['nr'] = range(len(df))
df['nr'] += 1

df.rename(columns={'Standard_Gls': 'Goals', 'Expected_xG': 'xG'}, inplace=True)

#%% Line chart
def create_mavg_chart(df, x, col1, col2, title, y_title, b4a_line, b4a_text, save_path, save_name, colors=['blue', 'skyblue'], b4a_color='orange',  window=4, save=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    import os
    
    # Prep data
    df[f'{col1}_mavg'] = df[col1].rolling(window=window).mean()
    df[f'{col2}_mavg'] = df[col2].rolling(window=window).mean()
    df['mavg_diff'] = df[f'{col1}_mavg'] - df[f'{col2}_mavg']
    
    # Colors and styling
    mycolor = '#5ECB43'
    background_color = '#3c3d3d'
    my_font_path = os.getcwd()+ r'\Athletic\Nexa-ExtraLight.ttf'
    my_font_props = font_manager.FontProperties(fname=my_font_path)
    c1, c2 = colors
    
    plt.figure(figsize=(10, 6), facecolor=background_color)
    ax = plt.gca()
    ax.set_facecolor(background_color)
    # Draw lines
    plt.plot(df[x], df[f'{col1}_mavg'], label=col1, color=c1)
    plt.plot(df[x], df[f'{col2}_mavg'], label=col2, color=c2)
    # Fill area between
    plt.fill_between(df[x], df[f'{col1}_mavg'], df[f'{col2}_mavg'],
                     where=(df[f'{col1}_mavg'] > df[f'{col2}_mavg']),
                     interpolate=True, color=c1, alpha=0.3)
    
    plt.fill_between(df[x], df[f'{col1}_mavg'], df[f'{col2}_mavg'],
                     where=(df[f'{col1}_mavg'] < df[f'{col2}_mavg']),
                     interpolate=True, color=c2, alpha=0.3)
    
    # Find max and min of xG difference
    max_idx = df['mavg_diff'].idxmax()
    min_idx = df['mavg_diff'].idxmin()
    
    # MAX arrow
    x_max = df.loc[max_idx, x]
    y1_max = df.loc[max_idx, f'{col1}_mavg']
    y2_max = df.loc[max_idx, f'{col2}_mavg']
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
    x_min = df.loc[min_idx, x]
    y1_min = df.loc[min_idx, f'{col1}_mavg']
    y2_min = df.loc[min_idx, f'{col2}_mavg']
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
    plt.ylabel(y_title, color='white')
    plt.title(title, fontsize=17, color='white')
    plt.legend()
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.grid(True)
    plt.tick_params(colors='white')
    xmax = int(len(df)*1.1)
    plt.xticks(ticks=range(window, xmax, int(window/2)))
    y_max_value = np.array([*[df[f'{col1}_mavg'].dropna()], *[df[f'{col2}_mavg'].dropna()]]).max()
    plt.yticks(ticks=np.arange(0.4, y_max_value, 0.2))
    plt.text(xmax-4, y_max_value*1.07,
             'ADAM JAKUS', color=mycolor,
             fontsize = 15, fontproperties=my_font_props,
             ha='center', va='center')
    # Create before-after line
    plt.axvline(x=b4a_line, color=b4a_color, linestyle='--', linewidth=1.5)
    plt.text(b4a_line*0.98, y_max_value*1.05,
             b4a_text, color=b4a_color, va='top', rotation=90)
    plt.tight_layout()
    if save:
        # Save plot
        from datetime import datetime, time
        today_str = datetime.today().strftime('%Y.%m.%d.')
        save_path = f"{save_path}\{today_str}, {save_name}"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()
        
create_mavg_chart(df, x='nr', col1='Goals', col2='xG', window=4,
                  title='Aston Villa Goals & xG', y_title='Value',
                  b4a_line=33, b4a_text='Debut of winter signings ', b4a_color='orange',
                  save=True, save_path=r'C:\Users\Adam\Dropbox\TSDP_output\fbref\2025.04', save_name='Aston Villa Goals & xG.png')