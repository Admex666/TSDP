import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import matplotlib.image as mpimg
import matplotlib.font_manager as font_manager
import urllib.request
from PIL import Image
from Athletic.understat_scraper import get_player_shotmap

# Set season, player and league
season = '2024'
player_id = '6160'
league = 'Serie_A'

df = get_player_shotmap(season, player_id, league)
url_logo = 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Bologna_F.C._1909_logo.svg/800px-Bologna_F.C._1909_logo.svg.png'
league_pretty = league.replace('_', ' ')
save = True

if len(df.player.unique()) > 1:
    print('Too many players.')
else:
    player = df.player[0]
    print(f'Player found: {player}')

if len(df.season.unique()) > 1:
    print('Too many seasons.')
else:
    year = str(df.season[0])
    print(f'Year found: {year}')

season = f'{year}-{int(year[-2:])+1}'
team = pd.Series([*df.h_team, *df.a_team]).mode()[0]

#%% Data shown
df['X'] = df['X']*100
df['Y'] = df['Y']*100

total_shots = df.shape[0]
total_goals = len(df.loc[df.result == 'Goal'])
total_xg = df.xG.sum()
xg_per_shot = total_xg / total_shots
points_avg_distance = df['X'].mean()
actual_avg_distance = 120 - (df['X'] * 1.2).mean()

#%% Colors
background_color = '#3c3d3d'
mycolor = '#5ECB43'

font_path = 'Athletic/Arvo-Regular.ttf'
my_font_path = 'Athletic/Nexa-ExtraLight.ttf'
font_props = font_manager.FontProperties(fname=font_path)
my_font_props = font_manager.FontProperties(fname=my_font_path)

fig = plt.figure(figsize=(8,12))
fig.patch.set_facecolor(background_color)

ax1 = fig.add_axes([0, 0.7, 1, 0.2])
ax1.set_facecolor(background_color)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)

# Player name
ax1.text(x=0.5, y=0.85,
         s=player, fontsize=20, fontproperties=font_props,
         fontweight='bold', color='white', ha='center')

# subtitle
ax1.text(x=0.5, y=0.75,
         s=f'All shots in the {league_pretty} {season}', fontsize=14, fontproperties=font_props,
         fontweight='bold', color='white', ha='center')

# Low- and high qual chance text
ax1.text(x=0.25, y=0.5,
         s='Low quality chance', fontsize=12, fontproperties=font_props,
         fontweight='bold', color='white', ha='center')
ax1.text(x=0.75, y=0.5,
         s='High quality chance', fontsize=12, fontproperties=font_props,
         fontweight='bold', color='white', ha='center')

# Small circle to big circle
for pos_x, size in zip([0, 0.05, 0.11, 0.17, 0.23],[1,2,3,4,5]):
    ax1.scatter(x=(0.37+pos_x), y=0.53, s=(100*size), c=background_color, 
                edgecolor='white', linewidth=0.8)

# Plot Goal and No Goal circles
ax1.text(x=0.45, y=0.27, s='Goal', fontsize=10, fontproperties=font_props, 
    color='white', ha='right')

ax1.scatter(x=0.47, y=0.3, s=100, color=mycolor, edgecolor='white', 
            linewidth=.8, alpha=.7)

ax1.scatter(x=0.53, y=0.3, s=100, color=background_color, 
    edgecolor='white', linewidth=.8)

ax1.text(x=0.55, y=0.27, s='No Goal', fontsize=10, fontproperties=font_props, 
    color='white', ha='left')

# Show images

# Download image 
urllib.request.urlretrieve(url_logo, 'temp_logo.png')

img = mpimg.imread('temp_logo.png')
ax_img = fig.add_axes([0.85, 0.81, 0.1, 0.1])
ax_img.imshow(img, alpha=0.99)

ax2 = fig.add_axes([0.05, 0.25, 0.9, 0.5])
ax2.set_facecolor(background_color)
# Add pitch to ax2
pitch = VerticalPitch(pitch_type='opta', half=True, pitch_color=background_color,
                      pad_bottom=0.5, line_color='white', linewidth=0.75,
                      axis=True, label=True)
pitch.draw(ax=ax2)

# Create avg distance 'lollipop'
ax2.scatter(x=90, y=points_avg_distance,
            s=100, color='white', linewidth=0.8)
ax2.plot([90,90], [100, points_avg_distance], color='white', linewidth=2)
ax2.text(x=90, y=points_avg_distance-4,
         s=f'Average Distance\n{actual_avg_distance:.1f} yards',
         fontsize=10, fontproperties=font_props, color='white', ha='center')

for i, x in df.iterrows():
    pitch.scatter(x['X'], x['Y'],
                  s=300*x['xG'], 
                  color=mycolor if x['result'] == 'Goal' else background_color,
                  ax=ax2, alpha=0.7, linewidth=0.8, edgecolor='white')


ax3 = fig.add_axes([0, 0.2, 1, 0.05])
ax3.set_facecolor(background_color)
# Plot texts of statistics
base = 0.08
for x_pos, title, var in zip([base, base+0.13, base+0.28, base+0.38],
                             ['Shots', 'Goals', 'xG', 'xG/Shot'],
                             [total_shots, total_goals, total_xg, xg_per_shot]):
    ax3.text(x=x_pos, y=0.5, s=title, fontsize=20, fontproperties=font_props,
             fontweight='bold', color='white', ha='left')
    ax3.text(x=x_pos, y=0, s=round(var,2), fontsize=16, fontproperties=font_props,
             color=mycolor, ha='left')
    
ax3.text(x=(base+0.57), y=0.35, s='ADAM JAKUS', fontsize=22, color=mycolor, fontproperties=my_font_props)
    
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax_img.axis('off')

# save or show
if save:
    from datetime import datetime
    todaystr = datetime.strftime(datetime.today(), format='%Y.%m.%d.')
    file_name = f'{todaystr}, {player} ({team})'
    save_path = f'C:/Users/Adam/Dropbox/TSDP_output/Athletic shotmaps/{file_name}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
else:
    plt.show()
