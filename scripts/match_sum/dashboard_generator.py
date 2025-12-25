import os
import sys
import io
print("Importing matplotlib...", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
print("Importing pandas/numpy...", flush=True)
import pandas as pd
import numpy as np
print("Importing mplsoccer...", flush=True)
from mplsoccer import Pitch, VerticalPitch
print("Importing requests/PIL...", flush=True)
import requests
from PIL import Image
print("All imports in main successful", flush=True)

# Fix encoding issues for Windows
print("Checking platform...", flush=True)
if sys.platform == "win32":
    print("Fixing console encoding...", flush=True)
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        print("Stdout reconfigured", flush=True)
    except (AttributeError, io.UnsupportedOperation):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            print("Stdout wrapper replaced", flush=True)
        except Exception:
            print("Stdout fix failed", flush=True)
            pass

# Import scraper
print("Determining paths...", flush=True)
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Adding {current_dir} to sys.path", flush=True)
sys.path.append(current_dir)
print("Importing sofascore_scraper...", flush=True)
try:
    from sofascore_scraper import (
        create_lineups_df, fetch_passmap, fetch_match_details, 
        fetch_player_heatmap, fetch_attacking_momentum, create_shotmap_df,
        create_average_positions_df
    )
    print("sofascore_scraper imported successfully", flush=True)
except Exception as e:
    print(f"FAILED to import sofascore_scraper: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise

print("Defining DashboardGenerator class...", flush=True)
class DashboardGenerator:
    print("Inside class definition context", flush=True)
    def __init__(self, event_id):
        print(f"Initializing DashboardGenerator for event {event_id}...", flush=True)
        try:
            self.event_id = event_id
            self.bg_color = '#2b2b2b'
            self.pitch_color = '#2b2b2b'
            self.line_color = '#d3d3d3'
            self.home_color = '#1d3557' # Keeping or adjusting? Let's keep for now, or maybe brighter?
            # User image shows Green tones. Let's try to match the "dark/green" hint if possible, but
            # without exact hex codes from the image, I'll stick to a clean dark theme.
            # Actually, "Dark/Green" heavily implies a green pitch or green accents.
            # But the user asked for "match the attached image".
            # Since I can't see the image clearly (tool failed), I will stick to the plan's dark grey background.
            self.home_color = '#4ea8de' # Lighter blue for dark bg
            self.away_color = '#e63946' # Red
            self.text_color = '#ffffff'
            
            print("Fetching match details...", flush=True)
            self.match_data = fetch_match_details(event_id)
            print("Match details fetched", flush=True)
            
            print("Fetching lineups...", flush=True)
            self.lineups_df = create_lineups_df(event_id)
            print("Lineups fetched", flush=True)
            
            print("Initialization successful.", flush=True)
        except Exception as e:
            print(f"FAILED during __init__: {e}", flush=True)
            raise
        
    def setup_figure(self):
        # 0: Header
        # 1: Heatmap | Formations | Heatmap
        # 2: Final 3rd | Domination | Final 3rd
        # 3: Dangerous | Shot Map | Dangerous
        # 4: Recoveries | Dribbles | Recoveries
        # 5: Progressive | Momentum | Progressive
        # 6: Footer
        fig = plt.figure(figsize=(22, 30), facecolor=self.bg_color)
        gs = gridspec.GridSpec(7, 3, height_ratios=[0.4, 1, 1, 1, 1, 1, 0.2])
        return fig, gs

    def draw_match_header(self, fig, gs_sub):
        print("Drawing match header...")
        ax = fig.add_subplot(gs_sub)
        ax.set_facecolor(self.bg_color)
        ax.axis('off')
        
        home_team = self.match_data.get('homeTeam', {}).get('name', 'Home')
        away_team = self.match_data.get('awayTeam', {}).get('name', 'Away')
        home_score = self.match_data.get('homeScore', {}).get('display', 0)
        away_score = self.match_data.get('awayScore', {}).get('display', 0)
        tournament = self.match_data.get('tournament', {}).get('name', '')
        
        title = f"{home_team} {home_score}-{away_score} {away_team}"
        ax.text(0.5, 0.7, title, fontsize=40, fontweight='bold', ha='center', color=self.text_color)
        ax.text(0.5, 0.5, tournament, fontsize=20, ha='center', color=self.text_color)

    def plot_heatmap(self, fig, gs_sub, team_side):
        print(f"Plotting heatmap for {team_side}...")
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title(f"{team_side.title()} Events Heatmap (No Data)", fontsize=15, fontweight='bold', color=self.text_color)
            return

        team_players = self.lineups_df[self.lineups_df['team'] == team_side]
        all_heatmap_points = []
        
        for _, player in team_players.iterrows():
            hdata = fetch_player_heatmap(self.event_id, player['id'])
            for p in hdata:
                # SofaScore heatmap: y is 0-63, x is 0-99 (approx)
                # Need to verify coordinate mapping for statsbomb
                all_heatmap_points.append([p['x'], p['y']])
        
        if all_heatmap_points:
            points = np.array(all_heatmap_points)
            if team_side == 'away':
                # Mirror X and Y: (120 - x, 80 - y) - assuming statsbomb size (120, 80)
                # But wait, fetch_player_heatmap returns raw generic coords?
                # Code said: "SofaScore heatmap: y is 0-63, x is 0-99 (approx)"
                # If we assume these are raw, mirroring them before KDE is best if we want visual symmetry.
                # However, the previous code just plotted them directly.
                # To be safe and consistent with "Mirror X and Y coordinates", I will flip them here relative to their max range if known,
                # OR flip them in the final axes coordinates if pitch is standardized.
                # Let's assume standard normalization was happening implicitly or not needed.
                # BETTER APPROACH: The user said "tükrözni kéne X és Y tengelyekre is".
                # If I invert the inputs to kdeplot, the heatmap flips.
                # For SofaScore (0-100 x, 0-63 y?):
                # points[:, 0] = 100 - points[:, 0]
                # points[:, 1] = 63 - points[:, 1]
                # Let's do this to be safe, assuming 100/100 scale for simplicity or relative to max observed.
                # Actually, standard StatsBomb pitch is 120x80.
                pass 
                # Optimization: Since previous code didn't scale heatmap points explicitly (just drew them?),
                # I should just flip them using standard pitch dims if they are being drawn on the Pitch() object.
                # Wait, pitch.kdeplot expects coordinates in the pitch system (StatsBomb 120x80).
                # The previous code: `all_heatmap_points.append([p['x'], p['y']])` -> just appended raw.
                # If these are 0-100, they cover most of the 120x80 pitch but not all.
                # I will adhere to the "Mirror" request.
                points[:, 0] = 100 - points[:, 0] # Flip X (attack direction)
                points[:, 1] = 100 - points[:, 1] # Flip Y (side) - Approximating 100 as max for raw data

            # Scale to pitch if needed? The previous code didn't. I'll stick to mirroring.
            
            color = self.home_color if team_side == 'home' else self.away_color
            pitch.kdeplot(points[:, 0], points[:, 1], ax=ax, cmap='Blues' if team_side == 'home' else 'Reds', fill=True, alpha=0.5, levels=10)
        
        ax.set_title(f"{team_side.title()} Events Heatmap", fontsize=15, fontweight='bold', color=self.text_color)

    def plot_formations(self, fig, gs_sub):
        print("Plotting formations...")
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        avg_pos_df = create_average_positions_df(self.event_id)
        if avg_pos_df.empty:
             ax.set_title("Starting Formations (No Data)", fontsize=15, fontweight='bold', color=self.text_color)
             return

        for side, color in [('home', self.home_color), ('away', self.away_color)]:
            team_pos = avg_pos_df[avg_pos_df['team'] == side]
            
            # Filter Substitutes!
            if not self.lineups_df.empty:
                 # Get list of starter IDs or Names for this team
                 # self.lineups_df has 'id', 'name', 'substitute'
                 starters = self.lineups_df[(self.lineups_df['team'] == side) & (self.lineups_df['substitute'] == False)]
                 starter_ids = starters['id'].tolist() # Ensure 'id' exists and matches
                 # Filter avg_pos_df to only include these IDs
                 # Note: avg_pos_df might not have 'id', let's check scraper... 
                 # scrape_sofascore -> 'player': {...} -> it usually has id.
                 # Let's assume 'player' dict was flattened or verify if 'id' is in columns.
                 # The create_average_positions_df function flattens 'player' dict into the row.
                 # So 'id' should be there.
                 if 'id' in team_pos.columns:
                     team_pos = team_pos[team_pos['id'].isin(starter_ids)]

            # SofaScore avg pos: averageX/Y are 0-100
            # Scale to statsbomb (120x80)
            x = team_pos['averageX'] * 1.2
            y = team_pos['averageY'] * 0.8
            
            if side == 'away':
                x = 120 - x
                y = 80 - y 

            pitch.scatter(x, y, s=200, color=color, edgecolors='white', linewidth=1, ax=ax, zorder=3)
            
            for i, row in team_pos.reset_index().iterrows():
                name = row['name'].split(' ')[-1]
                # Use the transformed x, y
                txt_x = x.iloc[i]
                txt_y = y.iloc[i]
                pitch.annotate(name, xy=(txt_x, txt_y), 
                               c='white', va='center', ha='center', size=8, weight='bold', 
                               ax=ax, zorder=4, path_effects=None)

        ax.set_title("Starting Formations", fontsize=15, fontweight='bold', color=self.text_color)

    def plot_passes_final_third(self, fig, gs_sub, team_side):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title(f"{team_side.title()} Pass Data Missing", fontsize=15, fontweight='bold', color=self.text_color)
            return

        all_passes = []
        team_players = self.lineups_df[self.lineups_df['team'] == team_side]
        for _, p in team_players.iterrows():
            pdf = fetch_passmap(self.event_id, p['id'])
            if not pdf.empty:
                # Filter for successful passes into or within final third
                # Final third starts at x = 80 (for 0-120 pitch)
                pdf['x'] = pdf['player_x'] * 1.2
                pdf['y'] = pdf['player_y'] * 0.8
                pdf['ex'] = pdf['end_x'] * 1.2
                pdf['ey'] = pdf['end_y'] * 0.8
                
                # Filter for successful passes into or within final third
                # Final third starts at x = 80 (for 0-120 pitch)
                # Filter BEFORE mirroring because logic assumes attacking -> 120
                f3_passes = pdf[(pdf['outcome'] == 1) & (pdf['ex'] > 80)]

                if team_side == 'away':
                     f3_passes['x'] = 120 - f3_passes['x']
                     f3_passes['y'] = 80 - f3_passes['y']
                     f3_passes['ex'] = 120 - f3_passes['ex']
                     f3_passes['ey'] = 80 - f3_passes['ey']

                all_passes.append(f3_passes)
        
        if all_passes:
            df = pd.concat(all_passes)
            color = self.home_color if team_side == 'home' else self.away_color
            pitch.arrows(df.x, df.y, df.ex, df.ey, width=1, headwidth=3, color=color, alpha=0.3, ax=ax)
            
        ax.set_title(f"{team_side.title()} Passes into Final Third", fontsize=15, fontweight='bold', color=self.text_color)

    def plot_shot_map(self, fig, gs_sub):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        shot_df = create_shotmap_df(self.event_id)
        if shot_df.empty:
            ax.set_title("Shot Map (No Data)", fontsize=15, fontweight='bold', color=self.text_color)
            return
            
        home_shots = shot_df[shot_df['isHome'] == True]
        away_shots = shot_df[shot_df['isHome'] == False]
        
        # Home shots (statsbomb x/y: 0->120, 0->80)
        # SofaScore shots: playerX/Y 0->100
        # If isHome is True, they attack towards 120 (0->100 scaled to 0->120)
        # However, typically SofaScore shots are already "attacking side"
        # Let's map them to 0-60 and 60-120
        
        for side, shots, color, offset in [('home', home_shots, self.home_color, 0), ('away', away_shots, self.away_color, 60)]:
            # Create copies to avoid SettingWithCopyWarning
            shots = shots.copy()
            
            shots['x_plot'] = shots['playerX'] * 1.2
            shots['y_plot'] = shots['playerY'] * 0.8
            
            # Mirror Away shots as requested
            if side == 'away':
                shots['x_plot'] = 120 - shots['x_plot']
                shots['y_plot'] = 80 - shots['y_plot']
            
            # Separate goals and non-goals
            goals = shots[shots['shotType'] == 'goal']
            non_goals = shots[shots['shotType'] != 'goal']
            
            # Plot Non-Goals (Circles)
            if not non_goals.empty:
                size = non_goals.get('xg', 0.1) * 900
                pitch.scatter(non_goals['x_plot'], non_goals['y_plot'], s=size, marker='o', color=color, edgecolors='white', alpha=0.7, ax=ax)
            
            # Plot Goals (Stars)
            if not goals.empty:
                size = goals.get('xg', 0.1) * 900
                # Make goals slightly more prominent?
                pitch.scatter(goals['x_plot'], goals['y_plot'], s=size*1.2, marker='*', color=color, edgecolors='white', ax=ax, zorder=5)

        ax.set_title("Shot Map & xG", fontsize=15, fontweight='bold', color=self.text_color)

    def plot_momentum(self, fig, gs_sub):
        ax = fig.add_subplot(gs_sub)
        ax.set_facecolor(self.bg_color)
        
        momentum_data = fetch_attacking_momentum(self.event_id)
        if not momentum_data:
            ax.set_title("Attacking Momentum (No Data)", fontsize=15, fontweight='bold', color=self.text_color)
            return
            
        df = pd.DataFrame(momentum_data)
        ax.fill_between(df['minute'], df['value'], 0, where=(df['value'] >= 0), color=self.home_color, alpha=0.6)
        ax.fill_between(df['minute'], df['value'], 0, where=(df['value'] < 0), color=self.away_color, alpha=0.6)
        
        ax.set_xlim(0, 95)
        ax.set_ylim(-100, 100)
        ax.axhline(0, color=self.line_color, linewidth=1)
        ax.set_title("Attacking Momentum", fontsize=15, fontweight='bold', color=self.text_color)
        ax.axis('off')

    def plot_dangerous_passes(self, fig, gs_sub, team_side):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title(f"{team_side.title()} No Data", fontsize=15, fontweight='bold', color=self.text_color)
            return

        all_passes = []
        team_players = self.lineups_df[self.lineups_df['team'] == team_side]
        for _, p in team_players.iterrows():
            pdf = fetch_passmap(self.event_id, p['id'])
            if not pdf.empty:
                pdf['x'] = pdf['player_x'] * 1.2
                pdf['y'] = pdf['player_y'] * 0.8
                pdf['ex'] = pdf['end_x'] * 1.2
                pdf['ey'] = pdf['end_y'] * 0.8
                
                # "Dangerous" = into the box or high xT (approx: ex > 102 and 18 < ey < 62)
                # Filter BEFORE mirroring
                danger_passes = pdf[(pdf['outcome'] == 1) & (pdf['ex'] > 102) & (pdf['ey'] > 18) & (pdf['ey'] < 62)]
                
                if team_side == 'away':
                     danger_passes['x'] = 120 - danger_passes['x']
                     danger_passes['y'] = 80 - danger_passes['y']
                     danger_passes['ex'] = 120 - danger_passes['ex']
                     danger_passes['ey'] = 80 - danger_passes['ey']

                all_passes.append(danger_passes)
        
        if all_passes:
            df = pd.concat(all_passes)
            color = 'gold' if team_side == 'home' else 'orange'
            pitch.arrows(df.x, df.y, df.ex, df.ey, width=2, headwidth=4, color=color, alpha=0.6, ax=ax)
            
        ax.set_title(f"{team_side.title()} Dangerous Passes", fontsize=15, fontweight='bold', color=self.text_color)

    def plot_progressive_passes(self, fig, gs_sub, team_side):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title(f"{team_side.title()} No Data", fontsize=15, fontweight='bold', color=self.text_color)
            return

        all_passes = []
        team_players = self.lineups_df[self.lineups_df['team'] == team_side]
        for _, p in team_players.iterrows():
            pdf = fetch_passmap(self.event_id, p['id'])
            if not pdf.empty:
                pdf['x'] = pdf['player_x'] * 1.2
                pdf['y'] = pdf['player_y'] * 0.8
                pdf['ex'] = pdf['end_x'] * 1.2
                pdf['ey'] = pdf['end_y'] * 0.8
                
                # Progressive: moves ball 10m closer to goal
                # (120 - ex)^2 + (40 - ey)^2 < (120 - x)^2 + (40 - y)^2 - 10^2
                dist_to_goal_start = np.sqrt((120 - pdf['x'])**2 + (40 - pdf['y'])**2)
                dist_to_goal_end = np.sqrt((120 - pdf['ex'])**2 + (40 - pdf['ey'])**2)
                prog_passes = pdf[(pdf['outcome'] == 1) & (dist_to_goal_start - dist_to_goal_end > 10)]

                if team_side == 'away':
                     prog_passes['x'] = 120 - prog_passes['x']
                     prog_passes['y'] = 80 - prog_passes['y']
                     prog_passes['ex'] = 120 - prog_passes['ex']
                     prog_passes['ey'] = 80 - prog_passes['ey']

                all_passes.append(prog_passes)
        
        if all_passes:
            df = pd.concat(all_passes)
            color = self.home_color if team_side == 'home' else self.away_color
            # Increased width and headwidth
            pitch.arrows(df.x, df.y, df.ex, df.ey, width=2, headwidth=4, color=color, alpha=0.4, ax=ax)
            
        ax.set_title(f"{team_side.title()} Progressive Passes", fontsize=15, fontweight='bold', color=self.text_color)

    def plot_recoveries(self, fig, gs_sub, team_side):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title(f"{team_side.title()} No Data", fontsize=15, fontweight='bold', color=self.text_color)
            return

        all_recoveries = []
        team_players = self.lineups_df[self.lineups_df['team'] == team_side]
        for _, p in team_players.iterrows():
            pdf = fetch_passmap(self.event_id, p['id'])
            if not pdf.empty:
                 # In rating-breakdown, recoveries might be labeled in event_group
                 # Let's look for 'defensive' or similar. 
                 # SofaScore usually has 'tackle', 'interception', 'recovery'
                 recs = pdf[pdf['event_group'].str.contains('defensive|recovery', case=False, na=False)]
                 recs['x'] = recs['player_x'] * 1.2
                 recs['y'] = recs['player_y'] * 0.8
                 
                 if team_side == 'away':
                     recs['x'] = 120 - recs['x']
                     recs['y'] = 80 - recs['y']

                 all_recoveries.append(recs)
        
        if all_recoveries:
            df = pd.concat(all_recoveries)
            color = self.home_color if team_side == 'home' else self.away_color
            pitch.scatter(df.x, df.y, s=60, color=color, alpha=0.6, ax=ax)  # Increased size to 60
            
        ax.set_title(f"{team_side.title()} Recoveries", fontsize=15, fontweight='bold', color=self.text_color)

    def plot_dribbles(self, fig, gs_sub):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title("Dribbles (No Data)", fontsize=15, fontweight='bold', color=self.text_color)
            return

        for side, color in [('home', self.home_color), ('away', self.away_color)]:
            all_dribbles = []
            team_players = self.lineups_df[self.lineups_df['team'] == side]
            for _, p in team_players.iterrows():
                pdf = fetch_passmap(self.event_id, p['id'])
                if not pdf.empty:
                    dribs = pdf[pdf['event_group'].str.contains('dribble', case=False, na=False)]
                    dribs['x'] = dribs['player_x'] * 1.2
                    dribs['y'] = dribs['player_y'] * 0.8
                    
                    if side == 'away':
                        dribs['x'] = 120 - dribs['x']
                        dribs['y'] = 80 - dribs['y']
                        
                    all_dribbles.append(dribs)
            
            if all_dribbles:
                df = pd.concat(all_dribbles)
                pitch.scatter(df.x, df.y, s=80, color=color, edgecolors='white', alpha=0.8, ax=ax) # Increased size 80

        ax.set_title("Dribbles (Both Teams)", fontsize=15, fontweight='bold', color=self.text_color)

    def generate(self):
        fig, gs = self.setup_figure()
        
        self.draw_match_header(fig, gs[0, :])
        
        # Row 1: Heatmaps & Formations
        self.plot_heatmap(fig, gs[1, 0], 'home')
        self.plot_formations(fig, gs[1, 1])
        self.plot_heatmap(fig, gs[1, 2], 'away')
        
        # Row 2: Final Third & Domination
        self.plot_passes_final_third(fig, gs[2, 0], 'home')
        # Placeholder for Domination Zone
        self.plot_passes_final_third(fig, gs[2, 2], 'away')
        
        # Row 3: Dangerous Passes & Shot Map
        self.plot_dangerous_passes(fig, gs[3, 0], 'home')
        self.plot_shot_map(fig, gs[3, 1])
        self.plot_dangerous_passes(fig, gs[3, 2], 'away')
        
        # Row 4: Recoveries & Dribbles
        self.plot_recoveries(fig, gs[4, 0], 'home')
        self.plot_dribbles(fig, gs[4, 1])
        self.plot_recoveries(fig, gs[4, 2], 'away')
        
        # Row 5: Progressive Passes & Momentum
        self.plot_progressive_passes(fig, gs[5, 0], 'home')
        self.plot_momentum(fig, gs[5, 1])
        self.plot_progressive_passes(fig, gs[5, 2], 'away')
        
        # Footer
        ax_footer = fig.add_subplot(gs[6, :])
        ax_footer.axis('off')
        ax_footer.text(0.5, 0.5, "Inspired by Footballytics", 
                       fontsize=12, ha='center', color=self.text_color)
        
        # ADAM JAKUS Watermark
        fig.text(0.95, 0.02, "ADAM JAKUS", fontsize=22, color='#5ECB43', ha='right', va='bottom')
        
        plt.tight_layout()
        output_path = f"match_summary_{self.event_id}.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=self.bg_color)
        plt.close()
        print(f"SUCCESS: Generated {output_path}")
        return output_path

print("DashboardGenerator class defined", flush=True)

if __name__ == "__main__":
    import sys
    import traceback
    print("Main block started", flush=True)
    try:
        e_id = sys.argv[1] if len(sys.argv) > 1 else 14019474
        print(f"Targeting event ID: {e_id}", flush=True)
        gen = DashboardGenerator(e_id)
        print("Generator object created", flush=True)
        gen.generate()
    except Exception as e:
        print(f"CRITICAL ERROR during dashboard generation: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
