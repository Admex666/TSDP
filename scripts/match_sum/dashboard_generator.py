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
            self.bg_color = '#f4eee0'
            self.pitch_color = '#f4eee0'
            self.line_color = '#4a4a4a'
            self.home_color = '#1d3557'
            self.away_color = '#e63946'
            self.text_color = '#22312b'
            
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
        ax.text(0.5, 0.3, tournament, fontsize=20, ha='center', color=self.text_color)

    def plot_heatmap(self, fig, gs_sub, team_side):
        print(f"Plotting heatmap for {team_side}...")
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title(f"{team_side.title()} Events Heatmap (No Data)", fontsize=15, fontweight='bold')
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
            color = self.home_color if team_side == 'home' else self.away_color
            pitch.kdeplot(points[:, 0], points[:, 1], ax=ax, cmap='Blues' if team_side == 'home' else 'Reds', fill=True, alpha=0.5, levels=10)
        
        ax.set_title(f"{team_side.title()} Events Heatmap", fontsize=15, fontweight='bold')

    def plot_formations(self, fig, gs_sub):
        print("Plotting formations...")
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        avg_pos_df = create_average_positions_df(self.event_id)
        if avg_pos_df.empty:
             ax.set_title("Starting Formations (No Data)", fontsize=15, fontweight='bold')
             return

        for side, color in [('home', self.home_color), ('away', self.away_color)]:
            team_pos = avg_pos_df[avg_pos_df['team'] == side]
            # SofaScore avg pos: averageX/Y are 0-100
            # Scale to statsbomb (120x80)
            x = team_pos['averageX'] * 1.2
            y = team_pos['averageY'] * 0.8
            pitch.scatter(x, y, s=200, color=color, edgecolors='white', linewidth=1, ax=ax, zorder=3)
            
            for _, row in team_pos.iterrows():
                name = row['name'].split(' ')[-1]
                pitch.annotate(name, xy=(row['averageX']*1.2, row['averageY']*0.8), 
                               c='black', va='center', ha='center', size=8, weight='bold', 
                               ax=ax, zorder=4, path_effects=None)

        ax.set_title("Starting Formations", fontsize=15, fontweight='bold')

    def plot_passes_final_third(self, fig, gs_sub, team_side):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title(f"{team_side.title()} Pass Data Missing", fontsize=15, fontweight='bold')
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
                
                # If away, they might be mirrored in SofaScore API or we need to flip
                # SofaScore data is usually consistent (attacking direction of the team in that half)
                # But let's assume they all attack 0 -> 100 in the raw data
                f3_passes = pdf[(pdf['outcome'] == 1) & (pdf['ex'] > 80)]
                all_passes.append(f3_passes)
        
        if all_passes:
            df = pd.concat(all_passes)
            color = self.home_color if team_side == 'home' else self.away_color
            pitch.arrows(df.x, df.y, df.ex, df.ey, width=1, headwidth=3, color=color, alpha=0.3, ax=ax)
            
        ax.set_title(f"{team_side.title()} Passes into Final Third", fontsize=15, fontweight='bold')

    def plot_shot_map(self, fig, gs_sub):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        shot_df = create_shotmap_df(self.event_id)
        if shot_df.empty:
            ax.set_title("Shot Map (No Data)", fontsize=15, fontweight='bold')
            return
            
        home_shots = shot_df[shot_df['isHome'] == True]
        away_shots = shot_df[shot_df['isHome'] == False]
        
        # Home shots (statsbomb x/y: 0->120, 0->80)
        # SofaScore shots: playerX/Y 0->100
        # If isHome is True, they attack towards 120 (0->100 scaled to 0->120)
        # However, typically SofaScore shots are already "attacking side"
        # Let's map them to 0-60 and 60-120
        
        for side, shots, color, offset in [('home', home_shots, self.home_color, 0), ('away', away_shots, self.away_color, 60)]:
            x = shots['playerX'] * 1.2
            y = shots['playerY'] * 0.8
            # If away, mirror to the right side if SofaScore gave 0-100 attacking coords
            # Usually SofaScore is 0-100 for the team's attack. Let's flip away to 120-0
            if side == 'away':
                x = 120 - x
                y = 80 - y # mirror y too for consistency if needed
            
            # Scatter by xG size if available (SofaScore uses 'xg')
            size = shots.get('xg', 0.1) * 500
            pitch.scatter(x, y, s=size, color=color, edgecolors='white', alpha=0.7, ax=ax)

        ax.set_title("Shot Map & xG", fontsize=15, fontweight='bold')

    def plot_momentum(self, fig, gs_sub):
        ax = fig.add_subplot(gs_sub)
        ax.set_facecolor(self.bg_color)
        
        momentum_data = fetch_attacking_momentum(self.event_id)
        if not momentum_data:
            ax.set_title("Attacking Momentum (No Data)", fontsize=15, fontweight='bold')
            return
            
        df = pd.DataFrame(momentum_data)
        ax.fill_between(df['minute'], df['value'], 0, where=(df['value'] >= 0), color=self.home_color, alpha=0.6)
        ax.fill_between(df['minute'], df['value'], 0, where=(df['value'] < 0), color=self.away_color, alpha=0.6)
        
        ax.set_xlim(0, 95)
        ax.set_ylim(-100, 100)
        ax.axhline(0, color=self.line_color, linewidth=1)
        ax.set_title("Attacking Momentum", fontsize=15, fontweight='bold')
        ax.axis('off')

    def plot_dangerous_passes(self, fig, gs_sub, team_side):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title(f"{team_side.title()} No Data", fontsize=15, fontweight='bold')
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
                danger_passes = pdf[(pdf['outcome'] == 1) & (pdf['ex'] > 102) & (pdf['ey'] > 18) & (pdf['ey'] < 62)]
                all_passes.append(danger_passes)
        
        if all_passes:
            df = pd.concat(all_passes)
            color = 'gold' if team_side == 'home' else 'orange'
            pitch.arrows(df.x, df.y, df.ex, df.ey, width=2, headwidth=4, color=color, alpha=0.6, ax=ax)
            
        ax.set_title(f"{team_side.title()} Dangerous Passes", fontsize=15, fontweight='bold')

    def plot_progressive_passes(self, fig, gs_sub, team_side):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title(f"{team_side.title()} No Data", fontsize=15, fontweight='bold')
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
                all_passes.append(prog_passes)
        
        if all_passes:
            df = pd.concat(all_passes)
            color = self.home_color if team_side == 'home' else self.away_color
            pitch.arrows(df.x, df.y, df.ex, df.ey, width=1, headwidth=3, color=color, alpha=0.4, ax=ax)
            
        ax.set_title(f"{team_side.title()} Progressive Passes", fontsize=15, fontweight='bold')

    def plot_recoveries(self, fig, gs_sub, team_side):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title(f"{team_side.title()} No Data", fontsize=15, fontweight='bold')
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
                 all_recoveries.append(recs)
        
        if all_recoveries:
            df = pd.concat(all_recoveries)
            color = self.home_color if team_side == 'home' else self.away_color
            pitch.scatter(df.x, df.y, s=30, color=color, alpha=0.6, ax=ax)
            
        ax.set_title(f"{team_side.title()} Recoveries", fontsize=15, fontweight='bold')

    def plot_dribbles(self, fig, gs_sub):
        ax = fig.add_subplot(gs_sub)
        pitch = Pitch(pitch_type='statsbomb', pitch_color=self.pitch_color, line_color=self.line_color)
        pitch.draw(ax=ax)
        
        if self.lineups_df.empty or 'team' not in self.lineups_df.columns:
            ax.set_title("Dribbles (No Data)", fontsize=15, fontweight='bold')
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
                    all_dribbles.append(dribs)
            
            if all_dribbles:
                df = pd.concat(all_dribbles)
                pitch.scatter(df.x, df.y, s=40, color=color, edgecolors='white', alpha=0.8, ax=ax)

        ax.set_title("Dribbles (Both Teams)", fontsize=15, fontweight='bold')

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
        ax_footer.text(0.5, 0.5, "Generated by MatchSum Analytics | Data via SofaScore", 
                       fontsize=12, ha='center', color=self.text_color)
        
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
