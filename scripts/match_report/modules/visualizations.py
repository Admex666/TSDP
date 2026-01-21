"""
Visualization module for creating charts and graphs
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mplsoccer import VerticalPitch, Pitch
import streamlit as st


class Visualizations:
    """Create all visualizations for the match report"""
    
    @staticmethod
    def create_radar_chart(home_metrics: Dict, away_metrics: Dict, 
                          home_team: str, away_team: str) -> go.Figure:
        """Create radar chart comparing two teams"""
        
        categories = [
            'Goals/90',
            'xG/90',
            'Possession %',
            'Pass Accuracy %',
            'Shot Accuracy %',
            'Tackles/90',
            'Aerial Duels %'
        ]
        
        # Normalize values to 0-100 scale
        home_values = [
            min(home_metrics.get('goals_per_90', 0) * 30, 100),  # Scale goals
            min(home_metrics.get('xg_per_90', 0) * 30, 100),     # Scale xG
            home_metrics.get('possession_pct', 0),
            home_metrics.get('pass_accuracy', 0),
            home_metrics.get('shooting_accuracy', 0),
            min(home_metrics.get('tackles_per_90', 0) * 5, 100),  # Scale tackles
            home_metrics.get('aerial_duels_won_pct', 0)
        ]
        
        away_values = [
            min(away_metrics.get('goals_per_90', 0) * 30, 100),
            min(away_metrics.get('xg_per_90', 0) * 30, 100),
            away_metrics.get('possession_pct', 0),
            away_metrics.get('pass_accuracy', 0),
            away_metrics.get('shooting_accuracy', 0),
            min(away_metrics.get('tackles_per_90', 0) * 5, 100),
            away_metrics.get('aerial_duels_won_pct', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=home_values,
            theta=categories,
            fill='toself',
            name=home_team,
            line=dict(color='#3498db', width=2),
            fillcolor='rgba(52, 152, 219, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=away_values,
            theta=categories,
            fill='toself',
            name=away_team,
            line=dict(color='#e74c3c', width=2),
            fillcolor='rgba(231, 76, 60, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Team Comparison",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_form_chart(form_matches: List[Dict], team_id: int, team_name: str) -> go.Figure:
        """Create line chart showing goals scored/conceded over recent matches"""
        
        matches_data = []
        for i, match in enumerate(form_matches[:10]):
            try:
                is_home = match['homeTeam']['id'] == team_id
                
                # Safely extract scores - handle different API response structures
                home_score = match.get('homeScore', {}).get('current', 
                             match.get('homeScore', {}).get('display', 0))
                away_score = match.get('awayScore', {}).get('current',
                             match.get('awayScore', {}).get('display', 0))
                
                if is_home:
                    goals_for = home_score
                    goals_against = away_score
                    opponent = match['awayTeam']['name']
                else:
                    goals_for = away_score
                    goals_against = home_score
                    opponent = match['homeTeam']['name']
                
                matches_data.append({
                    'match_num': len(form_matches[:10]) - i,
                    'opponent': opponent,
                    'goals_for': goals_for,
                    'goals_against': goals_against
                })
            except (KeyError, TypeError) as e:
                # Skip matches with incomplete data
                continue
        
        matches_data.reverse()
        df = pd.DataFrame(matches_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['match_num'],
            y=df['goals_for'],
            mode='lines+markers',
            name='Goals Scored',
            line=dict(color='#27ae60', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['match_num'],
            y=df['goals_against'],
            mode='lines+markers',
            name='Goals Conceded',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"{team_name} - Last 10 Matches",
            xaxis_title="Match Number (Recent →)",
            yaxis_title="Goals",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_shot_map(shotmap_df: pd.DataFrame, team_name: str, team_color: str = '#3498db') -> plt.Figure:
        """Create shot map visualization using mplsoccer"""
        
        if shotmap_df.empty:
            # Create empty figure with message
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.text(0.5, 0.5, 'No shot data available', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        
        # Create vertical pitch
        pitch = VerticalPitch(
            pitch_type='statsbomb',
            pitch_color='#22312b',
            line_color='white',
            linewidth=2,
            half=True  # Show only attacking half
        )
        
        fig, ax = pitch.draw(figsize=(8, 10))
        
        # Define colors for different shot types
        shot_colors = {
            'goal': '#ff4444',
            'on-target': '#ffaa00',
            'off-target': '#ffffff',
            'blocked': '#666666'
        }
        
        # Plot shots
        for shot_type, color in shot_colors.items():
            shots = shotmap_df[shotmap_df['shotType'] == shot_type]
            
            if not shots.empty:
                # Size based on xG if available
                if 'xG' in shots.columns:
                    sizes = shots['xG'] * 500
                else:
                    sizes = 100
                
                pitch.scatter(
                    shots['playerX'],
                    shots['playerY'],
                    s=sizes,
                    c=color,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=1,
                    ax=ax,
                    label=shot_type.capitalize()
                )
        
        # Add legend
        ax.legend(loc='upper left', fontsize=10, framealpha=0.8)
        
        # Add title and stats
        total_shots = len(shotmap_df)
        goals = len(shotmap_df[shotmap_df['shotType'] == 'goal'])
        on_target = len(shotmap_df[shotmap_df['shotType'].isin(['goal', 'on-target'])])
        
        if 'xG' in shotmap_df.columns:
            total_xg = shotmap_df['xG'].sum()
            title = f"{team_name}\nShots: {total_shots} | Goals: {goals} | On Target: {on_target} | xG: {total_xg:.2f}"
        else:
            title = f"{team_name}\nShots: {total_shots} | Goals: {goals} | On Target: {on_target}"
        
        ax.set_title(title, fontsize=14, color='white', pad=20)
        
        return fig
    
    @staticmethod
    def create_player_bars(players_data: List[Dict], metric: str, title: str, 
                          color: str = '#3498db') -> go.Figure:
        """Create horizontal bar chart for top players"""
        
        if not players_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No player data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Sort by metric value
        sorted_players = sorted(players_data, key=lambda x: x.get(metric, 0), reverse=True)[:5]
        
        names = [p['player']['name'] for p in sorted_players]
        values = [p.get(metric, 0) for p in sorted_players]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker=dict(color=color),
            text=values,
            textposition='auto'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis=dict(autorange="reversed"),
            height=300,
            margin=dict(l=150)
        )
        
        return fig
    
    @staticmethod
    def create_comparison_table(home_stats: Dict, away_stats: Dict, 
                               home_team: str, away_team: str) -> pd.DataFrame:
        """Create comparison table for key statistics"""
        
        metrics = {
            'Goals/90': ('goals_per_90', True),
            'xG/90': ('xg_per_90', True),
            'Shots/90': ('shots_per_90', True),
            'Shot Accuracy %': ('shooting_accuracy', True),
            'Possession %': ('possession_pct', True),
            'Pass Accuracy %': ('pass_accuracy', True),
            'Goals Conceded/90': ('goals_conceded_per_90', False),
            'Clean Sheets %': ('clean_sheet_pct', True),
            'Tackles/90': ('tackles_per_90', True),
            'Duels Won %': ('duels_won_pct', True)
        }
        
        data = []
        for metric_name, (key, higher_is_better) in metrics.items():
            home_val = home_stats.get(key, 0)
            away_val = away_stats.get(key, 0)
            
            # Determine which is better
            if higher_is_better:
                home_better = home_val > away_val
            else:
                home_better = home_val < away_val
            
            data.append({
                'Metric': metric_name,
                home_team: f"{'✓ ' if home_better else ''}{home_val}",
                away_team: f"{'✓ ' if not home_better else ''}{away_val}"
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_set_piece_comparison(home_sp: Dict, away_sp: Dict,
                                    home_team: str, away_team: str) -> go.Figure:
        """Create grouped bar chart for set piece comparison"""
        
        categories = ['Corners/90', 'FK Goals', 'Penalty Conv %']
        
        home_values = [
            home_sp.get('corners_per_90', 0),
            home_sp.get('free_kick_goals', 0),
            home_sp.get('penalty_conversion', 0)
        ]
        
        away_values = [
            away_sp.get('corners_per_90', 0),
            away_sp.get('free_kick_goals', 0),
            away_sp.get('penalty_conversion', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=home_team,
            x=categories,
            y=home_values,
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Bar(
            name=away_team,
            x=categories,
            y=away_values,
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title="Set Piece Comparison",
            barmode='group',
            height=400
        )
        
        return fig
