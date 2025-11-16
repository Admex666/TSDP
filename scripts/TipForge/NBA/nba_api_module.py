# Team abbreviations dict
TEAM_ABBREVIATIONS = {
    "Houston Rockets": "HOU",
    "Golden State Warriors": "GSW",
    "Los Angeles Lakers": "LAL",
    "Oklahoma City Thunder": "OKC",
    "Atlanta Hawks": "ATL",
    "Miami Heat": "MIA",
    "Brooklyn Nets": "BKN",
    "New York Knicks": "NYK",
    "Charlotte Hornets": "CHA",
    "Washington Wizards": "WAS",
    "Toronto Raptors": "TOR",
    "Chicago Bulls": "CHI",
    "Milwaukee Bucks": "MIL",
    "Detroit Pistons": "DET",
    "New Orleans Pelicans": "NOP",
    "Memphis Grizzlies": "MEM",
    "Cleveland Cavaliers": "CLE",
    "Orlando Magic": "ORL",
    "Minnesota Timberwolves": "MIN",
    "Utah Jazz": "UTA",
    "LA Clippers": "LAC",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "Dallas Mavericks": "DAL",
    "Phoenix Suns": "PHX",
    "San Antonio Spurs": "SAS",
    "Boston Celtics": "BOS",
    "Philadelphia 76ers": "PHI",
    "Indiana Pacers": "IND",
    "Denver Nuggets": "DEN"
}


# Season game log
from nba_api.stats.endpoints import leaguegamelog
def get_season_game_log(season='2025-26'):
    log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
    games = log.get_dict()

    print(f"Total games found: {len(games['resultSets'][0]['rowSet'])}")
    
    headers = games['resultSets'][0]['headers']
    rows = games['resultSets'][0]['rowSet']
    
    # DataFrame készítése
    df = pd.DataFrame(rows, columns=headers)
    
    # Game ID normalizálás (egy meccsnek 2 sora van, home és away)
    df['GAME_ID'] = df['GAME_ID'].astype(str)

    return df


# Play by play
from nba_api.stats.endpoints import playbyplayv3
import pandas as pd
def get_game_snapshots(game_id):
    plays_log = playbyplayv3.PlayByPlayV3(game_id=game_id, 
                            start_period='1', 
                            end_period='4')
    plays = plays_log.get_dict()

    actions = plays['game']['actions']
    #for act in actions:
    #    print(act['actionType'], act['clock'])
    #print(len(actions))
    df = pd.DataFrame(actions)

    df['time_remaining_period'] = df['clock'].apply(parse_clock)

    return df

# League Dash Stats (pre-game)
from nba_api.stats.endpoints import leaguedashteamstats
def get_team_dash_stats(season='2024-25', date_from=None, date_to='2025-02-05'):
    ldts_raw = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        date_from_nullable=date_from,
        date_to_nullable=date_to
    )

    ldts = ldts_raw.get_dict()

    headers = ldts['resultSets'][0]['headers']
    rows = ldts['resultSets'][0]['rowSet']

    # DataFrame készítése
    df = pd.DataFrame(rows, columns=headers)

    return df

# Inactive players
from nba_api.stats.endpoints import boxscoresummaryv2
def get_inactive_players(game_id):
    bs_raw = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
    bs = bs_raw.get_dict()

    for x in bs['resultSets']:
        if x['name'] == 'InactivePlayers':
            print(x)

            headers = x['headers']
            rows = x['rowSet']
            
            # DataFrame készítése
            df = pd.DataFrame(rows, columns=headers)
            
            return df
        
    return pd.DataFrame()

# Clock helper
def parse_clock(clock_str):
    """PT12M34S -> 754 másodperc"""
    if not clock_str or clock_str == '':
        return 0
    
    clock_str = clock_str.replace('PT', '').replace('S', '')
    parts = clock_str.split('M')
    
    if len(parts) == 2:
        minutes = int(parts[0]) if parts[0] else 0
        seconds = int(float(parts[1])) if parts[1] else 0
        return minutes * 60 + seconds
    return 0

# pbp data
import pandas as pd
def extract_snapshots(df, snapshot_times=[(1, 180), 
                                          (2, 360), (2, 0), 
                                          (3, 540), (3, 360), (3, 180), (3, 0),
                                          (4, 540), (4, 450), (4, 360), (4, 270), (4, 180), (4, 90), (4, 0)]):
    """
    Kivonatolja a kívánt pillanatokat a play-by-play dataframe-ből.
    
    Parameters:
    - df: pd.DataFrame, a playbyplayv3 hívás eredménye
    - snapshot_times: list of tuples, pl. [(4, 180), (3, 360), (3, 0)]
      ahol a tuple = (period, time_remaining_sec)
    
    Returns:
    - snapshots: list of dict, minden snapshot egy dictionary a kért feature-ökkel
    """
    snapshots = []
    
    # Előfeldolgozás
    df = df.copy()
    df['time_remaining_period'] = df['time_remaining_period'].astype(float)
    df['time_remaining_total'] = 720*4 - ((df['period']-1)*720 + (720-df['time_remaining_period']))
    
    # Csak scoring action-öket tartsunk meg a score trackinghez
    df_with_score = df[(df['scoreHome'].notna()) & (df['scoreAway'].notna()) &
                       (df['scoreHome'] != '') & (df['scoreAway'] != '') &
                       (df['actionType'] != 'period')].copy()
    
    df_with_score['scoreHome'] = df_with_score['scoreHome'].astype(int)
    df_with_score['scoreAway'] = df_with_score['scoreAway'].astype(int)
    
    for period, t_rem in snapshot_times:
        print(period, t_rem/60)
        t_rem_total = 720*4 - ((period-1)*720 + (720 - t_rem))
        # Legközelebbi action a kért időpontban (scoring action)
        mask = df_with_score['time_remaining_total'] >= t_rem_total
        
        if not mask.any():
            continue
            
        snapshot_row = df_with_score[mask].iloc[-1]
        
        current_home = snapshot_row['scoreHome']
        current_away = snapshot_row['scoreAway']
        
        # 6 perccel ezelőtti score megkeresése
        lookback_time = t_rem_total + 360
        lookback_mask = df_with_score['time_remaining_total'] >= lookback_time

        if lookback_mask.any():
            lookback_row = df_with_score[lookback_mask].iloc[-1]
            lookback_home = lookback_row['scoreHome']
            lookback_away = lookback_row['scoreAway']
        else:
            # Ha nincs 5 perccel ezelőtti adat (negyed eleje), 0-ról indulunk
            lookback_home = 0
            lookback_away = 0

        # 4 perccel ezelőtti score
        lookback_4min = t_rem_total + 240
        lookback_4min_mask = df_with_score['time_remaining_total'] >= lookback_4min
        
        if lookback_4min_mask.any():
            lookback_4min_row = df_with_score[lookback_4min_mask].iloc[-1]
            lookback_4min_home = lookback_4min_row['scoreHome']
            lookback_4min_away = lookback_4min_row['scoreAway']
        else:
            lookback_4min_home = 0
            lookback_4min_away = 0
        
        # 2 perccel ezelőtti score
        lookback_2min = t_rem_total + 120
        lookback_2min_mask = df_with_score['time_remaining_total'] >= lookback_2min
        
        if lookback_2min_mask.any():
            lookback_2min_row = df_with_score[lookback_2min_mask].iloc[-1]
            lookback_2min_home = lookback_2min_row['scoreHome']
            lookback_2min_away = lookback_2min_row['scoreAway']
        else:
            lookback_2min_home = 0
            lookback_2min_away = 0

        # ===== CSAPATOK AZONOSÍTÁSA =====
        # Első nem 0-0 score-nál nézd meg melyik csapat szerzett pontot
        home_team_tricode = None
        away_team_tricode = None

        for _, row in df_with_score.iterrows():
            if row['scoreHome'] > 0 and home_team_tricode is None:
                # Ez a csapat szerzett először home pontot
                home_team_tricode = row['teamTricode']
                home_team_id = row['teamId']
            if row['scoreAway'] > 0 and away_team_tricode is None:
                away_team_tricode = row['teamTricode']
                away_team_id = row['teamId']
            if home_team_tricode and away_team_tricode:
                break

        # ===== RUN SZÁMÍTÁS (pontsorozat) =====
        # Visszafelé haladva a snapshot-tól, hány pont lett szerzve egymás után
        run_home = 0
        run_away = 0
        last_scorer = None

        for _, row in df_with_score[df_with_score['time_remaining_total'] >= t_rem_total].iloc[::-1].iterrows():
            if pd.notna(row['pointsTotal']) and row['pointsTotal'] > 0:
                scorer = row['teamTricode']
                
                if last_scorer is None:
                    last_scorer = scorer
                
                if (row['shotResult'] == 'Made') & (scorer == last_scorer):
                    if scorer == home_team_tricode:
                        run_home = int(current_home) - row['scoreHome']
                    else:
                        run_away = int(current_away) - row['scoreAway']
                else:
                    break  # Másik csapat megszakította a sorozatot
        run = run_home - run_away
                
        # ===== PACE (possessions / elapsed time) =====
        # Possession számolás: FGA + TO + 0.44*FTA
        elapsed_time = (720 * 4) - t_rem_total  # másodperc
        
        possession_events = df[
            (df['time_remaining_total'] >= t_rem_total) & 
            ((df['actionType'] == 'Made Shot') | 
            (df['actionType'] == 'Free Throw') |
            (df['actionType'] == 'Missed Shot')|
            (df['actionType'] == 'Turnover'))
        ].copy()

        # FGA: missed + made shots
        fga_count = len(possession_events[(possession_events['actionType'] == 'Made Shot') | 
                                        (possession_events['actionType'] == 'Missed Shot')])

        # TO: turnovers
        to_count = len(possession_events[possession_events['actionType'] == 'Turnover'])

        # FTA: free throws (0.44 szorzó)
        fta_count = len(possession_events[possession_events['actionType'] == 'Free Throw']) * 0.44

        total_possessions = fga_count + to_count + fta_count

        pace_live = (total_possessions / (elapsed_time / 60)) if elapsed_time > 0 else 0  # possessions/perc


        # ===== JÁTÉKOS STATISZTIKÁK =====
        # Top scorer eddig a meccsen (snapshot időpontig)
        player_points_home = df[
            (df['time_remaining_total'] >= t_rem_total) & 
            (df['teamTricode'] == home_team_tricode) & 
            (df['shotValue'].notna()) &
            (df['shotResult'] == 'Made')
        ].groupby('playerName')['shotValue'].sum()

        player_ftm_home = df[
            (df['time_remaining_total'] >= t_rem_total) & 
            (df['teamTricode'] == home_team_tricode) & 
            (df['actionType'] == 'Free Throw') &
            (~df['description'].str.contains('MISS'))
        ].groupby('playerName').size()

        player_points_away = df[
            (df['time_remaining_total'] >= t_rem_total) & 
            (df['teamTricode'] == away_team_tricode) & 
            (df['shotValue'].notna()) &
            (df['shotResult'] == 'Made')
        ].groupby('playerName')['shotValue'].sum()

        player_ftm_away = df[
            (df['time_remaining_total'] >= t_rem_total) & 
            (df['teamTricode'] == away_team_tricode) & 
            (df['actionType'] == 'Free Throw') &
            (~df['description'].str.contains('MISS'))
        ].groupby('playerName').size()

        player_total_home = pd.merge(player_points_home.to_frame(), player_ftm_home.to_frame(), 
                                     how='outer', left_index=True, right_index=True)
        player_total_home.fillna(0, inplace=True)
        player_total_home['total'] = player_total_home.iloc[:,0] + player_total_home.iloc[:,1]
        
        player_total_away = pd.merge(player_points_away.to_frame(), player_ftm_away.to_frame(), 
                                     how='outer', left_index=True, right_index=True)
        player_total_away.fillna(0, inplace=True)
        player_total_away['total'] = player_total_away.iloc[:,0] + player_total_away.iloc[:,1]

        home_top_player_points_live = int(player_total_home['total'].max()) if len(player_total_home) > 0 else 0
        away_top_player_points_live = int(player_total_away['total'].max()) if len(player_total_away) > 0 else 0


        # ===== FOUL STATISZTIKÁK =====
        # Összes fault a csapatoknak
        home_foul_count = len(df[
            (df['time_remaining_total'] >= t_rem_total) & 
            (df['teamTricode'] == home_team_tricode) & 
            (df['actionType'] == 'Foul')
        ])

        away_foul_count = len(df[
            (df['time_remaining_total'] >= t_rem_total) & 
            (df['teamTricode'] == away_team_tricode) & 
            (df['actionType'] == 'Foul')
        ])

        # Foul trouble: játékosok akiknek ≥4 faultja van
        home_player_fouls = df[
            (df['time_remaining_total'] >= t_rem_total) & 
            (df['teamTricode'] == home_team_tricode) & 
            (df['actionType'] == 'Foul')
        ].groupby('playerName').size()

        away_player_fouls = df[
            (df['time_remaining_total'] >= t_rem_total) & 
            (df['teamTricode'] == away_team_tricode) & 
            (df['actionType'] == 'Foul')
        ].groupby('playerName').size()

        home_foul_trouble_players = len(home_player_fouls[home_player_fouls >= 4])
        away_foul_trouble_players = len(away_player_fouls[away_player_fouls >= 4])

        # Snapshot dictionary összeállítása
        snap_dict = {
            'period': int(period),
            'time_remaining_period': int(t_rem),
            
            # Aktuális score
            'home_score': int(current_home),
            'away_score': int(current_away),
            'score_diff': int(current_home - current_away),
            
            # Momentum: utolsó 6, 4 és 2 perc pontjai
            'home_points_last_6min': max(0, int(current_home - lookback_home)),
            'away_points_last_6min': max(0, int(current_away - lookback_away)),
            'momentum_diff6': int((current_home - lookback_home) - (current_away - lookback_away)),

            'home_points_last_4min': max(0, int(current_home - lookback_4min_home)),
            'away_points_last_4min': max(0, int(current_away - lookback_4min_away)),
            'momentum_diff4': int((current_home - lookback_4min_home) - (current_away - lookback_4min_away)),
            
            'home_points_last_2min': max(0, int(current_home - lookback_2min_home)),
            'away_points_last_2min': max(0, int(current_away - lookback_2min_away)),
            'momentum_diff2': int((current_home - lookback_2min_home) - (current_away - lookback_2min_away)),

            'run': run, # + if run for home
            'pace_live': round(pace_live, 2),
            'home_top_player_points_live': home_top_player_points_live,
            'away_top_player_points_live': away_top_player_points_live,
            'home_foul_count': home_foul_count,
            'away_foul_count': away_foul_count,
            'home_foul_trouble_players': home_foul_trouble_players,
            'away_foul_trouble_players': away_foul_trouble_players,

        }
        
        snapshots.append(snap_dict)
    
    return snapshots

def extract_pregame(season, game_date, team_ids):
    pregame = {}

    all_season_pg = get_team_dash_stats(season=season, date_from=None, date_to=game_date)
    team_ids = [int(team_id) for team_id in team_ids]
    row_home = all_season_pg[all_season_pg['TEAM_ID'] == team_ids[0]].iloc[0]
    row_away = all_season_pg[all_season_pg['TEAM_ID'] == team_ids[1]].iloc[0]

    poss_home = row_home['FGA'] + 0.4*row_home['FTA'] + 1.07*row_home['OREB']+ row_home['TOV']
    poss_away = row_away['FGA'] + 0.4*row_away['FTA'] + 1.07*row_away['OREB']+ row_away['TOV']

    pregame['home_ORtg'] = row_home['PTS'] / poss_home * 100
    pregame['away_ORtg'] = row_away['PTS'] / poss_away * 100
    pregame['home_DRtg'] = (row_home['PTS'] - row_home['PLUS_MINUS']) / poss_home * 100
    pregame['away_DRtg'] = (row_away['PTS'] - row_away['PLUS_MINUS']) / poss_away * 100
    pregame['home_NET_rtg'] = row_home['PLUS_MINUS'] / poss_home * 100
    pregame['away_NET_rtg'] = row_away['PLUS_MINUS'] / poss_away * 100
    pregame['home_PACE'] = poss_home / row_home['GP']
    pregame['away_PACE'] = poss_away / row_away['GP']
    pregame['home_TS%'] = row_home['PTS'] / (2 * (row_home['FGA'] + 0.44 * row_home['FTA']))
    pregame['away_TS%'] = row_away['PTS'] / (2 * (row_away['FGA'] + 0.44 * row_away['FTA']))
    pregame['home_AST_ratio'] = row_home['AST'] / row_home['FGM']
    pregame['away_AST_ratio'] = row_away['AST'] / row_away['FGM']
    pregame['home_EFG%'] = (row_home['FGM'] + 0.5*row_home['FG3M']) / row_home['FGA']
    pregame['away_EFG%'] = (row_away['FGM'] + 0.5*row_away['FG3M']) / row_away['FGA']
    pregame['home_OREB%'] = row_home['OREB'] / row_home['REB']
    pregame['away_OREB%'] = row_away['OREB'] / row_away['REB']
    pregame['home_turnover_ratio'] = row_home['TOV'] / poss_home * 100
    pregame['away_turnover_ratio'] = row_away['TOV'] / poss_away * 100

    return pregame

#pbp_df = get_game_snapshots('0022400724')
#snapshots = extract_snapshots(pbp_df)
#print(snapshots[-1])

print(extract_pregame('2024-25', game_date='2025-02-05', team_ids=['1610612760', '1610612756']))
print(get_inactive_players('0022400724'))