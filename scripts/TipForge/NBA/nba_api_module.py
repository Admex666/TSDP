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

# Game meta data
from nba_api.stats.endpoints import boxscoretraditionalv3
from datetime import datetime
def extract_game_meta(game_id):
    bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    bs_d = bs.get_dict()

    bst = bs_d['boxScoreTraditional']

    if game_id.startswith('002'):
        season_start = int(game_id[3:5])
        season = f"20{season_start}-{(season_start+1)}"
    else:
        season = "0000"

    meta = {
        "date": bs_d['meta']['time'].split('T')[0],
        "season": season,
        "home_name": bst['homeTeam']['teamName'],
        "home_id": bst['homeTeam']['teamId'],
        "away_name": bst['awayTeam']['teamName'],
        "away_id": bst['awayTeam']['teamId'],
    }

    return meta

# Season game log
from nba_api.stats.endpoints import leaguegamelog
def get_season_game_log(season='2025-26'):
    log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
    games = log.get_dict()

    #print(f"Total games found: {len(games['resultSets'][0]['rowSet'])}")
    
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
def get_inactive_players(game_id):
    from nba_api.stats.endpoints import boxscoresummaryv3

    bs_raw = boxscoresummaryv3.BoxScoreSummaryV3(game_id=TEST_GAME_ID)
    bs = bs_raw.get_dict()

    home_id = bs['boxScoreSummary']['homeTeam']['teamId']
    away_id = bs['boxScoreSummary']['awayTeam']['teamId']

    home_inactives = bs['boxScoreSummary']['homeTeam']['inactives']
    away_inactives = bs['boxScoreSummary']['awayTeam']['inactives']

    home_inactives = [{**player, 'TEAM_ID': home_id} for player in home_inactives]
    away_inactives = [{**player, 'TEAM_ID': away_id} for player in away_inactives]


    df = pd.DataFrame(home_inactives + away_inactives)

    return df

# Dash starters
from nba_api.stats.endpoints import leaguedashlineups
def get_dash_lineup(team_id, season, date_to=None):
    ldl = leaguedashlineups.LeagueDashLineups(season=season,
                                            date_to_nullable=date_to
                                            )
    df_ldl = ldl.get_data_frames()[0]
    team_row = df_ldl[df_ldl['TEAM_ID'] == team_id].iloc[0]
    
    group_id_str = team_row['GROUP_ID']
    if group_id_str.startswith('-'):
        group_id_str = group_id_str[1:]
    if group_id_str.endswith('-'):
        group_id_str = group_id_str[:-1]

    team_starters = group_id_str.split('-')

    return team_starters

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

def extract_pregame(game_id):
    meta = extract_game_meta(game_id)
    season, game_date, team_ids = meta['season'], meta['date'], [meta['home_id'], meta['away_id']]
    
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

def extract_injury(game_id):
    meta = extract_game_meta(game_id)
    season, game_date, team_ids = meta['season'], meta['date'], [meta['home_id'], meta['away_id']]
    injuries = {}

    inap = get_inactive_players(game_id)

    injuries['home_injury_count'] = len(inap[inap.TEAM_ID == team_ids[0]])
    injuries['away_injury_count'] = len(inap[inap.TEAM_ID == team_ids[1]])

    team_ids = [int(team_id) for team_id in team_ids]

    def calc_inap_starters(team_id):
        inap_team = inap[inap.TEAM_ID == team_id].personId.tolist()
        starters = get_dash_lineup(team_id, season=season, date_to=game_date)

        inap_strt = 0
        for id_ina in inap_team:
            if id_ina in starters:
                inap_strt += 1

        return inap_strt
    
    injuries['home_missing_starters'] = calc_inap_starters(team_ids[0])
    injuries['away_missing_starters'] = calc_inap_starters(team_ids[1])

    return injuries

def extract_advanced_stats(game_id):
    from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog
    import pandas as pd
    import numpy as np

    meta = extract_game_meta(game_id)
    game_date = meta['date']

    home_id, away_id = [meta['home_id'], meta['away_id']]
    home_starters = get_dash_lineup(team_id=meta['home_id'], season=meta['season'], date_to=game_date)
    away_starters = get_dash_lineup(team_id=meta['away_id'], season=meta['season'], date_to=game_date)

    # 1) Lekérjük a liga összes játékosát adott napig
    stats_raw = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2024-25",
        season_type_all_star="Regular Season",
        date_to_nullable=game_date
    ).get_dict()

    headers = stats_raw["resultSets"][0]["headers"]
    rows = stats_raw["resultSets"][0]["rowSet"]
    stats_df = pd.DataFrame(rows, columns=headers)

    # csak számokká alakítjuk, ahol kell
    numeric_cols = [
        "MIN","PTS","REB","AST","STL","BLK","TOV","FGA","FGM",
        "FTA","FTM","TS_PCT","USG_PCT"
    ]
    for c in numeric_cols:
        if c in stats_df.columns:
            stats_df[c] = pd.to_numeric(stats_df[c], errors="coerce")

    # 2) PER kiszámolása a dummy képlettel
    def compute_per(row):
        MIN = max(1, row.get("MIN", 1))  # elkerüljük a nullával való osztást
        FGM = row.get("FGM", 0)
        FGA = row.get("FGA", 0)
        FG3M = row.get("FG3M", 0)
        FTM = row.get("FTM", 0)
        FTA = row.get("FTA", 0)
        OREB = row.get("OREB", 0)
        DREB = row.get("DREB", 0)
        AST = row.get("AST", 0)
        STL = row.get("STL", 0)
        BLK = row.get("BLK", 0)
        TOV = row.get("TOV", 0)
        PF = row.get("PF", 0)
        
        # uPER egyszerűsített formula
        uPER = (
            FGM + 0.5*FG3M - FGA + 0.5*FTM - FTA
            + OREB + 0.7*DREB + 0.7*AST + 0.7*STL + 0.7*BLK
            - 0.7*TOV - 0.5*PF
        )
        
        per = uPER / MIN * 15  # 15-ös szorzóval normáljuk
        return per
    
    stats_df["PER"] = stats_df.apply(compute_per, axis=1)

    # 3) starter / bench kategória
    stats_df["IS_STARTER"] = stats_df["PLAYER_ID"].astype(str).apply(
        lambda pid:
            1 if pid in home_starters + away_starters else 0
    )

    def compute_ts(row):
        fga = row.get("FGA", 0)
        fta = row.get("FTA", 0)
        pts = row.get("PTS", 0)
        denom = 2 * (fga + 0.44 * fta)
        if denom > 0:
            return float(pts / denom)
        else:
            return 0.0

    stats_df["TS_PCT"] = stats_df.apply(compute_ts, axis=1)

    # 4) Szétbontjuk csapatonként
    def team_split(team_id, starter_list):
        team = stats_df[stats_df["TEAM_ID"] == team_id].copy()

        team["STARTER"] = team["PLAYER_ID"].astype(str).apply(
            lambda pid: pid in starter_list
        )

        starters_df = team[team["STARTER"] == True]
        bench_df = team[team["STARTER"] == False]

        return starters_df, bench_df


    home_st_df, home_bench_df = team_split(home_id, home_starters)
    away_st_df, away_bench_df = team_split(away_id, away_starters)


    # ---- számítások ----

    def avg_or_zero(series):
        return float(series.mean()) if len(series) > 0 else 0.0

    # PER
    home_starter_avg_PER = avg_or_zero(home_st_df["PER"])
    away_starter_avg_PER = avg_or_zero(away_st_df["PER"])
    home_bench_avg_PER   = avg_or_zero(home_bench_df["PER"])
    away_bench_avg_PER   = avg_or_zero(away_bench_df["PER"])

    # USG_PCT helyett approximált star_usage: starter PTS / összes csapat PTS
    def star_usage_approx(team_df, starter_df):
        team_pts = team_df["PTS"].sum()
        starter_pts = starter_df["PTS"].sum()
        if team_pts > 0:
            return float(starter_pts / team_pts)
        else:
            return 0.0

    home_star_usage = star_usage_approx(stats_df[stats_df["TEAM_ID"] == home_id], home_st_df)
    away_star_usage = star_usage_approx(stats_df[stats_df["TEAM_ID"] == away_id], away_st_df)


    # TS_PCT
    home_avg_TS = avg_or_zero(home_st_df["TS_PCT"])
    away_avg_TS = avg_or_zero(away_st_df["TS_PCT"])

    # Top3 scorer az adott dátumig (PTS átlag alapján)
    def top3_avg_pts(team_df):
        top3 = team_df.sort_values("PTS", ascending=False).head(3)
        return avg_or_zero(top3["PTS"]/top3["GP"])

    home_top3_points_avg = top3_avg_pts(stats_df[stats_df["TEAM_ID"] == home_id])
    away_top3_points_avg = top3_avg_pts(stats_df[stats_df["TEAM_ID"] == away_id])


    # ---- eredmény ----
    return {
        "home_starter_avg_PER": home_starter_avg_PER,
        "away_starter_avg_PER": away_starter_avg_PER,
        "home_bench_avg_PER":   home_bench_avg_PER,
        "away_bench_avg_PER":   away_bench_avg_PER,

        "home_star_usage": home_star_usage,
        "away_star_usage": away_star_usage,

        "home_avg_TS": home_avg_TS,
        "away_avg_TS": away_avg_TS,

        "home_top3_points_avg": home_top3_points_avg,
        "away_top3_points_avg": away_top3_points_avg
    }

def extract_form(game_id):
    form = {}

    meta = extract_game_meta(game_id)
    season = meta['season']

    def compute_team_form_metrics(gamelog, team_id, date):
        """
        gamelog: LeagueGameLog DF
        team_id: int
        date: 'YYYY-MM-DD' or datetime
        """
        import pandas as pd

        date = pd.to_datetime(date)

        # Csak a csapat meccsei, és csak az adott dátum előttiek
        df = gamelog[gamelog["TEAM_ID"] == team_id].copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df[df["GAME_DATE"] < date].sort_values("GAME_DATE")

        # Ha nincs korábbi meccs → minden metrika 0 / None
        if df.empty:
            return {
                "is_back_to_back": False,
                "rest_days": None,
                "recent_form10": None,
                "recent_form5": None,
                "recent_form3": None,
            }

        # --- GAME_DAY ---
        last_game_date = df["GAME_DATE"].iloc[-1]

        # --- REST DAYS ---
        rest_days = (date - last_game_date).days

        # --- BACK TO BACK ---
        # az előző meccs és az azt megelőző közti távolság
        if len(df) >= 2:
            prev_prev_date = df["GAME_DATE"].iloc[-2]
            diff = (last_game_date - prev_prev_date).days
            is_back_to_back = diff == 1
        else:
            is_back_to_back = False

        # --- RECENT FORM WINRATE ---
        def winrate(n):
            sub = df.tail(n)
            if sub.empty:
                return None
            return (sub["WL"] == "W").mean()

        recent_form10 = winrate(10)
        recent_form5 = winrate(5)
        recent_form3 = winrate(3)

        return {
            "is_back_to_back": is_back_to_back,
            "rest_days": rest_days,
            "recent_form10": recent_form10,
            "recent_form5": recent_form5,
            "recent_form3": recent_form3,
        }
    
    gamelog = get_season_game_log(season=season)

    form_home = compute_team_form_metrics(gamelog, meta['home_id'], meta['date'])
    form_away = compute_team_form_metrics(gamelog, meta['away_id'], meta['date'])
    form = {
        "home_is_back_to_back": form_home['is_back_to_back'],
        "home_rest_days": form_home['rest_days'],
        "home_recent_form10": form_home['recent_form10'],
        "home_recent_form5": form_home['recent_form5'],
        "home_recent_form3": form_home['recent_form3'],
        "away_is_back_to_back": form_away['is_back_to_back'],
        "away_rest_days": form_away['rest_days'],
        "away_recent_form10": form_away['recent_form10'],
        "away_recent_form5": form_away['recent_form5'],
        "away_recent_form3": form_away['recent_form3'],
    }

    return form

TEST_GAME_ID = '0022400724'
    
#pbp_df = get_game_snapshots(TEST_GAME_ID)
#snapshots = extract_snapshots(pbp_df)
#print(snapshots[-1])

#print(get_inactive_players(TEST_GAME_ID))

import time
import pandas as pd

def create_ml_row(game_id):

    timings = {}

    # --- Pre-game ---
    t0 = time.time()
    d_pregame = extract_pregame(game_id)
    timings["pregame"] = time.time() - t0
    time.sleep(1.0)

    # --- Injury ---
    t0 = time.time()
    d_injury = extract_injury(game_id)
    timings["injury"] = time.time() - t0
    time.sleep(1.0)

    # --- Advanced Stats ---
    t0 = time.time()
    d_advanced = extract_advanced_stats(game_id)
    timings["advanced"] = time.time() - t0
    time.sleep(1.0)

    # --- Form ---
    t0 = time.time()
    d_form = extract_form(game_id)
    timings["form"] = time.time() - t0
    time.sleep(1.0)

    # --- Print total time ---
    total = sum(timings.values())

    print("\n--- create_ml_row TIMINGS ---")
    for k, v in timings.items():
        print(f"{k:10s}: {v:6.3f} sec")
    print(f"TOTAL      : {total:6.3f} sec")
    print("--------------------------------\n")

    # --- Return final row ---
    return pd.Series({'game_id': game_id, **d_pregame, **d_injury, **d_advanced, **d_form})
