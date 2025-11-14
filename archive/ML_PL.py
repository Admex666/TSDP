# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:11:16 2024

@author: Adam
"""
#%%
import pandas as pd

df = pd.read_csv(r'C:\TwitterSportsDataProject\Random data\PL ML model\PremierLeague.csv')

#%% Prepare dataframe
df.info()

# only need stats, no odds
df = df.iloc[:,:25]
df.drop('Time', axis=1, inplace=True)

df.dropna(inplace=True)
df = df.reset_index(drop=True)

# Select specific season
season = '2023-2024'
df_season = df.loc[df.Season==season,:].reset_index(drop=True)

df_ht_mavg_big = pd.DataFrame({})

#%% Prepare table
# What do we need as columns?
# Goals, Shots, Shots OT, Corners, Fouls, Yellows, FullTimeResult(label)
df_season['HomeTeam'].unique()

for home_team_name in df_season['HomeTeam'].unique():
    df_home_team = df_season.loc[(df_season['HomeTeam'] == home_team_name) | (df_season['AwayTeam'] == home_team_name), :]
    df_home_team = df_home_team.sort_values('Date').reset_index(drop=True)
    df_home_team['MatchNr'] = range(1,len(df_home_team)+1)
    
    df_home_team_sum = df_away_team_sum = pd.DataFrame({
                                     'MatchNr': range(1,len(df_home_team)+1),
                                     'Goals':0,
                                     'Shots':0,
                                     'SOT':0,
                                     'Corners':0,
                                     'Fouls':0,
                                     'Yellows':0,
                                     'Goals_AG':0,
                                     'Shots_AG':0,
                                     'SOT_AG':0,
                                     'Corners_AG':0,
                                     'Fouls_AG':0,
                                     'Yellows_AG':0,
                                     'Result': 0})
    
    df_away_team = df_home_team.copy()
    for matchnumber in range(1,39):
        if df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeam.iloc[0] == home_team_name:
            away_team_name = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeam.iloc[0]
            df_away_team = df_season.loc[(df_season['HomeTeam'] == away_team_name) | (df_season['AwayTeam'] == away_team_name), :]
            df_away_team = df_away_team.sort_values('Date').reset_index(drop=True)
            df_away_team['MatchNr'] = range(1,len(df_away_team)+1)
            
            # Team stats
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Goals'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].FullTimeHomeTeamGoals.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Shots'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeamShots.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'SOT'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeamShotsOnTarget.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Corners'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeamCorners.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Fouls'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeamFouls.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Yellows'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeamYellowCards.iloc[0]
            
            # Against stats
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Goals_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].FullTimeAwayTeamGoals.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Shots_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeamShots.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'SOT_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeamShotsOnTarget.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Corners_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeamCorners.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Fouls_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeamFouls.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Yellows_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeamYellowCards.iloc[0]
            
            # Away team stats
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Goals'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].FullTimeAwayTeamGoals.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Shots'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].AwayTeamShots.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'SOT'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].AwayTeamShotsOnTarget.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Corners'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].AwayTeamCorners.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Fouls'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].AwayTeamFouls.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Yellows'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].AwayTeamYellowCards.iloc[0]
            # Against stats
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Goals_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].FullTimeHomeTeamGoals.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Shots_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].HomeTeamShots.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'SOT_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].HomeTeamShotsOnTarget.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Corners_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].HomeTeamCorners.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Fouls_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].HomeTeamFouls.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Yellows_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].HomeTeamYellowCards.iloc[0]
            
            # Result
            if df_home_team.loc[df_home_team.MatchNr == matchnumber].FullTimeResult.iloc[0] == 'H':
                df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Result'] = 'W'
            elif df_home_team.loc[df_home_team.MatchNr == matchnumber].FullTimeResult.iloc[0] == 'D':
                df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Result'] = 'D'
            elif df_home_team.loc[df_home_team.MatchNr == matchnumber].FullTimeResult.iloc[0] == 'A':
                df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Result'] = 'L'
             
                
        elif df_home_team.loc[df_home_team.MatchNr == matchnumber, 'AwayTeam'].iloc[0] == home_team_name:
            away_team_name = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeam.iloc[0]
            
            # Team stats
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Goals'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].FullTimeAwayTeamGoals.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Shots'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeamShots.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'SOT'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeamShotsOnTarget.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Corners'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeamCorners.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Fouls'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeamFouls.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Yellows'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].AwayTeamYellowCards.iloc[0]
            
            # Against stats
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Goals_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].FullTimeHomeTeamGoals.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Shots_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeamShots.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'SOT_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeamShotsOnTarget.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Corners_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeamCorners.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Fouls_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeamFouls.iloc[0]
            df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Yellows_AG'] = df_home_team.loc[df_home_team.MatchNr == matchnumber].HomeTeamYellowCards.iloc[0]
            
            # Away team stats
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Goals'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].FullTimeHomeTeamGoals.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Shots'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].HomeTeamShots.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'SOT'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].HomeTeamShotsOnTarget.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Corners'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].HomeTeamCorners.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Fouls'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].HomeTeamFouls.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Yellows'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].HomeTeamYellowCards.iloc[0]
            # Against stats
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Goals_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].FullTimeAwayTeamGoals.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Shots_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].AwayTeamShots.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'SOT_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].AwayTeamShotsOnTarget.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Corners_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].AwayTeamCorners.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Fouls_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].AwayTeamFouls.iloc[0]
            df_away_team_sum.loc[df_away_team_sum.MatchNr==matchnumber, 'Yellows_AG'] = df_away_team.loc[df_away_team.MatchNr == matchnumber].AwayTeamYellowCards.iloc[0]
            
            # Result
            if df_home_team.loc[df_home_team.MatchNr == matchnumber].FullTimeResult.iloc[0] == 'H':
                df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Result'] = 'L'
            elif df_home_team.loc[df_home_team.MatchNr == matchnumber].FullTimeResult.iloc[0] == 'D':
                df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Result'] = 'D'
            elif df_home_team.loc[df_home_team.MatchNr == matchnumber].FullTimeResult.iloc[0] == 'A':
                df_home_team_sum.loc[df_home_team_sum.MatchNr==matchnumber, 'Result'] = 'W'
    
    df_teams_sum = pd.merge(df_home_team_sum.iloc[:,:-1],
                            df_away_team_sum,
                            on='MatchNr')
    
    # Creating moving averages dataframe
    df_ht_mavg = df_teams_sum.iloc[3:,:].reset_index(drop=True).copy()
    df_ht_mavg.iloc[:,1:(len(df_ht_mavg.columns)-1)] = 0

    for x in range(len(df_ht_mavg)):
        for col in range(1,(len(df_ht_mavg.columns)-1)):
            for n in range(3):
                df_ht_mavg.iloc[x,col] += df_teams_sum.iloc[x+n,col] / 3

    df_ht_mavg_big = pd.concat([df_ht_mavg_big,df_ht_mavg], ignore_index=True)
    
df_ht_mavg_big = df_ht_mavg_big.iloc[:,1:]
#%% To Excel
df_ht_mavg_big.to_excel(r'C:\TwitterSportsDataProject\Random data\PL ML model\result_ML.xlsx', index=False)

#%% no draws data, to excel
df_ht_mavg_big_no_draws = df_ht_mavg_big.loc[df_ht_mavg_big.Result!='D',:]
df_ht_mavg_big_no_draws.to_excel(r'C:\TwitterSportsDataProject\Random data\PL ML model\result_ML.xlsx', index=False)

#%% Get current team URLs in Prem
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Premier League 2024-2025 overview page
league = 'ENG'
if league == 'ENG':
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    table_id = 'results2024-202591_overall'
    comp_id = '9'
    league_name = 'Premier-League'
elif league == 'ESP':
    url = "https://fbref.com/en/comps/12/La-Liga-Stats"
    table_id = 'results2024-2025121_overall'
    comp_id = '12'
    league_name = 'La-Liga'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table with team links
team_table = soup.find('table', {'id': table_id})
team_links = team_table.find_all('a')

# Extract team names and URLs
teams = []
base_url = "https://fbref.com"
for link in team_links:
    if "/squads/" in link['href']:  # Filter out team links
        teams.append({
            'team': link.text,
            'url': base_url + link['href']
        })

# Convert to DataFrame for easier handling
teams_df = pd.DataFrame(teams)

# Function to format team names for the URL
def format_team_name(name):
    return name.replace(" ", "-")  # Replace spaces with dashes

# Add columns for team ID and formatted name
teams_df['team_id'] = teams_df['url'].str.extract(r'/squads/([^/]+)/')  # Extract team ID from URL
teams_df['formatted_name'] = teams_df['team'].apply(format_team_name)  # Format team name

# Construct the match log URL
season = "2024-2025"
teams_df['match_log_url_shot'] = (
    "https://fbref.com/en/squads/" 
    + teams_df['team_id'] 
    + f"/{season}/matchlogs/c"+comp_id+"/shooting/" 
    + teams_df['formatted_name'] 
    + "-Match-Logs-"+league_name
)

teams_df['match_log_url_pass'] = (
    "https://fbref.com/en/squads/" 
    + teams_df['team_id'] 
    + f"/{season}/matchlogs/c"+comp_id+"/passing_types/" 
    + teams_df['formatted_name'] 
    + "-Match-Logs-"+league_name
)

teams_df['match_log_url_misc'] = (
    "https://fbref.com/en/squads/" 
    + teams_df['team_id'] 
    + f"/{season}/matchlogs/c"+comp_id+"/misc/" 
    + teams_df['formatted_name'] 
    + "-Match-Logs-"+league_name
)

del base_url, comp_id, league, season, table_id, teams, url, response, team_links

#%% Preparing df and functions for scrape
import fbref_module as fbref
from requests.exceptions import HTTPError
import numpy as np

def column_joiner(df):
    if type(df.columns.values[0]) != str:
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        return df
    else:
        return df

def scrape(URL, table_id):       
    df = fbref.read_html(URL, table_id)
    df = df[0]
    column_joiner(df)
    df = df.iloc[:-1,:]
    return df

df_rt_avg_big = pd.DataFrame({'team_name':[],'Goals':[],'Shots':[],'SOT':[],'Corners':[],'Fouls':[],'Yellows':[],
                          'Goals_AG':[],'Shots_AG':[],'SOT_AG':[],'Corners_AG':[],'Fouls_AG':[],
                          'Yellows_AG':[],'Result':[]})

#%% Scraping the actual data
import time
import random

for teamnr in range(len(teams_df)):
    retry_count = 3  # Number of retries in case of errors
    success = False  # Flag to check if scraping was successful

    while retry_count > 0 and not success:
        try:
            # URLs for the current team
            URL_shot = teams_df.loc[teamnr, 'match_log_url_shot']
            URL_pass = teams_df.loc[teamnr, 'match_log_url_pass']
            URL_misc = teams_df.loc[teamnr, 'match_log_url_misc']
            
            # Scrape data
            df_shot = scrape(URL_shot, 'matchlogs_for')
            df_pass = scrape(URL_pass, 'matchlogs_for')
            df_misc = scrape(URL_misc, 'matchlogs_for')
            df_shot_ag = scrape(URL_shot, 'matchlogs_against')
            df_pass_ag = scrape(URL_pass, 'matchlogs_against')
            df_misc_ag = scrape(URL_misc, 'matchlogs_against')
            
            for dataframe in [df_shot, df_pass, df_misc, df_shot_ag, df_pass_ag, df_misc_ag]:
                for ncol in range(len(dataframe.columns.unique())):
                    if ('For' in dataframe.columns[ncol]) or ('Against' in dataframe.columns[ncol]):
                        col_old_name = dataframe.columns[ncol]
                        col_new_name = dataframe.columns[ncol].split('_')[1]
                        dataframe.rename(columns={col_old_name:col_new_name}, inplace=True)
            
            # Create the DataFrame for team stats
            df_real_team_sum = pd.DataFrame({
                'MatchNr': range(1, 4),
                'Goals': df_shot['Gls'].iloc[-3:].values,
                'Shots': df_shot['Sh'].iloc[-3:].values,
                'SOT': df_shot['SoT'].iloc[-3:].values,
                'Corners': df_pass['Pass Types_CK'].iloc[-3:].values,
                'Fouls': df_misc['Performance_Fls'].iloc[-3:].values,
                'Yellows': df_misc['Performance_CrdY'].iloc[-3:].values,
                'Goals_AG': df_shot_ag['GF'].iloc[-3:].values,
                'Shots_AG': df_shot_ag['Standard_Sh'].iloc[-3:].values,
                'SOT_AG': df_shot_ag['Standard_SoT'].iloc[-3:].values,
                'Corners_AG': df_pass_ag['Pass Types_CK'].iloc[-3:].values,
                'Fouls_AG': df_misc_ag['Performance_Fls'].iloc[-3:].values,
                'Yellows_AG': df_misc_ag['Performance_CrdY'].iloc[-3:].values,
                'Result': 0  # Adjust based on how you calculate the result
            })

            # Calculate averages
            df_rt_avg = pd.DataFrame({
                'team_name': [teams_df.loc[teamnr, 'formatted_name']],
                **{col: np.average(df_real_team_sum[col]) for col in df_real_team_sum.columns[1:]}
            })

            success = True  # Mark as successful

        except HTTPError as e:
            print(f"HTTPError for team {teams_df.loc[teamnr, 'formatted_name']}: {e}")
            retry_count -= 1
            if retry_count > 0:
                delay = random.uniform(5, 10)  # Wait 5-10 seconds before retrying
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to scrape team {teams_df.loc[teamnr, 'formatted_name']} after multiple retries.")
                # Assign default error values
                df_rt_avg = pd.DataFrame({
                    'team_name': [teams_df.loc[teamnr, 'formatted_name']],
                    'Goals': ['Error'],
                    'Shots': ['Error'],
                    'SOT': ['Error'],
                    'Corners': ['Error'],
                    'Fouls': ['Error'],
                    'Yellows': ['Error'],
                    'Goals_AG': ['Error'],
                    'Shots_AG': ['Error'],
                    'SOT_AG': ['Error'],
                    'Corners_AG': ['Error'],
                    'Fouls_AG': ['Error'],
                    'Yellows_AG': ['Error'],
                    'Result': ['Error']
                })

        except Exception as e:
            print(f"Unexpected error for team {teams_df.loc[teamnr, 'formatted_name']}: {e}")
            retry_count = 0  # Skip retries for unknown errors
            df_rt_avg = pd.DataFrame({
                'team_name': [teams_df.loc[teamnr, 'formatted_name']],
                'Goals': ['Error'],
                'Shots': ['Error'],
                'SOT': ['Error'],
                'Corners': ['Error'],
                'Fouls': ['Error'],
                'Yellows': ['Error'],
                'Goals_AG': ['Error'],
                'Shots_AG': ['Error'],
                'SOT_AG': ['Error'],
                'Corners_AG': ['Error'],
                'Fouls_AG': ['Error'],
                'Yellows_AG': ['Error'],
                'Result': ['Error']
            })

    # Append results to the main DataFrame
    df_rt_avg_big = pd.concat([df_rt_avg_big, df_rt_avg], ignore_index=True)

    # Random delay before the next team
    delay = random.uniform(40, 90)  # Wait x seconds between teams
    print(f"Waiting for {delay:.2f} seconds before scraping the next team...")
    time.sleep(delay)

del delay, success, retry_count, teamnr

#%% To prediction excel
matchup_list = [5, 7, 6, 9, 4,
                4, 10, 8, 7, 9,
                5, 8, 10, 2, 2,
                1, 3, 3, 1, 6]
df_rt_avg_big['matchup'] = matchup_list

df_rt_avg_big_formatted = pd.merge(df_rt_avg_big,
                                   df_rt_avg_big,
                                   on='matchup').query("team_name_x != team_name_y")
df_rt_avg_big_formatted.drop_duplicates(subset='matchup', inplace=True)

df_rt_avg_big_formatted.to_excel(r'C:\TwitterSportsDataProject\Random data\PL ML model\result_ML_prediction.xlsx', index=False)

