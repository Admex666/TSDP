"""
FBref scraper module integration
Copied from existing fbref_module.py
"""
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from io import StringIO


def read_html_upd(URL, table_id):
    """
    Read HTML table using Selenium with Chrome
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    driver.get(URL)
    
    try:
        table = WebDriverWait(driver, 40).until(
            EC.presence_of_element_located((By.ID, table_id))
        )
        
        table_html = table.get_attribute('outerHTML')
        df = pd.read_html(StringIO(table_html))[0]
        
        return [df]
    finally:
        driver.quit()


def to_dataframe(df):
    """Transform scraped data into proper dataframes"""
    if (type(df) == list) & (len(df)==1):
        df = df[0]
        return df
    elif type(df) == pd.core.frame.DataFrame:
        df = df.dropna(subset=['Rk'])
        return df
    else:
        print('Unknown df type')
        return df


def column_joiner(df):
    """Join multi-level column names"""
    if type(df.columns.values[0]) != str:
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        return df
    else:
        return df


def format_column_names(df):
    """Format column names by removing 'Unnamed' prefixes"""
    for col in df.columns:
        if 'Unnamed' in col:
            col_new= col.split('_')[-1]
            df.rename(columns={col:col_new}, inplace=True)
    return df


def scrape(URL, table_id):
    """Main scraping function"""
    df = column_joiner(to_dataframe(read_html_upd(URL, table_id)))
    return df


def team_dict_get(countrycode):
    """Get competition ID and league name from country code"""
    team_dict = {
        'ENG': {'comp_id':'9', 'league':'Premier-League'},
        'ESP': {'comp_id':'12', 'league':'La-Liga'},
        'GER': {'comp_id':'20', 'league':'Bundesliga'},
        'ITA': {'comp_id':'11', 'league':'Serie-A'},
        'FRA': {'comp_id':'13', 'league':'Ligue-1'},
        'UCL': {'comp_id': '8', 'league': 'Champions-League'},
        'UEL': {'comp_id':'19', 'league':'Europa-League'},
        'UECL': {'comp_id':'882', 'league':'Conference-League'},
        'HUN': {'comp_id':'46', 'league':'NB-I'},
        'BRA': {'comp_id':'24', 'league':'Serie-A'},
        'AUT': {'comp_id': '56', 'league': 'Austrian-Bundesliga'},
        'BEL': {'comp_id': '37', 'league': 'Belgian-Pro-League'},
        'USA': {'comp_id': '22', 'league': 'Major-League-Soccer'},
        'POR': {'comp_id': '32', 'league': 'Primeira-Liga'},
        'NED': {'comp_id': '23', 'league': 'Eredivisie'},
        'Big5': {'comp_id': 'Big5', 'league': 'Big-5-European-Leagues'}
    }
    
    comp_id = team_dict.get(countrycode).get('comp_id')
    league_name = team_dict.get(countrycode).get('league')
    
    return comp_id, league_name


def get_all_player_data(countrycode, year=False):
    """
    Get all player data for a league from FBref
    
    Args:
        countrycode: Country/league code (e.g., 'HUN', 'ENG', 'GER')
        year: Season year (e.g., '2024-2025'), False for current season
    
    Returns:
        DataFrame with comprehensive player statistics
    """
    comp_id, league_name = team_dict_get(countrycode)
    stats_list = ['standard', 'keeper', 'keeper_adv', 'defense', 'passing', 'gca', 'misc', 'shooting', 'possession', 'passing_types']
    url_list = ['stats', 'keepers', 'keepersadv', 'defense', 'passing', 'gca', 'misc', 'shooting', 'possession', 'passing_types']
    
    dfs = {}
    
    for stat, url in zip(stats_list, url_list):
        if countrycode == "Big5":
            if year:
                URL = f"https://fbref.com/en/comps/{comp_id}/{year}/{url}/players/{year}-{league_name}-Stats" 
            else:
                URL = f"https://fbref.com/en/comps/{comp_id}/{url}/players/{league_name}-Stats"
        else:
            if year:
                URL = f"https://fbref.com/en/comps/{comp_id}/{year}/{url}/{year}-{league_name}-Stats" 
            else:
                URL = f"https://fbref.com/en/comps/{comp_id}/{url}/{league_name}-Stats#all_stats_{stat}"
        
        df = format_column_names(scrape(URL, f'stats_{stat}'))
        df.drop(df[df['Rk']=='Rk'].index, inplace=True)
        dfs[stat] = df
        print(f'df_{stat} found.')
    
    # Merge all dataframes
    df_standard = dfs['standard'].copy()
    df_standard.rename(columns={'Playing Time_90s': '90s'}, inplace=True)
    
    df_analyse = df_standard.copy()
    for stat in stats_list[1:]:
        df_analyse = pd.merge(df_analyse, dfs[stat],
                              on=['Player', 'Squad'], how='left',
                              suffixes=['','_remove'])
        # Remove duplicate columns
        df_analyse.drop([i for i in df_analyse.columns if 'remove' in i],
                       axis=1, inplace=True)
    
    df_analyse.drop(columns='Matches', inplace=True, errors='ignore')
    df_analyse.iloc[:, 7:] = df_analyse.iloc[:, 7:].astype(float)
    
    return df_analyse


def get_gamelog(countrycode, season="2024-2025"):
    """
    Get match schedule/results for a league
    
    Args:
        countrycode: Country/league code
        season: Season (e.g., '2024-2025')
    
    Returns:
        DataFrame with match fixtures and results
    """
    comp_id, league_name = team_dict_get(countrycode)
    url = f"https://fbref.com/en/comps/{comp_id}/{season}/schedule/{season}-{league_name}-Scores-and-Fixtures"
    table_id = f"sched_{season}_{comp_id}_1"
    
    gamelog = scrape(url, table_id)
    
    return gamelog
