def read_html(URL, table_id):
    import pandas as pd
    df = pd.read_html(URL, attrs={"id": table_id})
    return df

def read_html_upd(URL, table_id):
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from io import StringIO
    
    options = webdriver.ChromeOptions()
    #options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    #driver = webdriver.Chrome(service=Service(r"C:\Users\Adam\.wdm\drivers\chromedriver\win64\114.0.5735.90\chromedriver.exe"), options=options)

    
    driver.get(URL)
    
    try:
        table = WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.ID, table_id))
        )
        
        table_html = table.get_attribute('outerHTML')
        
        df = pd.read_html(StringIO(table_html))[0]
        
        return [df]
    finally:
        driver.quit()
        
def read_html_upd_firefox(URL, table_id):
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from webdriver_manager.firefox import GeckoDriverManager
    from io import StringIO
    
    options = webdriver.FirefoxOptions()
    # options.add_argument('--headless')
    
    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
    
    driver.get(URL)
    
    try:
        table = WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.ID, table_id))
        )
        
        table_html = table.get_attribute('outerHTML')
        
        df = pd.read_html(StringIO(table_html))[0]
        
        return [df]
    finally:
        driver.quit()

# creating a function for transforming scraped data into proper dataframes
def to_dataframe(df):
    import pandas as pd
    if (type(df) == list) & (len(df)==1):
        df = df[0]
        return df
    elif type(df) == pd.core.frame.DataFrame:
        df = df.dropna(subset=['Rk'])
        return df
    else:
        print('Unknown df type')
        
    if type(df) != pd.core.frame.DataFrame:
        print('Not a pandas dataframe')
    else:
        return df

def column_joiner(df):
    if type(df.columns.values[0]) != str:
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        return df
    else:
        return df

def format_column_names(df):
    for ncol in range(len(df.columns)):
        if 'Unnamed' in df.columns[ncol]:
            col_old_name = df.columns[ncol]
            col_new_name = df.columns[ncol].split('_')[3]
            df.rename(columns={col_old_name:col_new_name}, inplace=True)
        else:
            pass
    return df

def scrape(URL, table_id):
    df = column_joiner(to_dataframe(read_html_upd(URL, table_id)))
    return df

def scrape_ffox(URL, table_id):
    df = column_joiner(to_dataframe(read_html_upd_firefox(URL, table_id)))
    return df

def scrape_prev(URL, table_id):
    df = column_joiner(to_dataframe(read_html(URL, table_id)))
    return df

def season_to_next(season):
    season_list = season.split('-')
    season_list[0] = str(int(season_list[0])+1)
    season_list[1] = str(int(season_list[1])+1)
    new_season = '-'.join(season_list)
    return new_season

def linreg(col1, col2):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    model = LinearRegression()
    array_col1 = np.array([col1]).reshape((-1,1))
    
    model.fit(array_col1, col2)
    
    r2 = model.score(array_col1, col2)
    b0 = model.intercept_
    b1 = model.coef_[0]
    return [r2, b0, b1]

def team_dict_get(countrycode):
    team_dict = {'ENG': {'comp_id':'9', 'league':'Premier-League'},
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
                 'WCQ_SA': {'comp_id': '4', 'league': 'WCQ----CONMEBOL-M'},
                 'UNL': {'comp_id': '677', 'league': 'UEFA-Nations-League'}
                 }
    
    comp_id = team_dict.get(countrycode).get('comp_id')
    league_name = team_dict.get(countrycode).get('league')
    
    return comp_id, league_name


def stats_dict():
    combined_stats = {
        'Performance_Sh': {'name': 'Shots', 'category': 'Offensive', 'significance': 'positive'},
        'Carries_Carries': {'name': 'Carries', 'category': 'Possession', 'significance': 'positive'},
        'Carries_PrgC': {'name': 'Progressive Carries', 'category': 'Possession', 'significance': 'positive'},
        'Total_Cmp': {'name': 'Passes Completed', 'category': 'Passing', 'significance': 'positive'},
        'Total_Cmp%': {'name': 'Pass Completion %', 'category': 'Passing', 'significance': 'positive'},
        'Total_TotDist': {'name': 'Total Passing Distance', 'category': 'Passing', 'significance': 'positive'},
        'Total_PrgDist': {'name': 'Progressive Passing Distance', 'category': 'Passing', 'significance': 'positive'},
        'Short_Cmp%': {'name': 'Short Pass Completion %', 'category': 'Passing', 'significance': 'positive'},
        'Medium_Cmp': {'name': 'Medium Passes Completed', 'category': 'Passing', 'significance': 'positive'},
        'Medium_Att': {'name': 'Medium Passes Attempted', 'category': 'Passing', 'significance': 'positive'},
        'Medium_Cmp%': {'name': 'Medium Pass Completion %', 'category': 'Passing', 'significance': 'positive'},
        'Long_Cmp': {'name': 'Long Passes Completed', 'category': 'Passing', 'significance': 'positive'},
        'Long_Cmp%': {'name': 'Long Pass Completion %', 'category': 'Passing', 'significance': 'positive'},
        '1/3': {'name': 'Passes into Final Third', 'category': 'Passing', 'significance': 'positive'},
        'PPA': {'name': 'Passes into Penalty Area', 'category': 'Passing', 'significance': 'positive'},
        'PrgP': {'name': 'Progressive Passes', 'category': 'Passing', 'significance': 'positive'},
        'Tackles_Tkl': {'name': 'Tackles', 'category': 'Defensive', 'significance': 'positive'},
        'Tackles_Att 3rd': {'name': 'Tackles in Attacking Third', 'category': 'Defensive', 'significance': 'positive'},
        'Challenges_Tkl': {'name': 'Dribblers Tackled', 'category': 'Defensive', 'significance': 'positive'},
        'Challenges_Att': {'name': 'Dribbles Challenged', 'category': 'Defensive', 'significance': 'positive'},
        'Challenges_Tkl%': {'name': 'Tackle Success %', 'category': 'Defensive', 'significance': 'positive'},
        'Tkl+Int': {'name': 'Tackles + Interceptions', 'category': 'Defensive', 'significance': 'positive'},
        'Touches_Att 3rd': {'name': 'Touches in Attacking Third', 'category': 'Possession', 'significance': 'positive'},
        'Carries_TotDist': {'name': 'Total Carrying Distance', 'category': 'Possession', 'significance': 'positive'},
        'Carries_PrgDist': {'name': 'Progressive Carrying Distance', 'category': 'Possession', 'significance': 'positive'},
        'Carries_1/3': {'name': 'Carries into Final Third', 'category': 'Possession', 'significance': 'positive'},
        'Carries_Mis': {'name': 'Miscontrols', 'category': 'Possession', 'significance': 'negative'},
        'Receiving_Rec': {'name': 'Passes Received', 'category': 'Possession', 'significance': 'positive'},
        'Receiving_PrgR': {'name': 'Progressive Passes Received', 'category': 'Possession', 'significance': 'positive'},
        'Performance_Fls': {'name': 'Fouls Committed', 'category': 'Miscellaneous', 'significance': 'negative'},
        'Performance_Recov': {'name': 'Ball Recoveries', 'category': 'Defensive', 'significance': 'positive'},
        'Aerial Duels_Won': {'name': 'Aerial Duels Won', 'category': 'Aerial Duels', 'significance': 'positive'},
        'Aerial Duels_Lost': {'name': 'Aerial Duels Lost', 'category': 'Aerial Duels', 'significance': 'negative'},
        'Aerial Duels_Won%': {'name': 'Aerial Duels Won %', 'category': 'Aerial Duels', 'significance': 'positive'},
        'Performance_SoT': {'name': 'Shots on Target', 'category': 'Offensive', 'significance': 'positive'},
        'Expected_xG': {'name': 'Expected Goals', 'category': 'Offensive', 'significance': 'positive'},
        'Expected_npxG': {'name': 'Non-Penalty Expected Goals', 'category': 'Offensive', 'significance': 'positive'},
        'Long_Att': {'name': 'Long Passes Attempted', 'category': 'Passing', 'significance': 'positive'},
        'KP': {'name': 'Key Passes', 'category': 'Offensive', 'significance': 'positive'},
        'CrsPA': {'name': 'Crosses into Penalty Area', 'category': 'Passing', 'significance': 'positive'},
        'Touches_Att Pen': {'name': 'Touches in Attacking Penalty Area', 'category': 'Possession', 'significance': 'positive'},
        'Take-Ons_Tkld': {'name': 'Take-Ons Tackled', 'category': 'Possession', 'significance': 'negative'},
        'Take-Ons_Tkld%': {'name': 'Take-Ons Tackled %', 'category': 'Possession', 'significance': 'negative'},
        'Performance_Crs': {'name': 'Crosses', 'category': 'Passing', 'significance': 'positive'},
        'Tackles_Mid 3rd': {'name': 'Tackles in Middle Third', 'category': 'Defensive', 'significance': 'positive'},
        'Take-Ons_Succ%': {'name': 'Successful Take-Ons %', 'category': 'Possession', 'significance': 'positive'},
        'Carries_Dis': {'name': 'Dispossessed', 'category': 'Possession', 'significance': 'negative'},
        'Take-Ons_Att': {'name': 'Take-Ons Attempted', 'category': 'Possession', 'significance': 'positive'},
        'Take-Ons_Succ': {'name': 'Successful Take-Ons', 'category': 'Possession', 'significance': 'positive'},
        'Total_Att': {'name': 'Passes Attempted', 'category': 'Passing', 'significance': 'positive'},
        'Tackles_TklW': {'name': 'Tackles Won', 'category': 'Defensive', 'significance': 'positive'},
        'Challenges_Lost': {'name': 'Challenges Lost', 'category': 'Defensive', 'significance': 'negative'},
        'Blocks_Blocks': {'name': 'Blocks', 'category': 'Defensive', 'significance': 'positive'},
        'Blocks_Pass': {'name': 'Passes Blocked', 'category': 'Defensive', 'significance': 'positive'},
        'Clr': {'name': 'Clearances', 'category': 'Defensive', 'significance': 'positive'},
        'Touches_Touches': {'name': 'Touches', 'category': 'Possession', 'significance': 'positive'},
        'Touches_Live': {'name': 'Live-Ball Touches', 'category': 'Possession', 'significance': 'positive'},
        'Carries_CPA': {'name': 'Carries into Penalty Area', 'category': 'Possession', 'significance': 'positive'},
        'Performance_Off': {'name': 'Offsides', 'category': 'Miscellaneous', 'significance': 'negative'},
        'Performance_TklW': {'name': 'Tackles Won', 'category': 'Defensive', 'significance': 'positive'},
        'Performance_Ast': {'name': 'Assists', 'category': 'Offensive', 'significance': 'positive'},
        'Ast': {'name': 'Assists', 'category': 'Offensive', 'significance': 'positive'},
        'Expected_xAG': {'name': 'Expected Assisted Goals', 'category': 'Offensive', 'significance': 'positive'},
        'xAG': {'name': 'Expected Assists', 'category': 'Offensive', 'significance': 'positive'},
        'Performance_Fld': {'name': 'Fouls Drawn', 'category': 'Miscellaneous', 'significance': 'positive'},
        'Performance_Gls': {'name': 'Goals', 'category': 'Offensive', 'significance': 'positive'},
        'Performance_Int': {'name': 'Interceptions', 'category': 'Defensive', 'significance': 'positive'},
        'Int': {'name': 'Interceptions', 'category': 'Defensive', 'significance': 'positive'},
        'Performance_Sh/90': {'name': 'Shots per 90', 'category': 'Offensive', 'significance': 'positive'},
        'Performance_PK': {'name': 'Penalties Scored', 'category': 'Offensive', 'significance': 'positive'},
        'Performance_PKatt': {'name': 'Penalties Attempted', 'category': 'Offensive', 'significance': 'positive'},
        'Performance_CrdY': {'name': 'Yellow Cards', 'category': 'Disciplinary', 'significance': 'negative'},
        'Performance_CrdR': {'name': 'Red Cards', 'category': 'Disciplinary', 'significance': 'negative'},
        'Short_Cmp': {'name': 'Short Passes Completed', 'category': 'Passing', 'significance': 'positive'},
        'Short_Att': {'name': 'Short Passes Attempted', 'category': 'Passing', 'significance': 'positive'},
        'Tackles_Def 3rd': {'name': 'Tackles in Defensive Third', 'category': 'Defensive', 'significance': 'positive'},
        'Blocks_Sh': {'name': 'Shots Blocked', 'category': 'Defensive', 'significance': 'positive'},
        'Err': {'name': 'Errors Leading to Shots', 'category': 'Defensive', 'significance': 'negative'},
        'Touches_Def Pen': {'name': 'Touches in Defensive Penalty Area', 'category': 'Defensive', 'significance': 'positive'},
        'Touches_Def 3rd': {'name': 'Touches in Defensive Third', 'category': 'Defensive', 'significance': 'positive'},
        'Touches_Mid 3rd': {'name': 'Touches in Middle Third', 'category': 'Defensive', 'significance': 'positive'},
        'Performance_2CrdY': {'name': 'Second Yellow Cards', 'category': 'Disciplinary', 'significance': 'negative'},
        'Performance_PKwon': {'name': 'Penalties Won', 'category': 'Offensive', 'significance': 'positive'},
        'Performance_PKcon': {'name': 'Penalties Conceded', 'category': 'Defensive', 'significance': 'negative'},
        'Performance_OG': {'name': 'Own Goals', 'category': 'Defensive', 'significance': 'negative'},
        'Expected_PSxG+/-': {'name': 'Goals Prevented', 'category': 'Goalkeeping', 'significance': 'positive'},
        'Performance_Save%': {'name': 'Save Rate %', 'category': 'Goalkeeping', 'significance': 'positive'},
        'Passes_Launch%': {'name': 'Launched Pass Rate %', 'category': 'Passing', 'significance': 'positive'},
        'SoTA/GA': {'name': 'Shots on target per Goals against', 'category': 'Goalkeeping', 'significance': 'positive'},
        'Per 90 Minutes_G-PK': {'name': 'Non-Penalty Goals per 90', 'category': 'Offense', 'significance': 'positive'},
        'Per 90 Minutes_npxG': {'name': 'Non-Penalty Expected Goals per 90', 'category': 'Offense', 'significance': 'positive'},
        'Per 90 Minutes_Ast': {'name': 'Assists per 90', 'category': 'Offense', 'significance': 'positive'},
        'Per 90 Minutes_xAG':  {'name': 'Expected Assisted Goals per 90', 'category': 'Offense', 'significance': 'positive'},
        'Progression_PrgP':  {'name': 'Progressive Passes', 'category': 'Passing', 'significance': 'positive'},
        'Progression_PrgC':  {'name': 'Progressive Carries', 'category': 'Possession', 'significance': 'positive'},
        'Progression_PrgR':  {'name': 'Progressive Runs', 'category': 'Possession', 'significance': 'positive'},
        'SCA_SCA90':  {'name': 'Shot Creating Actions per 90', 'category': 'Offense', 'significance': 'positive'},
    }
    
    return combined_stats