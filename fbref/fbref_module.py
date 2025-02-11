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
    
    options = webdriver.ChromeOptions()
    #options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    driver.get(URL)
    
    try:
        table = WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.ID, table_id))
        )
        
        table_html = table.get_attribute('outerHTML')
        
        df = pd.read_html(table_html)[0]
        
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