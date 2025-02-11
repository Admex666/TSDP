import fbref_module as fbref
import numpy as np

URL = 'https://fbref.com/en/players/bcb2ccd8/Peter-Gulacsi'

#%% scraping and formatting data for radar chart
df_scout_summary = fbref.fbref_to_dataframe(
    fbref.fbref_read_html(URL, 'scout_summary_GK'))
df_scout_summary = df_scout_summary.dropna().reset_index(drop=True)

df_scout_summary.info()

df_scout_summary.to_excel(r'C:\TwitterSportsDataProject\fbref scrapes\fbref_scout_summary.xlsx')
