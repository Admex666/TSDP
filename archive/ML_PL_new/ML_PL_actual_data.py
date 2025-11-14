import pandas as pd
from datetime import datetime, time
import os
wd_old = os.getcwd()
if wd_old != 'C:\\Users\\Adam\\..Data\\TSDP':
    wd_base = wd_old.split('\\')[:4]
    wd_new = '\\'.join(wd_base)+'\\TSDP'
    os.chdir(wd_new)

#%% Get data that matters
input_path = 'ML_PL_new/predictions.xlsx'
df_predprobs = pd.read_excel(input_path, sheet_name='pred_probabilities')
today = datetime.today()
span = pd.to_timedelta('5D')
end_date = datetime.combine((today + span), time(23,59))
mask_dates = (df_predprobs.Date > today) & (df_predprobs.Date <= end_date)
df_actual = df_predprobs[mask_dates]

df_output = df_actual[['Date', 'HomeTeam', 'AwayTeam', 'H_gNB_prob', 'D_gNB_prob', 'A_gNB_prob']]
df_output.loc[:,['H_odds', 'D_odds', 'A_odds']] = None

#%% To excel
output_path = 'ML_PL_new\\actual_preds.xlsx'
df_output.to_excel(output_path, index=False)