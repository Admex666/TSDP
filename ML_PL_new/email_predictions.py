import pandas as pd
from datetime import datetime

# Import predictions
preds = pd.read_excel('ML_PL_new/predictions.xlsx', sheet_name='predictions')
predprobs = pd.read_excel('ML_PL_new/predictions.xlsx', sheet_name='pred_probabilities')

# Choose best models
profits = pd.read_excel('ML_PL_new/paperbets.xlsx', sheet_name='profits')
## predprobs: O/U2.5 gNB!, FTR_DT?; preds: FTR_gNB?

predprobs_needed = predprobs[['Date', 'HomeTeam', 'AwayTeam', 'Over_gNB_prob', 'Under_gNB_prob']]

# Matches that didn't start yet
today = datetime.today()
bet_advise = predprobs_needed[today <= predprobs_needed.Date]
bet_advise['Over_gNB_fairodds'] = 1/bet_advise['Over_gNB_prob']
bet_advise['Under_gNB_fairodds'] = 1/bet_advise['Under_gNB_prob']

email_text = ''
for i, row in bet_advise.iterrows():
    row_text = f"{row['HomeTeam']} vs. {row['AwayTeam']}, {row['Date']}\nOver: {row['Over_gNB_fairodds']:.2f}\nUnder: {row['Under_gNB_fairodds']:.2f}\n\n"
    email_text += row_text

print(email_text)

#%% Email alert
import smtplib
from email.message import EmailMessage
import calendar

def email_alert(subject, body, to):
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to
    
    user = 'admexalert@gmail.com'
    msg['from'] = user
    password = 'lcfdpuxprphfxphe'
    # c/WH=.G8pCZ@ay(u:wRebT
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)
    
    server.quit()
    
if __name__ == '__main__':
    date_str = f'{today.day} {calendar.month_name[today.month]} {today.year}'
    email_alert(f'ML Tips, {date_str}', email_text, 'adam.jakus99@gmail.com')
