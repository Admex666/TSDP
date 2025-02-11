#%% Scraper
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd



# Set up Chrome options
chrome_options = webdriver.ChromeOptions()
# Uncomment the line below if you want to run in headless mode
# chrome_options.add_argument('--headless')

# Initialize the Chrome WebDriver
service = ChromeService(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Navigate to the website
    url = 'https://www.betshoot.com/dropping-odds/'
    driver.get(url)

    # Wait for the matches to load (adjust timeout as needed)
    wait = WebDriverWait(driver, 10)
    matches = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'dfem')))

    # Create list to store data
    data = []
    
    # Process each match
    for match in matches:
        try:
            league = match.find_element(By.CLASS_NAME, 'pre').text.strip()
            datetime = match.find_element(By.CLASS_NAME, 'datetime').text.strip()
            match_teams = match.find_element(By.CLASS_NAME, 'match').text.strip()
            
            # Get odds
            odds_divs = match.find_element(By.CLASS_NAME, 'pre1').find_elements(By.TAG_NAME, 'div')
            odds_before_after = [div.text.split('\n') for div in odds_divs]
            
            # Get labels
            labels = [span.text for span in match.find_element(By.CLASS_NAME, 'pre2').find_elements(By.TAG_NAME, 'span')]
            
            # Create match data dictionary
            match_data = {
                'League': league,
                'DateTime': datetime,
                'Match': match_teams
            }
            
            # Add odds with their corresponding labels, now for both before and after
            for label, odds_pair in zip(labels, odds_before_after):
                if len(odds_pair) >= 2:  # Ensure we have both before and after odds
                    match_data[f'{label}_bef'] = odds_pair[0]
                    match_data[f'{label}_aft'] = odds_pair[1]
                
            data.append(match_data)
            
        except Exception as e:
            print(f"Error processing match: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Display the first few rows
    print(df.head())

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Always close the driver
    driver.quit()
    
#%% Formatting the dataframe
df.drop(columns=['O 2.5_bef', 'O 2.5_aft', 'U 2.5_bef', 'U 2.5_aft'], inplace=True)
df.iloc[:,3:11] = df.iloc[:,3:11].astype(float)

#%% Print the odds drops
# Initialize an empty string to store the odds drop messages
odds_drop_messages = ""

# Loop through the dataframe to check for odds drops
for row in range(len(df)):
    for bef_odds_nr, aft_odds_nr in [[3,4],[5,6],[7,8],[9,10]]:
        if df.iloc[row,bef_odds_nr] > df.iloc[row,aft_odds_nr]:
            match_curr = df.loc[row,'Match']
            # Append the odds drop message to the string
            odds_drop_messages += "ODDS DROP\n"
            odds_drop_messages += f"{match_curr}\n"
            odds_drop_messages += f"Bet on {df.columns[aft_odds_nr].split('_')[0]} if over {df.iloc[row,aft_odds_nr]}\n"
            odds_drop_messages += "\n"  # Add an extra line break for readability

#%% Import email alert
import email_alert as ea
mail = 'adam.jakus99@gmail.com'
ea.email_alert('Odds drops alert', odds_drop_messages, mail)
