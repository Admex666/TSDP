from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

url = "https://www.tippmixpro.hu/hu/fogadas/i/bajnoksag-lokacio/kosarlabda/8/usa/229/nba-2025-2026/274663790763708416"

chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_options)

print(f"Navigating to {url} ...")
driver.get(url)
time.sleep(5)

html = driver.page_source

# 1. Check parent frame
print("======= PARENT FRAME =======")
print(f"Contains 'Lakers': {'Lakers' in html}")
print(f"Contains 'SportsIframe': {'SportsIframe' in html}")
print(f"Contains 'EventItem' class: {'EventItem' in html}")

iframes = driver.find_elements(By.TAG_NAME, "iframe")
print(f"Found {len(iframes)} iframes.")
for i, frame in enumerate(iframes):
    src = frame.get_attribute('src')
    fid = frame.get_attribute('id')
    print(f"Iframe {i}: ID='{fid}', SRC='{src}'")

# 2. Check SportsIframe if it exists
try:
    driver.switch_to.frame("SportsIframe")
    time.sleep(2)
    ihml = driver.page_source
    print("\n======= INSIDE SportsIframe =======")
    print(f"Contains 'Lakers': {'Lakers' in ihml}")
    print(f"Contains 'EventItem' class: {'EventItem' in ihml}")
    
    # Let's print the first few text elements that look like team names
    participants = driver.find_elements(By.CSS_SELECTOR, ".Details__ParticipantName")
    print(f"Found {len(participants)} participant elements.")
    for p in participants[:10]:
        print(f" - {p.text.strip()}")
        
except Exception as e:
    print(f"Could not switch to SportsIframe: {e}")

driver.quit()
