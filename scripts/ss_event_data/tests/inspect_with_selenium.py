import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
options.add_argument('--user-data-dir=C:/Users/Adam/AppData/Local/ChromeSelenium')
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')
options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

driver = webdriver.Chrome(options=options)
url = 'https://www.sofascore.com/hu/football/match/sabah-fk-sk-slavia-praha/qUsDghc#id:16939017'
print(f"Navigating to: {url}")
driver.get(url)

print("Waiting 6 seconds for page and widgets to load...")
time.sleep(6)

print("Page title:", driver.title)

# Check canvases
canvases = driver.find_elements("tag name", "canvas")
print(f"Found {len(canvases)} canvas element(s) in main page:")
for c in canvases:
    cid = c.get_attribute("id")
    cengine = c.get_attribute("data-engine")
    print(f" - Canvas id='{cid}', data-engine='{cengine}'")

# Check iframes
iframes = driver.find_elements("tag name", "iframe")
print(f"Found {len(iframes)} iframe(s):")
for f in iframes:
    fid = f.get_attribute("id")
    fsrc = f.get_attribute("src")
    print(f" - iframe id='{fid}', src='{fsrc[:80] if fsrc else ''}'")

# Evaluate Babylon
babylon_check = driver.execute_script("return typeof BABYLON !== 'undefined';")
print(f"Is BABYLON defined in main window? {babylon_check}")

driver.quit()
print("Test completed.")
