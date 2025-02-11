"""
import requests
from bs4 import BeautifulSoup

# URL of the webpage
url = 'https://footystats.org/matches'

# Add headers with a User-Agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Send a GET request with custom headers
response = requests.get(url, headers=headers)
response.raise_for_status()  # Ensure request was successful

# Parse the page content with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.prettify())
# Example: Find all match entries
#matches = soup.find_all('div', class_='team-item stat very-green')


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set up the WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Open the webpage
driver.get('https://footystats.org/matches')

try:
    # Wait for the content to load
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'div.team-item.stat.very-green'))
    )
    
    # Find and print the specific div
    divs = driver.find_elements(By.CSS_SELECTOR, 'div.team-item.stat.very-green')
    for div in divs:
        print(div.text)
except Exception as e:
    print("Error:", e)
finally:
    # Close the WebDriver
    driver.quit()




from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set up the WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Open the webpage
driver.get('https://footystats.org/matches')

try:
    # Wait for the main container to be present
    main_container = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#content > div.ui-body.ms > div.ui-wrapper.mobile-ms.cf > div.ui-col.matchSearch.last > ul.ui-team-results.matches"))
    )

    # Find all divs that match the given CSS selector
    elements = driver.find_elements(By.CSS_SELECTOR, "#content > div.ui-body.ms > div.ui-wrapper.mobile-ms.cf > div.ui-col.matchSearch.last > ul.ui-team-results.matches > div:nth-child(2) > div.scroll-wrapper > div:nth-child(2) > div")

    # Print the number of elements found
    print(f"Found {len(elements)} elements")

    # Print the text of each element to debug
    for i, element in enumerate(elements):
        print(f"Element {i}: {element.text}")

except Exception as e:
    print("Error:", e)
finally:
    # Close the WebDriver
    driver.quit()
"""

### pip install html5lib

import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen

URL = 'https://footystats.org/matches'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
req = Request(URL, headers=headers)
html = urlopen(req).read()

leves = soup(html, 'html.parser')

tablak = pd.read_html(str(leves))
## itt a pandas együttműkődik a beautifulsoup tagelésével

tablak