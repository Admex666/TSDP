from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Set up WebDriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run in headless mode to avoid opening the browser window

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

# Open the webpage
url = 'https://www.sofascore.com/hu/football/match/como-hellas-verona/bebseeb#id:12499306,tab:statistics'
driver.get(url)

# Wait for the page to fully load
time.sleep(5)  # You can adjust this or replace with WebDriverWait

# Get the fully rendered page source
page_source = driver.page_source

# Parse the page source with BeautifulSoup
soup = BeautifulSoup(page_source, 'html.parser')

# Locate the image element using the CSS selector provided
image_element1 = soup.select_one('div:nth-child(1) > div > a > div > img')
image_element2 = soup.select_one('div:nth-child(3) > div > a > div > img')

if image_element1:
    # Extract the 'alt' attribute from the image element
    alt_text_home = image_element1.get('alt')
    print("Home:", alt_text_home)
else:
    print("Image element not found")

if image_element2:
    # Extract the 'alt' attribute from the image element
    alt_text_away = image_element2.get('alt')
    print("Away:", alt_text_away)
else:
    print("Image element not found")

driver.quit()
