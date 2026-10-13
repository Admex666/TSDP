import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
# options.add_argument('--headless=new')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--window-size=1920,1080')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')
options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

driver = webdriver.Chrome(options=options)
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": """
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        })
    """
})

url = 'https://www.sofascore.com/hu/football/match/sabah-fk-sk-slavia-praha/qUsDghc#id:16939017'
driver.get(url)

print('Waiting for Turnstile / redirect...')
for sec in range(12):
    time.sleep(1)
    if 'captcha' not in driver.current_url:
        print(f'Redirected to match page after {sec+1}s!')
        break

print('Current URL:', driver.current_url)
print('Title:', driver.title)
body_text = driver.find_element('tag name', 'body').text
print('Body snippet:', repr(body_text[:300]))

# Check if ws was created
logs = driver.get_log('performance')
ws_urls = set()
ws_messages = 0
for entry in logs:
    log = json.loads(entry['message'])['message']
    method = log.get('method', '')
    if method == 'Network.webSocketCreated':
        ws_urls.add(log['params'].get('url'))
    if method == 'Network.webSocketFrameReceived':
        ws_messages += 1

print('WS URLs:', ws_urls)
print('WS Messages received:', ws_messages)

driver.quit()
