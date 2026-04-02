import os
import requests
from bs4 import BeautifulSoup
import re
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_URL = "https://promo.betfair.com/betfairsp/prices"
RAW_DIR = "data/raw"

def get_csv_links(year="2026"):
    logging.info(f"Fetching directory listing from {BASE_URL}")
    response = requests.get(BASE_URL)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    
    # Files have names like dwbfpricesukwin02042026.csv
    # We want to match our target year to limit downloading
    for a in soup.find_all('a'):
        href = a.get('href')
        if href and "dwbf" in href and href.endswith(f"{year}.csv"):
            if not href.startswith("http"):
                # some links could be relative
                link = f"{BASE_URL}/{href.split('/')[-1]}"
            else:
                link = href
            links.append(link)
            
    logging.info(f"Found {len(links)} files for year {year}")
    return links

def download_file(url):
    filename = url.split('/')[-1]
    filepath = os.path.join(RAW_DIR, filename)
    
    if os.path.exists(filepath):
        # We can skip downloading if already there
        pass
    else:
        try:
            logging.info(f"Downloading {filename}...")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    links = get_csv_links("2026")
    
    # Download in parallel to speed it up
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_file, links)
        
    logging.info("Download step complete.")

if __name__ == "__main__":
    main()
