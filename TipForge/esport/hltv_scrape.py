import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup
import time

# Cloudscraper automatikusan kezeli a Cloudflare védelmet
scraper = cloudscraper.create_scraper()

url = "https://www.hltv.org/results"
response = scraper.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Parse matches
    matches = soup.find_all('div', class_='result-con')
    
    match_data = []
    for match in matches:
        try:
            # Csapatnevek - a 'team' class-t keresük
            teams = match.find_all('div', class_='team')
            team1 = teams[0].get_text(strip=True) if len(teams) > 0 else "N/A"
            team2 = teams[1].get_text(strip=True) if len(teams) > 1 else "N/A"
            
            # Eredmény - a result-score cellában
            score_element = match.find('td', class_='result-score')
            if score_element:
                score_spans = score_element.find_all('span')
                if len(score_spans) >= 2:
                    score = f"{score_spans[0].get_text(strip=True)}-{score_spans[1].get_text(strip=True)}"
                else:
                    score = score_element.get_text(strip=True)
            else:
                score = "N/A"
            
            # Esemény név
            event_element = match.find('span', class_='event-name')
            event = event_element.get_text(strip=True) if event_element else "N/A"
            
            # Match link
            match_link = match.find('a', class_='a-reset')
            match_url = f"https://www.hltv.org{match_link['href']}" if match_link and match_link.get('href') else "N/A"
            
            match_data.append({
                'team1': team1,
                'team2': team2,
                'score': score,
                'event': event,
                'match_url': match_url
            })
            
        except Exception as e:
            print(f"Error parsing match: {e}")
            continue
    
    print(f"Found {len(match_data)} matches")
    if match_data:
        pd.DataFrame(match_data).to_csv('hltv_results.csv', index=False)
        print("Data saved to hltv_results.csv")
    else:
        print("No match data found")
        
else:
    print(f"Failed to fetch page: {response.status_code}")