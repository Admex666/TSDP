import tls_client

session = tls_client.Session(
    client_identifier="chrome_120",
    random_tls_extension_order=True
)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

print("Testing seasoned scraper...")

try:
    # 1. Visit homepage to maybe set cookies? 
    # (SofaScore API usually doesn't strictly require this but good practice)
    # resp_home = session.get("https://www.sofascore.com", headers=headers)
    # print(f"Home Status: {resp_home.status_code}")

    # 2. API Call
    url = "https://www.sofascore.com/api/v1/unique-tournament/17/seasons"
    resp = session.get(url, headers=headers)
    print(f"API Status: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        print("Success!")
        for s in data['seasons'][:5]:
            print(f"{s['year']}: {s['id']}")
            
except Exception as e:
    print(f"Error: {e}")
