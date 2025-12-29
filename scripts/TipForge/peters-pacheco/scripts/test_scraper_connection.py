import tls_client

identifiers = [
    "chrome_120", 
    "chrome_118", 
    "safari_16_0", 
    "firefox_117",
    "opera_90"
]

url = "https://www.sofascore.com/api/v1/unique-tournament/17/seasons"

for ident in identifiers:
    print(f"Testing {ident}...")
    try:
        sess = tls_client.Session(client_identifier=ident)
        resp = sess.get(url)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print("Success!")
            # Print first 2 seasons to confirm structure
            data = resp.json()
            seasons = data.get('seasons', [])
            for s in seasons[:3]:
                print(f"Season: {s['name']} (Year: {s['year']}) -> ID: {s['id']}")
            break
    except Exception as e:
        print(f"Error: {e}")
