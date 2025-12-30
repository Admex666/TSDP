from curl_cffi import requests

def debug_with_cookies():
    print("--- Testing Session Warming (Homepage -> API) ---")
    
    # Session automatically handles cookies
    session = requests.Session()
    
    # 1. Visit Homepage to get Cookies
    try:
        print("1. Visiting Homepage...")
        resp_home = session.get(
            "https://www.sofascore.com/",
            impersonate="chrome110",
            timeout=10
        )
        print(f"Homepage Status: {resp_home.status_code}")
        print(f"Cookies collected: {session.cookies.get_dict()}")
        
    except Exception as e:
        print(f"Homepage failed: {e}")
        return

    # 2. Call API with the warmed session
    target_url = "https://www.sofascore.com/api/v1/unique-tournament/17/season/41886/events/round/1"
    print(f"\n2. Calling API: {target_url}")
    
    try:
        resp_api = session.get(
            target_url,
            impersonate="chrome110",
            timeout=10
        )
        print(f"API Status: {resp_api.status_code}")
        
        if resp_api.status_code == 200:
            print("SUCCESS! Data received.")
            # print(resp_api.json())
        else:
            print("FAILED.")
            
    except Exception as e:
        print(f"API failed: {e}")

if __name__ == "__main__":
    debug_with_cookies()
