# oddsportal.py

def scrape_oddsportal_fixed(event_url, headless=False):
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import pandas as pd
    import re, time
    from datetime import datetime

    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("user-agent=Mozilla/5.0")
    
    driver = webdriver.Chrome(options=opts)
    driver.get(event_url)
    wait = WebDriverWait(driver, 20)
    
    all_data = []
    page = 1
    
    while True:
        print(f"🔍 Oldal {page} feldolgozása...")

        time.sleep(2)

        # Scroll, hogy minden betöltsön
        for _ in range(80):  # több, kisebb görgetés
            driver.execute_script("window.scrollBy(0, 150);")  # kis lépés
            time.sleep(0.075)
            if _ % 20 == 0:
                time.sleep(1)

        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.eventRow")))

        event_rows = driver.find_elements(By.CSS_SELECTOR, "div.eventRow")
        current_date = None

        for event in event_rows:
            # dátum keresése
            date_found = False
            
            # Dátum keresése az event teljes szövegében
            date_text = event.text.strip()
            if any(month in date_text for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                lines = date_text.split('\n')
                for line in lines:
                    if any(month in line for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                        # Dátum formázása - eltávolítjuk a " -" utáni részt
                        clean_date = line.split(' -')[0].strip()
                        current_date = clean_date
                        
                        # Dátum átalakítása YYYY-MM-dd formátumba
                        try:
                            date_obj = datetime.strptime(current_date, '%d %b %Y')
                            current_date = date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            # Ha nem sikerül átalakítani, marad az eredeti
                            pass
                        
                        date_found = True
                        break
            
            # Ha nincs dátum, de van game-row, akkor meccs sor
            try:
                game_row = event.find_element(By.CSS_SELECTOR, "div[data-testid='game-row']")
            except:
                continue

            # csapatnevek
            participants = game_row.find_elements(By.CSS_SELECTOR, "a[title]")
            if len(participants) < 2:
                continue

            home_team = participants[0].get_attribute("title").strip()
            away_team = participants[1].get_attribute("title").strip()

            # oddsok
            odds_blocks = event.find_elements(By.CSS_SELECTOR, "div[data-testid^='odd-container'] p")
            odds = []
            for p in odds_blocks:
                try:
                    odds_text = p.text.strip().replace(",", ".")
                    if re.match(r"^\d+(\.\d+)?$", odds_text):
                        odds.append(float(odds_text))
                except:
                    continue
            
            home_odds = odds[0] if len(odds) > 0 else None
            away_odds = odds[1] if len(odds) > 1 else None

            all_data.append({
                "Date": current_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "away_odds": away_odds
            })

        # Következő oldal ellenőrzése
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, f"a.pagination-link[data-number='{page + 1}']")
            if next_button.is_enabled():
                print(f"➡️ Következő oldal: {page + 1}")
                driver.execute_script("arguments[0].click();", next_button)
                page += 1
                time.sleep(2)  # Várakozás az oldal betöltésére
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(1)
                continue
            else:
                break
        except:
            # Ha nincs következő oldal, kilépünk
            break

    driver.quit()
    
    df = pd.DataFrame(all_data).drop_duplicates(subset=["home_team","away_team","home_odds","away_odds"])
    print(f"✅ Összesen {len(df)} meccs feldolgozva {page} oldalról.")
    return df