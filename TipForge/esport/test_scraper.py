"""
Test script a Match Scraper teszteléséhez.
"""

from scrapers.h2h_scraper import H2HScraper
import pandas as pd

# Állítsd be az event_id-t, amit tesztelni szeretnél
TEST_URL = "https://www.hltv.org/matches/2384305/faze-vs-aurora-esports-world-cup-2025"  # Cseréld le a kívánt event_id-ra

print("="*80)
print(f"🧪 SCRAPER TESZT - {TEST_URL}")
print("="*80)

# Scraper inicializálása
scraper = H2HScraper()  # headless=False --> látod mi történik

# Scrape futtatása
print(f"\n📊 Scraping indítása...")
df = scraper.scrape_match_h2h(TEST_URL)

# Eredmény kiírása
print("\n" + "="*80)
print("📋 EREDMÉNY:")
print("="*80)

if df.empty:
    print("❌ Nincs adat! A scraper nem talált meccseket.")
else:
    print(f"\n✅ {len(df)} meccs találva\n")
    
    for col in df.columns:
        print(f"{col}: {df.loc[0, col]}")

print("\n" + "="*80)
print("✅ TESZT BEFEJEZVE")
print("="*80)