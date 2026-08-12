import sys
sys.path.append(r"C:\Users\Adam\Data\TSDP\modules")
from fbref_module import scrape_undetected

TEST_URL = "https://fbref.com/en/comps/9/2020-2021/2020-2021-Premier-League-Stats"
TEST_TABLE_ID = "results2020-202191_overall"

df = scrape_undetected(TEST_URL, TEST_TABLE_ID)

if df is not None:
    print("\n=== EREDMÉNY ===")
    print(df.head())
    print(f"\nSorok száma: {len(df)}")
else:
    print("\nSIKERTELEN LEKÉRDEZÉS.")
