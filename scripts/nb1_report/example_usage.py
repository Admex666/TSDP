"""
Példa használat - Liga Riport Generátor
Ezt a scriptet futtatva egy minta riportot generálsz
"""

from report_generator import LigaReportGenerator
import json
from pathlib import Path


def create_sample_data():
    """
    Létrehoz egy minta adatfájlt teszteléshez
    Ez mutatja, milyen formátumban kell átadni az adatokat
    """
    sample_data = {
        "league_name": "Premier League",
        "season": "2024/25",
        "round_number": 10,
        "date_range": {
            "start": "2024-11-02",
            "end": "2024-11-04"
        },
        "logo_path": "assets/premier_league_logo.png",
        
        "teams": [
            {
                "team_id": "mci",
                "team_name": "Manchester City",
                "position": 1,
                "position_history": [2, 1, 1, 1],
                "matches_played": 10,
                "wins": 7,
                "draws": 2,
                "losses": 1,
                "points": 23,
                "goals_for": 25,
                "goals_against": 8,
                "goal_difference": 17,
                "xg_total": 23.5,
                "xga_total": 7.2,
                "last_5_results": "WWDWW",
                "last_5_points": 13
            },
            {
                "team_id": "ars",
                "team_name": "Arsenal",
                "position": 2,
                "position_history": [1, 2, 2, 2],
                "matches_played": 10,
                "wins": 6,
                "draws": 3,
                "losses": 1,
                "points": 21,
                "goals_for": 22,
                "goals_against": 10,
                "goal_difference": 12,
                "xg_total": 20.8,
                "xga_total": 9.5,
                "last_5_results": "WDWDW",
                "last_5_points": 11
            },
            {
                "team_id": "liv",
                "team_name": "Liverpool",
                "position": 3,
                "position_history": [3, 3, 4, 3],
                "matches_played": 10,
                "wins": 6,
                "draws": 2,
                "losses": 2,
                "points": 20,
                "goals_for": 21,
                "goals_against": 12,
                "goal_difference": 9,
                "xg_total": 19.2,
                "xga_total": 11.8,
                "last_5_results": "WLWWD",
                "last_5_points": 10
            },
            {
                "team_id": "che",
                "team_name": "Chelsea",
                "position": 4,
                "position_history": [5, 4, 3, 4],
                "matches_played": 10,
                "wins": 5,
                "draws": 3,
                "losses": 2,
                "points": 18,
                "goals_for": 19,
                "goals_against": 13,
                "goal_difference": 6,
                "xg_total": 17.5,
                "xga_total": 13.2,
                "last_5_results": "DWWDW",
                "last_5_points": 10
            },
            {
                "team_id": "mun",
                "team_name": "Manchester United",
                "position": 14,
                "position_history": [12, 13, 14, 14],
                "matches_played": 10,
                "wins": 3,
                "draws": 2,
                "losses": 5,
                "points": 11,
                "goals_for": 12,
                "goals_against": 17,
                "goal_difference": -5,
                "xg_total": 14.8,
                "xga_total": 15.2,
                "last_5_results": "LLWDL",
                "last_5_points": 4
            },
        ],
        
        "matches": [
            {
                "match_id": "m101",
                "date": "2024-11-02",
                "home_team": "Manchester City",
                "away_team": "Arsenal",
                "home_goals": 2,
                "away_goals": 1,
                "home_xg": 1.8,
                "away_xg": 1.2,
                "home_shots": 15,
                "away_shots": 10,
                "home_possession": 58,
                "away_possession": 42
            },
            {
                "match_id": "m102",
                "date": "2024-11-02",
                "home_team": "Liverpool",
                "away_team": "Chelsea",
                "home_goals": 1,
                "away_goals": 1,
                "home_xg": 2.1,
                "away_xg": 0.9,
                "home_shots": 18,
                "away_shots": 8,
                "home_possession": 62,
                "away_possession": 38
            },
            {
                "match_id": "m103",
                "date": "2024-11-03",
                "home_team": "Manchester United",
                "away_team": "West Ham",
                "home_goals": 0,
                "away_goals": 2,
                "home_xg": 0.8,
                "away_xg": 2.3,
                "home_shots": 9,
                "away_shots": 14,
                "home_possession": 48,
                "away_possession": 52
            }
        ],
        
        "next_round_fixtures": [
            {
                "fixture_id": "f111",
                "date": "2024-11-09",
                "home_team": "Arsenal",
                "away_team": "Liverpool",
                "home_position": 2,
                "away_position": 3,
                "difficulty_rating": 8.5,
                "description": "Tabellaszomszédok csatája - Bajnoki Liga helyért"
            },
            {
                "fixture_id": "f112",
                "date": "2024-11-09",
                "home_team": "Chelsea",
                "away_team": "Manchester City",
                "home_position": 4,
                "away_position": 1,
                "difficulty_rating": 9.0,
                "description": "Címvédő elleni rangadó"
            },
            {
                "fixture_id": "f113",
                "date": "2024-11-10",
                "home_team": "Manchester United",
                "away_team": "Leicester",
                "home_position": 14,
                "away_position": 16,
                "difficulty_rating": 6.0,
                "description": "Bennmaradási harc"
            }
        ]
    }
    
    # Mentés JSON fájlba
    Path("data").mkdir(exist_ok=True)
    with open("data/round_10_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("✓ Minta adatfájl létrehozva: data/round_10_sample.json")
    return "data/round_10_sample.json"


def example_basic_usage():
    """
    Alap használat példa
    """
    print("\n" + "="*60)
    print("PÉLDA 1: Alap használat")
    print("="*60 + "\n")
    
    # 1. Minta adat létrehozása
    data_path = create_sample_data()
    
    # 2. Generátor inicializálása
    generator = LigaReportGenerator("config.yaml")
    
    # 3. Riport generálás
    pdf_path = generator.generate_report(data_path)
    
    print(f"\n✓ Sikeres generálás! PDF helye: {pdf_path}")


def example_custom_config():
    """
    Egyedi konfiguráció használata
    """
    print("\n" + "="*60)
    print("PÉLDA 2: Egyedi konfiguráció")
    print("="*60 + "\n")
    
    # Egyedi config létrehozása Bundesliga-hoz
    custom_config = {
        "league": {
            "name": "Bundesliga",
            "season": "2024/25",
            "country": "Németország"
        },
        "zone_limits": {
            "champions_league": 4,
            "europa_league": 5,
            "conference_league": 6,
            "relegation_start": 16  # 18 csapatos liga
        },
        "thresholds": {
            "good_form": 2.5,
            "bad_form": 1.0,
            "overperforming": 5,
            "underperforming": 5,
            "surprise_factor": 2.0,
            "relegation_zone": 16
        },
        "colors": {
            "champions_league": "green!30",
            "europa_league": "blue!20",
            "conference_league": "cyan!15",
            "relegation": "red!25",
            "neutral": "gray!10",
            "positive": "green!50",
            "negative": "red!50",
            "warning": "yellow!40"
        },
        "output": {
            "directory": "output",
            "filename_pattern": "bundesliga_round_{round}"
        }
    }
    
    # Mentés
    import yaml
    with open("config_bundesliga.yaml", "w", encoding="utf-8") as f:
        yaml.dump(custom_config, f, default_flow_style=False, allow_unicode=True)
    
    print("✓ Egyedi Bundesliga konfig létrehozva")
    print("  Használat: LigaReportGenerator('config_bundesliga.yaml')")


def example_data_from_your_scraper():
    """
    Példa a saját scraper adatainak integrálására
    """
    print("\n" + "="*60)
    print("PÉLDA 3: Saját adatok integrálása")
    print("="*60 + "\n")
    
    print("""
    # A te scraper kódod (ezt te implementálod):
    
    from your_scraper import scrape_league_data
    
    # Adatok lekérése
    raw_data = scrape_league_data(
        league="Premier League",
        round_number=10
    )
    
    # Formázás a generátor formátumára
    formatted_data = {
        "league_name": raw_data["league"],
        "season": raw_data["season"],
        "round_number": raw_data["round"],
        "date_range": {
            "start": raw_data["start_date"],
            "end": raw_data["end_date"]
        },
        "teams": [
            {
                "team_id": team["id"],
                "team_name": team["name"],
                "position": team["rank"],
                "points": team["pts"],
                "xg_total": team["xg"],
                # ... stb
            }
            for team in raw_data["teams"]
        ],
        "matches": [
            # ... hasonló konverzió
        ]
    }
    
    # Mentés
    import json
    with open("data/current_round.json", "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    # Generálás
    from report_generator import LigaReportGenerator
    generator = LigaReportGenerator()
    generator.generate_report("data/current_round.json")
    """)


def example_batch_processing():
    """
    Több forduló egyszerre feldolgozása
    """
    print("\n" + "="*60)
    print("PÉLDA 4: Batch feldolgozás")
    print("="*60 + "\n")
    
    print("""
    # Több forduló egyszerre
    
    from report_generator import LigaReportGenerator
    from pathlib import Path
    
    generator = LigaReportGenerator()
    
    # Összes JSON fájl a data/ mappában
    for json_file in Path("data").glob("round_*.json"):
        print(f"Feldolgozás: {json_file}")
        try:
            pdf_path = generator.generate_report(str(json_file))
            print(f"  ✓ Kész: {pdf_path}")
        except Exception as e:
            print(f"  ✗ Hiba: {e}")
    """)


def main():
    """
    Főprogram - példák futtatása
    """
    print("\n" + "="*70)
    print(" "*20 + "LIGA RIPORT GENERÁTOR")
    print(" "*25 + "Példák")
    print("="*70)
    
    print("\nVálassz egy példát:")
    print("  1 - Alap használat (teljes demo)")
    print("  2 - Egyedi konfiguráció létrehozása")
    print("  3 - Saját adatok integrálása (leírás)")
    print("  4 - Batch feldolgozás (leírás)")
    print("  0 - Kilépés")
    
    choice = input("\nVálasztás (1-4): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_custom_config()
    elif choice == "3":
        example_data_from_your_scraper()
    elif choice == "4":
        example_batch_processing()
    elif choice == "0":
        print("\nViszlát!")
    else:
        print("\nÉrvénytelen választás!")


if __name__ == "__main__":
    main()