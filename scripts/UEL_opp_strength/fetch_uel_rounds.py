import sys
import os
import json

# Add the project root to sys.path to allow importing from modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

try:
    from modules.SofaScore_module import scrape_sofascore
except ImportError:
    print(f"Error: Could not import SofaScore_module from {project_root}")
    sys.exit(1)

def main():
    tournament_id = 679
    season_id = 76984
    rounds = range(1, 9)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    for round_nr in rounds:
        print(f"Fetching round {round_nr}...")
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_nr}"
        
        data = scrape_sofascore(url)
        
        if data:
            filename = os.path.join(output_dir, f"uel_round_{round_nr}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Saved round {round_nr} to {filename}")
        else:
            print(f"Failed to fetch round {round_nr}")

if __name__ == "__main__":
    main()
