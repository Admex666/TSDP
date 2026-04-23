import sys
import os
import json

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.SofaScore_module import scrape_sofascore

def test_incidents():
    event_id = 14561741 # Szarvaskend vs Ferencváros
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/incidents"
    print(f"Fetching incidents for event {event_id}...")
    data = scrape_sofascore(url)
    
    if data and 'incidents' in data:
        print(f"Total incidents found: {len(data['incidents'])}")
        # Print first goal incident if exists
        for inc in data['incidents']:
            if inc.get('incidentType') == 'goal':
                print("\nExample Goal Incident:")
                print(json.dumps(inc, indent=2))
                break
    else:
        print("No incidents found or error fetching.")

if __name__ == "__main__":
    test_incidents()
