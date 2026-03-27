import json
import os
import pandas as pd

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    all_events = []
    
    # Iterate through all round files
    for i in range(1, 9):
        filename = os.path.join(directory, f"uel_round_{i}.json")
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found.")
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        events = data.get('events', [])
        for event in events:
            # Flatten some fields
            event_data = {
                'round': event.get('roundInfo', {}).get('round'),
                'event_id': event.get('id'),
                'startTimestamp': event.get('startTimestamp'),
                'homeTeam_name': event.get('homeTeam', {}).get('name'),
                'homeTeam_id': event.get('homeTeam', {}).get('id'),
                'awayTeam_name': event.get('awayTeam', {}).get('name'),
                'awayTeam_id': event.get('awayTeam', {}).get('id'),
                'homeScore': event.get('homeScore', {}).get('current'),
                'awayScore': event.get('awayScore', {}).get('current'),
                'status_type': event.get('status', {}).get('type'),
                'status_desc': event.get('status', {}).get('description'),
                'winnerCode': event.get('winnerCode')
            }
            all_events.append(event_data)
    
    if all_events:
        df = pd.DataFrame(all_events)
        output_file = os.path.join(directory, "uel_2025_26_rounds_1-8.csv")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Successfully saved {len(all_events)} events to {output_file}")
    else:
        print("No events found to process.")

if __name__ == "__main__":
    main()
