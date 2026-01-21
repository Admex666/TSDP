import sys
import os

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'modules'))

from SofaScore_module import scrape_sofascore
import json

def test_sofascore_api():
    """
    Test the SofaScore API by fetching events for rounds 1-7
    Tournament ID: 7 (likely a major league)
    Season ID: 76953
    """
    
    base_url = "https://www.sofascore.com/api/v1/unique-tournament/7/season/76953/events/round/{round_nr}"
    
    results = {}
    
    print("=" * 60)
    print("Testing SofaScore API with enhanced anti-403 protection")
    print("=" * 60)
    print()
    
    for round_nr in range(1, 8):
        print(f"\n{'='*60}")
        print(f"Fetching Round {round_nr}...")
        print(f"{'='*60}")
        
        url = base_url.format(round_nr=round_nr)
        print(f"URL: {url}")
        
        # Call the scrape function
        data = scrape_sofascore(url)
        
        if data:
            # Store results
            results[f"round_{round_nr}"] = data
            
            # Print summary
            if 'events' in data:
                num_events = len(data['events'])
                print(f"✓ Success! Found {num_events} events in round {round_nr}")
                
                # Print first event details if available
                if num_events > 0:
                    first_event = data['events'][0]
                    home_team = first_event.get('homeTeam', {}).get('name', 'Unknown')
                    away_team = first_event.get('awayTeam', {}).get('name', 'Unknown')
                    print(f"  Example match: {home_team} vs {away_team}")
            else:
                print(f"✓ Success! Data received but no 'events' key found")
                print(f"  Keys in response: {list(data.keys())}")
        else:
            print(f"✗ Failed to fetch data for round {round_nr}")
            results[f"round_{round_nr}"] = None
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for v in results.values() if v is not None)
    failed = len(results) - successful
    
    print(f"Total rounds tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    # Save results to JSON file
    output_file = os.path.join(os.path.dirname(__file__), 'test_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    test_sofascore_api()
