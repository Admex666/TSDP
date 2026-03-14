# Import sofascore module from E:\Data\TSDP\modules

import sys
sys.path.append(r"E:\Data\TSDP\modules")

from SofaScore_module import *

# read top40_leagues.json 
with open("top40_leagues.json", "r") as f:
    top40_leagues = json.load(f)

league_id = top40_leagues[0]['id']
print(league_id)

url_s = f"https://www.sofascore.com/api/v1/unique-tournament/{league_id}/seasons"
seasons = scrape_sofascore(url_s)
season_id = seasons['seasons'][1]['id']
print(season_id)

url = f"https://www.sofascore.com/api/v1/unique-tournament/{league_id}/season/{season_id}/rounds"
rounds = scrape_sofascore(url)
#print(rounds)
round_nr = rounds['rounds'][0]['round']

url2 = f"https://www.sofascore.com/api/v1/unique-tournament/{league_id}/season/{season_id}/events/round/{round_nr}"
round = scrape_sofascore(url2)
#print(round)

# save json
#with open(f"round_{round_nr}.json", "w") as f:
#    json.dump(round, f, indent=4)

for event in round['events']:
    # check if finished
    if event['status']['type'] != 'finished':
        continue

    # team names with scores
    print(f"{event['homeTeam']['name']} {event['homeScore']['display']} - {event['awayScore']['display']} {event['awayTeam']['name']} ({event['id']}, {event['startTimestamp']})")