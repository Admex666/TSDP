import pandas as pd
import time

# ---- Read the event file ----
"""
# measure the time of reading the file
start_time = time.time()
event = pd.read_excel("00_event.xlsx")
end_time = time.time()
print(f"Time taken to read the file: {end_time - start_time:.0f}s ({len(event)/1000:.0f}k rows)")
print(f"Time to read 10k rows: {(end_time - start_time)/len(event) * 10_000:.0f}s")

goals = event[event.event_name == "Goal"]
print(f"Goals: {len(goals)}")

goals.to_excel("00_goals.xlsx", index=False)

print(f"Saved {len(goals)} goals to 00_goals.xlsx")
"""

# ---- Calculate goal differences ----

goals = pd.read_excel("00_goals.xlsx")
# only this seasn
goals = goals[goals.season_name == "2025/2026"]
goals.sort_values(by=['match_date', 'match_id', 'match_half_id', 'time_min', 'time_sec'], inplace=True)
goals = goals.reset_index(drop=True)

display_cols = ['match_date', 'match_half_id', 'time_min', 'time_sec', 'team_name', 'opponent_name']
#print(goals[display_cols].head(10))

# calculate goals for current state

# 1. Hányadik gólnál tart az a csapat, amelyik épp lőtt?
goals['team_score'] = goals.groupby(['match_id', 'team_name']).cumcount() + 1

# 2. Hányadik gól ez összesen a mérkőzésen?
goals['match_goal_number'] = goals.groupby('match_id').cumcount() + 1

# 3. Mennyi volt az ellenfél állása ebben a pillanatban?
# (Összes gól - amit ez a csapat lőtt)
goals['opponent_score'] = goals['match_goal_number'] - goals['team_score']

# 4. Aktuális gólkülönbség (a gólszerző szemszögéből)
# Ha pozitív, vezetnek, ha negatív, még hátrányban vannak (pl. szépítés 0-2-ről 1-2-re)
goals['score_diff'] = goals['team_score'] - goals['opponent_score']

# 5. Abszolút gólkülönbség (ez kell a Garbage Time-hoz)
goals['abs_margin'] = goals['score_diff'].abs()

# Ellenőrzésképp nézzünk bele
display_cols_with_score = display_cols + ['team_score', 'opponent_score', 'score_diff']
#print(goals[display_cols_with_score].head(20))

# ---- Calculate garbage time ----

# 4+ @ 45, 3+ @ 60, 2+ @ 87

goals['garbage_time'] = ((goals.time_min >= 45) & ((goals.score_diff >= 4) | (goals.score_diff <= -4))) | \
                       ((goals.time_min >= 60) & ((goals.score_diff >= 3) | (goals.score_diff <= -3))) | \
                       ((goals.time_min >= 87) & ((goals.score_diff >= 2) | (goals.score_diff <= -2)))

#print(goals[goals.garbage_time == True][display_cols_with_score].head(10))
print(f"Goals in garbage time: {len(goals[goals.garbage_time == True])}/{len(goals)} ({len(goals[goals.garbage_time == True])/len(goals) * 100:.2f}%)")


# ---- Garbage time per team ----

# 1. Idő átváltása másodpercre a pontos számításhoz
goals['total_sec'] = goals.time_min * 60 + goals.time_sec

# 2. Következő gól idejének lekérése meccsenként
# Ha nincs több gól, a meccs végét (90:00 = 5400 mp) vesszük alapul
goals['next_goal_sec'] = goals.groupby('match_id')['total_sec'].shift(-1).fillna(90 * 60)

# 3. GT küszöbök meghatározása (másodpercben) az aktuális gólkülönbséghez
def get_gt_threshold(margin):
    abs_margin = abs(margin)
    if abs_margin >= 4: return 45 * 60
    if abs_margin >= 3: return 60 * 60
    if abs_margin >= 2: return 87 * 60
    return 999999 # Sosem lesz GT ennél a különbségnél

goals['gt_threshold_sec'] = goals['abs_margin'].apply(get_gt_threshold)

# 4. A Garbage Time kezdete ebben az intervallumban: 
# vagy a gól pillanata, vagy a küszöb elérése (amelyik később van)
goals['interval_gt_start'] = goals[['total_sec', 'gt_threshold_sec']].max(axis=1)

# 5. GT hossza az adott gól és a következő esemény (gól/vége) között
# (Ha a következő gól hamarabb van, mint a küszöb, akkor 0 lesz)
goals['gt_duration_sec'] = (goals['next_goal_sec'] - goals['interval_gt_start']).clip(lower=0)

# 6. Összesítés csapatonként
# Mivel minden gól sorban van, a meccseket a hazai és vendég csapat szerint is összegezzük
match_gt = goals.groupby(['match_id', 'team_name', 'opponent_name'])['gt_duration_sec'].sum().reset_index()

# Kigyűjtjük mindkét csapatot minden meccsről
teams_a = match_gt[['team_name', 'gt_duration_sec']].rename(columns={'team_name': 'team'})
teams_b = match_gt[['opponent_name', 'gt_duration_sec']].rename(columns={'opponent_name': 'team'})

team_totals = pd.concat([teams_a, teams_b]).groupby('team')['gt_duration_sec'].sum() / 60
team_totals = team_totals.sort_values(ascending=False).round(1)

print("\n--- Garbage Time per Team (minutes) ---")
print(team_totals.head(10))

# Egy példa meccs ellenőrzése
example_match = goals[goals.match_id == goals.match_id.iloc[0]]
# print(example_match[['time_min', 'abs_margin', 'gt_duration_sec']])
