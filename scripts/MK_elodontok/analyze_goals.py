import pandas as pd

def analyze_goals():
    df = pd.read_csv("magyar_kupa_incidents.csv")
    
    # Filter only goals
    goals = df[df['type'] == 'goal'].copy()
    
    # Exclude extra time (91-120+) and penalty shootouts
    # Usually ET goals have time between 91 and 120
    # Penalty shootout goals usually have empty time or 'penaltyShootout' type (already filtered by type=='goal')
    goals = goals[goals['time'] <= 90]
    
    def get_interval(row):
        t = row['time']
        at = row['added_time']
        
        if pd.isna(at) or at == 0:
            if t <= 15: return "01-15"
            if t <= 30: return "16-30"
            if t < 45: return "31-45"
            if t == 45: return "31-45" # Can't be sure if it was 45' or 45'+ without added_time
            if t <= 60: return "46-60"
            if t <= 75: return "61-75"
            if t < 90: return "76-90"
            if t == 90: return "76-90"
        else:
            if t == 45: return "45+"
            if t == 90: return "90+"
        return "Unknown"

    goals['interval'] = goals.apply(get_interval, axis=1)
    
    # Special case: SofaScore sometimes puts time > 90 for added time if added_time is not used correctly
    # but with our enrichment it should be fine.
    
    stats = goals['interval'].value_counts().sort_index()
    
    print("--- Gólok eloszlása (15 perces szakaszok) ---")
    print(stats.to_string())
    
    # Percentages
    total = stats.sum()
    print(f"\nÖsszes gól (rendes játékidőben): {total}")
    for interval, count in stats.items():
        print(f"{interval}: {count} gól ({count/total*100:.1f}%)")

if __name__ == "__main__":
    analyze_goals()
