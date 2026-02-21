# Feature Engineering Specification

This document outlines the features to be extracted and engineered for the ML model, mapping raw data to predictive variables.

## 1. Raw Variables from Data Sources

### A. Race Data (Historical & Today)
- `distance`: Float (e.g., 1800 from "1800A")
- `start_type`: Categorical ("A" for Auto-start, "G" for Standing start)
- `track_quality`: Categorical (e.g., "jó", "sáros")
- `temperature`: Float (°C)
- `prize_total`: Float (Total prize money of the race)
- `horse_id`, `driver_id`: Identifiers for joining

### B. Participant Career Stats (JSON)
- `career_yearly_stats`: List of Dicts (Wins, places, earnings per year)
- `form_history`: List of Dicts (Last 5-10 races, placements, km_times)

## 2. Engineered Features (Calculated per Horse/Driver)

### A. Horse Performance Metrics
- `horse_win_rate_lifetime`: Total wins / Total runs
- `horse_avg_km_time_12m`: Average Km Time over the last 12 months
- `horse_best_km_time_dist`: Best Km Time at the current race distance
- `horse_galopp_rate`: Frequency of disqualification (rank == "gal.")
- `horse_days_since_last_race`: Rest period (fatigue/readiness)

### B. Driver/Jockey Skills
- `driver_win_rate_lifetime`: Driver's overall success rate
- `driver_top3_rate_12m`: Percent of races finishing in 1st, 2nd, or 3rd lately
- `driver_experience`: Total number of career starts

### C. Contextual Features (The "Fit")
- `horse_distance_suitability`: Deviation of current distance from horse's "ideal" distance
- `horse_driver_synergy`: Historical success rate of this specific Horse+Driver pair
- `speed_vs_field`: `horse_avg_km_time` vs average of all other participants in the same race

## 3. The Target Variable (Label)
- `win`: Binary (1 if rank == "I.", else 0)
- `placed`: Binary (1 if rank in ["I.", "II.", "III."], else 0) - For broader safety

## 4. Normalization Strategy
- **Z-Score Scaling:** Apply to Km Times and Distance (within each race).
- **One-Hot Encoding:** For track quality and start type.
