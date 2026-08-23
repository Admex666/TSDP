import os
import sys
from typing import Optional, Union
import pandas as pd

# Path setup to import SofaScore_module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "modules")))
import SofaScore_module as ssm


def scrape_live_events(
    match_id_or_url: Union[int, str],
    duration_seconds: int = 45,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Scrapes live WebSocket match stream and returns a parsed event-level DataFrame
    using SofaScore_module functions.
    
    Args:
        match_id_or_url: SofaScore match ID (e.g. 16483643) or full match URL
        duration_seconds: Duration of live stream capture in seconds
        output_json: Optional path to save raw WebSocket JSON stream
        output_csv: Optional path to save parsed DataFrame CSV
        
    Returns:
        pd.DataFrame containing live micro-events, coordinates, and situations.
    """
    print(f"Connecting to live stream for match: {match_id_or_url} ({duration_seconds}s)...")
    df = ssm.fetch_live_match_events(
        event_id=match_id_or_url,
        duration_seconds=duration_seconds,
        output_file=output_json
    )
    
    print(f"Captured {len(df)} unique live micro-events.")
    
    if output_csv and not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Saved parsed event DataFrame to {output_csv}")
        
    return df


if __name__ == "__main__":
    match_id = 16483643
    
    df_events = scrape_live_events(
        match_id_or_url=match_id,
        duration_seconds=45,
        output_json=f"live_stream_{match_id}.json",
        output_csv=f"live_events_{match_id}.csv"
    )
    
    if not df_events.empty:
        print("\n--- Event Distribution ---")
        print(df_events["name"].value_counts())
        print("\n--- Sample Events ---")
        display_cols = [c for c in ["time_captured", "match_minute", "match_seconds", "name", "situation", "team", "x", "y"] if c in df_events.columns]
        print(df_events[display_cols].head(10).to_string())
