import os
import sys
import json
import time
import base64
import urllib.request
from typing import Optional, Union, List, Dict, Any
import websocket
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "modules")))
import SofaScore_module as ssm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def record_live_stream_via_cdp(
    match_id: Union[int, str] = 16939017,
    duration_seconds: int = 45,
    output_json: Optional[str] = None
) -> (pd.DataFrame, List[Dict[str, Any]]):
    """
    Connects to the active Chrome tab via Chrome DevTools Protocol (port 9222)
    and captures all live Sportradar & SofaScore WebSocket frames.
    """
    clean_id = str(match_id)
    print(f"Connecting to CDP at http://127.0.0.1:9222 for match {clean_id}...")
    
    resp = urllib.request.urlopen("http://127.0.0.1:9222/json")
    tabs = json.loads(resp.read().decode())
    target_tab = None
    for t in tabs:
        if clean_id in t.get("url", ""):
            target_tab = t
            break
            
    if not target_tab:
        raise RuntimeError(f"Tab for match {clean_id} not found in running Chrome!")
        
    ws_url = target_tab["webSocketDebuggerUrl"]
    print(f"Found active tab: {target_tab.get('title')}")
    print(f"Debugger URL: {ws_url}")
    
    ws = websocket.create_connection(ws_url, timeout=10, suppress_origin=True)
    
    # Enable Network tracking in CDP
    ws.send(json.dumps({"id": 1, "method": "Network.enable"}))
    
    recorded_events = []
    start_time = time.time()
    print(f"Recording live stream for {duration_seconds} seconds...")
    
    ws.settimeout(1.0)
    while time.time() - start_time < duration_seconds:
        try:
            msg = ws.recv()
            if not msg:
                continue
            data = json.loads(msg)
            method = data.get("method")
            if method == "Network.webSocketFrameReceived":
                payload = data.get("params", {}).get("response", {}).get("payloadData", "")
                ts = data.get("params", {}).get("timestamp", time.time())
                
                if payload:
                    try:
                        decoded_bytes = base64.b64decode(payload)
                        decoded_text = decoded_bytes.decode('utf-8', errors='ignore')
                    except Exception:
                        decoded_text = payload
                        
                    entry = {
                        "timestamp": ts,
                        "time_str": time.strftime("%H:%M:%S"),
                        "raw_payload": decoded_text
                    }
                    recorded_events.append(entry)
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as e:
            # print(f"Recv exception: {e}")
            continue
            
    ws.close()
    print(f"Capture finished. Total raw WebSocket frames recorded: {len(recorded_events)}")
    
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(recorded_events, f, ensure_ascii=False, indent=2)
        print(f"Saved raw frames to {output_json}")
        
    df = ssm.parse_live_stream_to_df(recorded_events)
    return df, recorded_events


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    json_path = os.path.join(DATA_DIR, "raw", f"live_stream_{match_id}.json")
    csv_path = os.path.join(DATA_DIR, "processed", f"live_events_{match_id}.csv")
    
    df, raw_frames = record_live_stream_via_cdp(
        match_id=match_id,
        duration_seconds=45,
        output_json=json_path
    )
    
    print("\n--- Parsed Live Event Summary ---")
    print(f"Total Unique Micro-Events: {len(df)}")
    if not df.empty:
        df.to_csv(csv_path, index=False)
        print(f"Saved DataFrame to {csv_path}")
        print("\nEvent Distribution:")
        print(df["name"].value_counts())
        print("\nSample Rows:")
        cols = [c for c in ["time_captured", "match_minute", "match_seconds", "name", "situation", "team", "x", "y"] if c in df.columns]
        print(df[cols].head(15).to_string())
    else:
        print("Note: Match might be currently in half-time break (szünet) or waiting for restart.")
