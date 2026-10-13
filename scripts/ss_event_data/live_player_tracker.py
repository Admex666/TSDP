import os
import sys
import json
import time
import urllib.request
from typing import Optional, Union, List, Dict, Any
import websocket
import pandas as pd

# Reconfigure stdout for utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class SofaScorePlayerTracker:
    """
    Advanced Player Tracking Extractor for SofaScore & Sportradar 3D Match Tracker.
    Connects via Chrome DevTools Protocol (CDP) to extract 22-player real-time coordinates
    from the Babylon.js 3D engine and Sportradar widget state.
    """

    def __init__(self, match_id: Union[int, str] = 16939017, cdp_port: int = 9222):
        self.match_id = str(match_id)
        self.cdp_port = cdp_port
        self.ws = None
        self.msg_id = 1
        self.contexts = []

    def connect(self, auto_launch: bool = False):
        """Connects to Chrome tab with matching match ID."""
        cdp_url = f"http://127.0.0.1:{self.cdp_port}/json"
        print(f"Connecting to Chrome CDP at {cdp_url} for match #{self.match_id}...")
        
        tabs = None
        try:
            resp = urllib.request.urlopen(cdp_url, timeout=3)
            tabs = json.loads(resp.read().decode())
        except Exception as e:
            raise RuntimeError(
                f"Nem sikerült csatlakozni a Chrome-hoz a {self.cdp_port}-es porton!\n"
                f"A böngészőt remote-debugging porttal kell indítani. Részletek alább."
            )

        target_tab = None
        for t in tabs:
            if self.match_id in t.get("url", "") or "sofascore.com" in t.get("url", ""):
                target_tab = t
                break

        if not target_tab:
            available_tabs = [f"'{t.get('title')}' ({t.get('url')})" for t in tabs if t.get('type') == 'page']
            raise RuntimeError(
                f"Target tab for match #{self.match_id} not found!\n"
                f"Available tabs: \n" + "\n".join(available_tabs)
            )

        ws_url = target_tab["webSocketDebuggerUrl"]
        print(f"Connected to tab: {target_tab.get('title')}")
        print(f"WebSocket URL: {ws_url}")
        self.ws = websocket.create_connection(ws_url, timeout=10, suppress_origin=True)

        # Enable Runtime and Page domains to access JavaScript contexts and iframes
        self.send_cdp("Runtime.enable")
        self.send_cdp("Page.enable")

    def send_cdp(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Sends command over CDP and returns response."""
        self.msg_id += 1
        payload = {"id": self.msg_id, "method": method, "params": params or {}}
        self.ws.send(json.dumps(payload))
        
        while True:
            raw = self.ws.recv()
            if not raw:
                continue
            data = json.loads(raw)
            if data.get("id") == self.msg_id:
                return data

    def eval_js(self, expression: str, context_id: Optional[int] = None) -> Any:
        """Evaluates JavaScript expression in tab or specific execution context."""
        params = {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True
        }
        if context_id is not None:
            params["contextId"] = context_id
            
        res = self.send_cdp("Runtime.evaluate", params)
        result_obj = res.get("result", {}).get("result", {})
        if "value" in result_obj:
            return result_obj["value"]
        if "description" in result_obj:
            return result_obj["description"]
        return res

    def inspect_player_tracking_environment(self) -> Dict[str, Any]:
        """
        Diagnoses Babylon.js scenes, Sportradar widget objects, and mesh hierarchies
        to locate the 22 player nodes.
        """
        print("\n--- Inspecting 3D Engine & Player Meshes ---")
        
        diag_script = """
        (() => {
            const result = {
                hasBabylon: typeof BABYLON !== 'undefined',
                babylonVersion: typeof BABYLON !== 'undefined' ? BABYLON.Engine.Version : null,
                engineCount: typeof BABYLON !== 'undefined' ? BABYLON.Engine.Instances.length : 0,
                canvases: Array.from(document.querySelectorAll('canvas')).map(c => ({
                    id: c.id,
                    className: c.className,
                    width: c.width,
                    height: c.height,
                    engine: c.getAttribute('data-engine')
                })),
                iframes: Array.from(document.querySelectorAll('iframe')).map(f => ({
                    id: f.id,
                    src: f.src,
                    name: f.name
                })),
                srGlobals: Object.keys(window).filter(k => k.toLowerCase().includes('sportradar') || k.toLowerCase().includes('lmt') || k.toLowerCase().includes('tracker')),
                meshes: []
            };

            // Inspect Babylon.js scene meshes if available
            if (result.hasBabylon && BABYLON.Engine.Instances.length > 0) {
                const engine = BABYLON.Engine.Instances[0];
                if (engine.scenes && engine.scenes.length > 0) {
                    const scene = engine.scenes[0];
                    result.sceneCount = engine.scenes.length;
                    result.meshCount = scene.meshes.length;
                    result.transformNodeCount = scene.transformNodes ? scene.transformNodes.length : 0;
                    
                    // Look for player meshes / nodes
                    const nodes = [...scene.meshes, ...(scene.transformNodes || [])];
                    const playerCandidates = nodes.filter(m => {
                        const n = (m.name || '').toLowerCase();
                        return n.includes('player') || n.includes('jersey') || n.includes('team') || n.includes('home') || n.includes('away');
                    });
                    
                    result.playerCandidatesCount = playerCandidates.length;
                    result.sampleCandidateNames = playerCandidates.slice(0, 30).map(m => m.name);
                }
            }
            return result;
        })()
        """
        diag_data = self.eval_js(diag_script)
        print("Inspection Results:")
        print(json.dumps(diag_data, indent=2))
        return diag_data

    def record_22_players(
        self,
        duration_seconds: int = 30,
        interval_seconds: float = 0.5,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Continuously extracts 22 player positions and ball coordinates at high frequency.
        """
        print(f"\nRecording 22 players tracking for {duration_seconds}s (interval: {interval_seconds}s)...")
        
        extract_players_script = """
        (() => {
            const snapshot = {
                timestamp: Date.now(),
                time_str: new Date().toLocaleTimeString(),
                players: [],
                ball: null
            };

            if (typeof BABYLON === 'undefined' || BABYLON.Engine.Instances.length === 0) {
                return snapshot;
            }

            const scene = BABYLON.Engine.Instances[0].scenes[0];
            if (!scene) return snapshot;

            const allNodes = [...scene.meshes, ...(scene.transformNodes || [])];

            for (const node of allNodes) {
                const name = (node.name || '');
                const low = name.toLowerCase();

                // Ball tracking
                if (low.includes('ball') && node.position) {
                    snapshot.ball = {
                        name: name,
                        x: node.position.x,
                        y: node.position.y,
                        z: node.position.z
                    };
                }

                // Player tracking candidates
                if ((low.includes('player') || low.includes('jersey') || low.includes('shirt')) && node.position) {
                    const team = low.includes('away') ? 'away' : (low.includes('home') ? 'home' : 'unknown');
                    snapshot.players.push({
                        name: name,
                        team: team,
                        x: node.position.x,
                        y: node.position.y,
                        z: node.position.z,
                        enabled: node.isEnabled ? node.isEnabled() : true
                    });
                }
            }
            return snapshot;
        })()
        """

        records = []
        start_time = time.time()
        step = 0

        while time.time() - start_time < duration_seconds:
            try:
                snap = self.eval_js(extract_players_script)
                if isinstance(snap, dict):
                    t_str = snap.get("time_str", "")
                    ts = snap.get("timestamp", time.time())
                    ball_info = snap.get("ball") or {}
                    
                    players = snap.get("players", [])
                    for p in players:
                        records.append({
                            "step": step,
                            "timestamp": ts,
                            "time_str": t_str,
                            "player_name": p.get("name"),
                            "team": p.get("team"),
                            "x": p.get("x"),
                            "y": p.get("y"),
                            "z": p.get("z"),
                            "ball_x": ball_info.get("x"),
                            "ball_y": ball_info.get("y"),
                            "ball_z": ball_info.get("z")
                        })
                step += 1
                time.sleep(interval_seconds)
            except Exception as e:
                print(f"Sampling exception: {e}")
                time.sleep(1)

        df = pd.DataFrame(records)
        print(f"\nCaptured {len(df)} player position snapshots across {step} sampling steps.")

        if output_csv and not df.empty:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"Saved 22-player tracking dataset to: {output_csv}")

        return df

    def close(self):
        if self.ws:
            self.ws.close()


if __name__ == "__main__":
    match_id = 16939017
    out_csv = os.path.join(DATA_DIR, "processed", f"player_tracking_3d_{match_id}.csv")
    
    tracker = SofaScorePlayerTracker(match_id=match_id)
    try:
        tracker.connect()
        diag = tracker.inspect_player_tracking_environment()
        
        # If Babylon/canvas is detected, start recording
        df_players = tracker.record_22_players(
            duration_seconds=20,
            interval_seconds=0.5,
            output_csv=out_csv
        )
        if not df_players.empty:
            print("\nSample Tracked Players:")
            print(df_players.head(10).to_string())
    except Exception as err:
        print(f"\n[Player Tracker Status]: {err}")
    finally:
        tracker.close()
