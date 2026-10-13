import os
import sys
import json
import ast
from typing import Optional, Union, List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mplsoccer import Pitch

# Reconfigure stdout for utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def parse_event_data(csv_path: str) -> pd.DataFrame:
    """Load and parse event data CSV, normalizing trajectory points and situations."""
    df = pd.read_csv(csv_path)
    
    # Process trajectory_points if present
    parsed_trajectories = []
    for _, row in df.iterrows():
        traj_raw = row.get("trajectory_points")
        if pd.notna(traj_raw):
            try:
                if isinstance(traj_raw, str):
                    traj = ast.literal_eval(traj_raw)
                else:
                    traj = traj_raw
                parsed_trajectories.append(traj)
            except Exception:
                parsed_trajectories.append(None)
        else:
            parsed_trajectories.append(None)
            
    df["parsed_trajectory"] = parsed_trajectories
    return df


def generate_static_pitch_visualization(
    csv_path: str,
    output_png: str,
    match_title: str = "Live Tracking Visualizer"
) -> str:
    """
    Generates a high-quality pitch visualization of the tracking data using mplsoccer and matplotlib.
    """
    df = parse_event_data(csv_path)
    if df.empty:
        raise ValueError(f"No events found in {csv_path}")

    # Create Dark-themed Custom Pitch (100x100 coordinate system matching SofaScore/Sportradar)
    pitch = Pitch(
        pitch_type='custom',
        pitch_length=100,
        pitch_width=100,
        pitch_color='#0f172a',
        line_color='#334155',
        line_zorder=2,
        linewidth=1.8,
        goal_type='box',
        goal_alpha=0.6
    )

    fig, ax = pitch.draw(figsize=(14, 9.5))
    fig.patch.set_facecolor('#0b0f19')

    # Color palette for tactical situations
    color_map = {
        'safe': '#38bdf8',       # Sky blue - buildup / secure phase
        'attack': '#f97316',     # Vibrant orange - attacking third
        'dangerous': '#ef4444',  # Red - dangerous box penetration
        'default': '#94a3b8'     # Slate - neutral
    }

    # Filter events with valid coordinates
    coord_events = df[df['x'].notna() & df['y'].notna()].copy().reset_index(drop=True)
    
    # Extract unique match situation events for annotations
    sit_map = {}
    last_sit = "safe"
    for _, r in df.iterrows():
        if pd.notna(r.get("situation")):
            last_sit = str(r["situation"]).lower()
        if pd.notna(r.get("x")) and pd.notna(r.get("y")):
            ev_id = r.get("event_id")
            sit_map[ev_id] = last_sit

    # Draw trajectory lines and arrows
    step_num = 1
    plotted_points = []

    for idx, row in coord_events.iterrows():
        x = float(row['x'])
        y = float(row['y'])
        ev_id = row.get('event_id')
        sit = sit_map.get(ev_id, 'default')
        color = color_map.get(sit, color_map['default'])
        
        traj = row.get('parsed_trajectory')
        if traj and isinstance(traj, list) and len(traj) >= 2:
            # Trajectory points are ordered: [latest, ..., earliest] or [start, ..., end]
            # In Sportradar payload, coords are chronological reverse or direct.
            t_pts = [(float(pt['X']), float(pt['Y'])) for pt in traj if 'X' in pt and 'Y' in pt]
            if len(t_pts) >= 2:
                # Reverse to get chronological sequence: earliest -> latest
                t_pts_chrono = list(reversed(t_pts))
                xs = [p[0] for p in t_pts_chrono]
                ys = [p[1] for p in t_pts_chrono]
                
                # Draw path curve with alpha
                ax.plot(xs, ys, color=color, alpha=0.55, linewidth=2.5, linestyle='--', zorder=3)
                
                # Draw intermediate flow arrows
                for i in range(len(xs) - 1):
                    dx = xs[i+1] - xs[i]
                    dy = ys[i+1] - ys[i]
                    if abs(dx) > 1 or abs(dy) > 1:
                        ax.annotate(
                            '', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                            arrowprops=dict(
                                arrowstyle="->", color=color, lw=1.6, alpha=0.7,
                                mutation_scale=14
                            ),
                            zorder=3
                        )
        elif pd.notna(row.get('start_x')) and pd.notna(row.get('start_y')):
            sx = float(row['start_x'])
            sy = float(row['start_y'])
            ax.annotate(
                '', xy=(x, y), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=2.2, alpha=0.8,
                    mutation_scale=16
                ),
                zorder=3
            )

        # Plot ball location point
        ax.scatter(
            x, y,
            s=160,
            color=color,
            edgecolors='#ffffff',
            linewidths=2,
            zorder=4
        )
        
        # Annotation badge with Step number
        time_lbl = row.get('time_captured', '')
        time_part = f" ({time_lbl})" if pd.notna(time_lbl) and time_lbl else ""
        badge_text = f"#{step_num}{time_part}\n[{sit.upper()}]"
        
        ax.text(
            x + 1.2, y + 1.2,
            badge_text,
            color='#f8fafc',
            fontsize=8.5,
            fontweight='semibold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e293b', edgecolor=color, alpha=0.85, lw=1.2),
            zorder=5
        )
        
        plotted_points.append({'step': step_num, 'x': x, 'y': y, 'sit': sit, 'color': color})
        step_num += 1

    # Connect major sequence progression
    if len(plotted_points) > 1:
        seq_xs = [p['x'] for p in plotted_points]
        seq_ys = [p['y'] for p in plotted_points]
        ax.plot(seq_xs, seq_ys, color='#64748b', linestyle=':', linewidth=1.5, alpha=0.4, zorder=2)

    # Attack direction indicator
    ax.annotate(
        '', xy=(65, -2), xytext=(35, -2),
        arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2, mutation_scale=15)
    )
    ax.text(50, -2.2, "Attacking Direction (Home) →", color="#94a3b8", fontsize=10, ha='center', va='top')

    # Title & Metadata Card
    fig.text(
        0.12, 0.95,
        match_title,
        color='#f8fafc',
        fontsize=18,
        fontweight='bold'
    )
    
    meta_info = f"Tracked Events: {len(coord_events)}  |  Scale: 100x100 Pitch Coordinates  |  Engine: Sportradar / SofaScore 3D Stream"
    fig.text(
        0.12, 0.92,
        meta_info,
        color='#94a3b8',
        fontsize=11
    )

    # Legend
    legend_elements = [
        patches.Patch(facecolor=color_map['safe'], edgecolor='#ffffff', label='Safe Phase (Buildup)'),
        patches.Patch(facecolor=color_map['attack'], edgecolor='#ffffff', label='Attack Phase (Forward Progression)'),
        patches.Patch(facecolor=color_map['dangerous'], edgecolor='#ffffff', label='Dangerous (Penalty Box Area)')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        facecolor='#1e293b',
        edgecolor='#475569',
        labelcolor='#f8fafc',
        fontsize=9.5,
        framealpha=0.9
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"Static pitch image successfully saved to: {output_png}")
    return output_png


def generate_interactive_html_visualizer(
    csv_path: str,
    output_html: str,
    match_id: Union[int, str] = 16939017,
    match_title: str = "Sabah FK vs SK Slavia Praha"
) -> str:
    """
    Generates a standalone, interactive HTML5 + JS pitch player for tracking coordinates.
    Includes playback controls, scrubbing timeline, real-time ball trajectories, and event inspector.
    """
    df = parse_event_data(csv_path)
    
    # Clean and structure events list for JS embedding
    events_payload = []
    for idx, r in df.iterrows():
        ev = {
            "index": idx + 1,
            "time_captured": str(r.get("time_captured", "")),
            "match_minute": float(r["match_minute"]) if pd.notna(r.get("match_minute")) else None,
            "match_seconds": float(r["match_seconds"]) if pd.notna(r.get("match_seconds")) else None,
            "type": str(r.get("type", "")),
            "name": str(r.get("name", "")),
            "situation": str(r.get("situation", "safe")),
            "team": str(r.get("team", "home")),
            "x": float(r["x"]) if pd.notna(r.get("x")) else None,
            "y": float(r["y"]) if pd.notna(r.get("y")) else None,
            "start_x": float(r["start_x"]) if pd.notna(r.get("start_x")) else None,
            "start_y": float(r["start_y"]) if pd.notna(r.get("start_y")) else None,
            "trajectory": r.get("parsed_trajectory") if r.get("parsed_trajectory") else [],
            "event_id": str(r.get("event_id", ""))
        }
        events_payload.append(ev)

    events_json = json.dumps(events_payload, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{match_title} | 3D Live Match Tracking Visualizer</title>
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-card: #151d2f;
            --bg-panel: #1b253b;
            --border-color: #27354f;
            --accent-safe: #38bdf8;
            --accent-attack: #f97316;
            --accent-dangerous: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --pitch-green: #113824;
            --pitch-line: rgba(255, 255, 255, 0.45);
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: linear-gradient(180deg, #182238 0%, var(--bg-primary) 100%);
            padding: 16px 28px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-title h1 {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.01em;
        }}
        .header-title p {{
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-top: 3px;
        }}
        .badge {{
            padding: 5px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .badge-live {{
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.4);
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .badge-live::before {{
            content: '';
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #ef4444;
            display: inline-block;
            box-shadow: 0 0 8px #ef4444;
        }}
        .main-container {{
            display: grid;
            grid-template-columns: 1fr 360px;
            gap: 20px;
            padding: 20px 28px;
            flex: 1;
        }}
        @media (max-width: 1100px) {{
            .main-container {{
                grid-template-columns: 1fr;
            }}
        }}
        .pitch-card {{
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }}
        .pitch-container {{
            position: relative;
            width: 100%;
            aspect-ratio: 105 / 68;
            background: var(--pitch-green);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: inset 0 0 60px rgba(0,0,0,0.5);
            border: 2px solid rgba(255,255,255,0.15);
        }}
        svg.pitch-svg {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        .controls-card {{
            background: var(--bg-panel);
            border-radius: 10px;
            padding: 14px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            border: 1px solid var(--border-color);
        }}
        .btn-group {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        button.btn {{
            background: #25334e;
            color: #ffffff;
            border: 1px solid #374b6e;
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 0.85rem;
            cursor: pointer;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }}
        button.btn:hover {{
            background: #334568;
            border-color: var(--accent-safe);
        }}
        button.btn:active {{
            transform: scale(0.97);
        }}
        .timeline-slider {{
            flex: 1;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        input[type="range"] {{
            flex: 1;
            accent-color: var(--accent-safe);
            height: 6px;
            cursor: pointer;
        }}
        .sidebar {{
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            height: calc(100vh - 140px);
        }}
        .sidebar-header {{
            padding: 14px 18px;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-panel);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .sidebar-header h2 {{
            font-size: 0.95rem;
            font-weight: 600;
        }}
        .event-list {{
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .event-item {{
            background: #172136;
            border: 1px solid #23324d;
            border-radius: 8px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .event-item:hover {{
            background: #1e2c47;
            border-color: #38bdf8;
        }}
        .event-item.active {{
            background: #1a2a46;
            border-color: var(--accent-safe);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.25);
        }}
        .event-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .event-time {{
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--text-muted);
        }}
        .pill {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .pill-safe {{ background: rgba(56, 189, 248, 0.18); color: var(--accent-safe); }}
        .pill-attack {{ background: rgba(249, 115, 22, 0.18); color: var(--accent-attack); }}
        .pill-dangerous {{ background: rgba(239, 68, 68, 0.18); color: var(--accent-dangerous); }}
        .event-body {{
            font-size: 0.78rem;
            color: #cbd5e1;
            display: flex;
            justify-content: space-between;
        }}
        .legend-row {{
            display: flex;
            gap: 16px;
            font-size: 0.75rem;
            color: var(--text-muted);
            align-items: center;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
        }}
        /* Animation elements on SVG */
        .ball-pulse {{
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0% {{ r: 2; opacity: 1; }}
            100% {{ r: 6; opacity: 0; }}
        }}
    </style>
</head>
<body>

<header>
    <div class="header-title">
        <h1>{match_title} (ID: {match_id})</h1>
        <p>Live Micro-Events & 3D Tracking Trajectories (SofaScore / Sportradar)</p>
    </div>
    <div class="badge badge-live">Live Stream Data</div>
</header>

<div class="main-container">
    <div class="pitch-card">
        <div class="pitch-container" id="pitchContainer">
            <!-- Full Pitch SVG (100x100 viewBox matching normalized coordinates) -->
            <svg class="pitch-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                <!-- Lawn stripes -->
                <defs>
                    <pattern id="grassStripes" width="10" height="100" patternUnits="userSpaceOnUse">
                        <rect width="5" height="100" fill="#143e28" />
                        <rect x="5" width="5" height="100" fill="#0f3421" />
                    </pattern>
                </defs>
                <rect width="100" height="100" fill="url(#grassStripes)" />

                <!-- Pitch markings -->
                <!-- Outer Border -->
                <rect x="3" y="3" width="94" height="94" fill="none" stroke="var(--pitch-line)" stroke-width="0.8" />
                
                <!-- Halfway line -->
                <line x1="50" y1="3" x2="50" y2="97" stroke="var(--pitch-line)" stroke-width="0.8" />
                
                <!-- Center circle -->
                <circle cx="50" cy="50" r="10" fill="none" stroke="var(--pitch-line)" stroke-width="0.8" />
                <circle cx="50" cy="50" r="0.6" fill="var(--pitch-line)" />
                
                <!-- Home Penalty Area (Left) -->
                <rect x="3" y="24" width="16" height="52" fill="none" stroke="var(--pitch-line)" stroke-width="0.8" />
                <rect x="3" y="37" width="5.5" height="26" fill="none" stroke="var(--pitch-line)" stroke-width="0.8" />
                <circle cx="11" cy="50" r="0.6" fill="var(--pitch-line)" />
                <path d="M 19 42 A 9 9 0 0 1 19 58" fill="none" stroke="var(--pitch-line)" stroke-width="0.8" />
                <!-- Home Goal -->
                <rect x="0.8" y="44" width="2.2" height="12" fill="rgba(255,255,255,0.2)" stroke="var(--pitch-line)" stroke-width="0.8" />

                <!-- Away Penalty Area (Right) -->
                <rect x="81" y="24" width="16" height="52" fill="none" stroke="var(--pitch-line)" stroke-width="0.8" />
                <rect x="91.5" y="37" width="5.5" height="26" fill="none" stroke="var(--pitch-line)" stroke-width="0.8" />
                <circle cx="89" cy="50" r="0.6" fill="var(--pitch-line)" />
                <path d="M 81 42 A 9 9 0 0 0 81 58" fill="none" stroke="var(--pitch-line)" stroke-width="0.8" />
                <!-- Away Goal -->
                <rect x="97" y="44" width="2.2" height="12" fill="rgba(255,255,255,0.2)" stroke="var(--pitch-line)" stroke-width="0.8" />

                <!-- Dynamic Layer for Trajectories & Points -->
                <g id="trajectoryLayer"></g>
                <g id="playersLayer"></g>
                <g id="ballLayer"></g>
            </svg>
        </div>

        <div class="controls-card">
            <div class="btn-group">
                <button class="btn" id="btnPrev">⏮ Prev</button>
                <button class="btn" id="btnPlay">▶ Play</button>
                <button class="btn" id="btnNext">Next ⏭</button>
            </div>
            <div class="timeline-slider">
                <span id="currentTime" style="font-size: 0.85rem; font-weight: 600; min-width: 60px;">--:--</span>
                <input type="range" id="eventSlider" min="0" max="0" value="0">
                <span id="stepCounter" style="font-size: 0.85rem; color: var(--text-muted); min-width: 50px;">0 / 0</span>
            </div>
            <div class="legend-row">
                <div class="legend-item"><span class="dot" style="background: var(--accent-safe)"></span> Safe</div>
                <div class="legend-item"><span class="dot" style="background: var(--accent-attack)"></span> Attack</div>
                <div class="legend-item"><span class="dot" style="background: var(--accent-dangerous)"></span> Dangerous</div>
            </div>
        </div>
    </div>

    <div class="sidebar">
        <div class="sidebar-header">
            <h2>Micro-Event Stream</h2>
            <span id="eventCount" style="font-size: 0.8rem; color: var(--text-muted);">0 events</span>
        </div>
        <div class="event-list" id="eventList"></div>
    </div>
</div>

<script>
    const eventsData = {events_json};

    // Filter to only events with coordinates or trajectory
    const trackableEvents = eventsData.filter(e => (e.x !== null && e.y !== null) || (e.trajectory && e.trajectory.length > 0));

    let currentIndex = 0;
    let isPlaying = false;
    let playInterval = null;

    const trajectoryLayer = document.getElementById('trajectoryLayer');
    const ballLayer = document.getElementById('ballLayer');
    const eventList = document.getElementById('eventList');
    const eventSlider = document.getElementById('eventSlider');
    const btnPlay = document.getElementById('btnPlay');
    const btnPrev = document.getElementById('btnPrev');
    const btnNext = document.getElementById('btnNext');
    const currentTimeEl = document.getElementById('currentTime');
    const stepCounterEl = document.getElementById('stepCounter');
    const eventCountEl = document.getElementById('eventCount');

    // Init slider bounds
    eventSlider.max = Math.max(0, trackableEvents.length - 1);
    eventCountEl.textContent = `${{trackableEvents.length}} records`;

    function getSituationColor(sit) {{
        switch ((sit || '').toLowerCase()) {{
            case 'attack': return '#f97316';
            case 'dangerous': return '#ef4444';
            default: return '#38bdf8';
        }}
    }}

    // Render sidebar items
    function populateSidebar() {{
        eventList.innerHTML = '';
        trackableEvents.forEach((ev, idx) => {{
            const div = document.createElement('div');
            div.className = `event-item ${{idx === currentIndex ? 'active' : ''}}`;
            div.dataset.index = idx;
            
            const colorClass = `pill-${{ev.situation.toLowerCase()}}`;
            div.innerHTML = `
                <div class="event-header">
                    <span class="event-time">${{ev.time_captured || 'Live'}} (${{ev.match_minute ? ev.match_minute + "'" : '45\''}})</span>
                    <span class="pill ${{colorClass}}">${{ev.situation.toUpperCase()}}</span>
                </div>
                <div class="event-body">
                    <span>${{ev.name || ev.type}}</span>
                    <span>X: ${{ev.x ? ev.x.toFixed(1) : '-'}}, Y: ${{ev.y ? ev.y.toFixed(1) : '-'}}</span>
                </div>
            `;
            div.addEventListener('click', () => {{
                setStep(idx);
            }});
            eventList.appendChild(div);
        }});
    }}

    // Render pitch graphics for specific index
    function renderStep(idx) {{
        if (trackableEvents.length === 0) return;
        const current = trackableEvents[idx];
        const color = getSituationColor(current.situation);

        // Update indicators
        currentTimeEl.textContent = current.time_captured || '--:--';
        stepCounterEl.textContent = `${{idx + 1}} / ${{trackableEvents.length}}`;
        eventSlider.value = idx;

        // Highlight sidebar item
        document.querySelectorAll('.event-item').forEach((item, i) => {{
            item.classList.toggle('active', i === idx);
            if (i === idx) item.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
        }});

        // Clear layers
        trajectoryLayer.innerHTML = '';
        ballLayer.innerHTML = '';

        // 1. Draw historical trail up to current step
        for (let i = 0; i <= idx; i++) {{
            const ev = trackableEvents[i];
            const evColor = getSituationColor(ev.situation);

            // Trajectory polyline
            if (ev.trajectory && ev.trajectory.length >= 2) {{
                const pts = [...ev.trajectory].reverse();
                let pointsStr = pts.map(p => `${{p.X}},${{p.Y}}`).join(' ');
                const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
                pathEl.setAttribute('points', pointsStr);
                pathEl.setAttribute('fill', 'none');
                pathEl.setAttribute('stroke', evColor);
                pathEl.setAttribute('stroke-width', i === idx ? '1.5' : '0.8');
                pathEl.setAttribute('stroke-dasharray', i === idx ? 'none' : '2,2');
                pathEl.setAttribute('opacity', i === idx ? '0.9' : '0.4');
                trajectoryLayer.appendChild(pathEl);
            }}

            // Point marker
            if (ev.x !== null && ev.y !== null) {{
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', ev.x);
                circle.setAttribute('cy', ev.y);
                circle.setAttribute('r', i === idx ? '2.4' : '1.2');
                circle.setAttribute('fill', evColor);
                circle.setAttribute('stroke', '#ffffff');
                circle.setAttribute('stroke-width', i === idx ? '0.8' : '0.4');
                circle.setAttribute('opacity', i === idx ? '1.0' : '0.5');
                trajectoryLayer.appendChild(circle);
            }}
        }}

        // 2. Render Active Ball with Glow
        if (current.x !== null && current.y !== null) {{
            // Pulse wave
            const pulse = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            pulse.setAttribute('cx', current.x);
            pulse.setAttribute('cy', current.y);
            pulse.setAttribute('class', 'ball-pulse');
            pulse.setAttribute('fill', color);
            ballLayer.appendChild(pulse);

            // Actual Ball Marker
            const ball = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            ball.setAttribute('cx', current.x);
            ball.setAttribute('cy', current.y);
            ball.setAttribute('r', '2.6');
            ball.setAttribute('fill', '#ffffff');
            ball.setAttribute('stroke', color);
            ball.setAttribute('stroke-width', '1.2');
            ballLayer.appendChild(ball);

            // Situation Label Above Ball
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', current.x);
            text.setAttribute('y', Math.max(5, current.y - 3.5));
            text.setAttribute('fill', '#ffffff');
            text.setAttribute('font-size', '2.8');
            text.setAttribute('font-weight', 'bold');
            text.setAttribute('text-anchor', 'middle');
            text.textContent = `[${{current.situation.toUpperCase()}}] (${{current.x.toFixed(0)}}, ${{current.y.toFixed(0)}})`;
            ballLayer.appendChild(text);
        }}
    }}

    function setStep(idx) {{
        currentIndex = Math.max(0, Math.min(trackableEvents.length - 1, idx));
        renderStep(currentIndex);
    }}

    function togglePlay() {{
        isPlaying = !isPlaying;
        btnPlay.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
        if (isPlaying) {{
            playInterval = setInterval(() => {{
                if (currentIndex >= trackableEvents.length - 1) {{
                    currentIndex = 0;
                }} else {{
                    currentIndex++;
                }}
                renderStep(currentIndex);
            }}, 1200);
        }} else {{
            clearInterval(playInterval);
        }}
    }}

    btnPlay.addEventListener('click', togglePlay);
    btnPrev.addEventListener('click', () => {{
        if (isPlaying) togglePlay();
        setStep(currentIndex - 1);
    }});
    btnNext.addEventListener('click', () => {{
        if (isPlaying) togglePlay();
        setStep(currentIndex + 1);
    }});
    eventSlider.addEventListener('input', (e) => {{
        if (isPlaying) togglePlay();
        setStep(parseInt(e.target.value));
    }});

    // Initialize
    populateSidebar();
    renderStep(0);
</script>

</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Interactive HTML Visualizer successfully saved to: {output_html}")
    return output_html


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
    
    # Default match
    match_id = 16939017
    csv_file = os.path.join(DATA_PROCESSED_DIR, f"live_events_{match_id}.csv")
    
    if not os.path.exists(csv_file):
        print(f"Target CSV {csv_file} does not exist! Searching for available files...")
        avail = [f for f in os.listdir(DATA_PROCESSED_DIR) if f.startswith("live_events_")]
        if avail:
            csv_file = os.path.join(DATA_PROCESSED_DIR, avail[0])
            print(f"Using fallback: {csv_file}")
        else:
            print("No event CSV found to visualize.")
            sys.exit(1)
            
    print(f"Generating visualizations for: {csv_file}")
    
    out_png = os.path.join(DATA_OUTPUT_DIR, f"tracking_{match_id}_pitch.png")
    out_html = os.path.join(DATA_OUTPUT_DIR, f"tracking_visualizer_{match_id}.html")
    
    generate_static_pitch_visualization(
        csv_path=csv_file,
        output_png=out_png,
        match_title=f"Sabah FK vs SK Slavia Praha (#{match_id}) - Live Tracking"
    )
    
    generate_interactive_html_visualizer(
        csv_path=csv_file,
        output_html=out_html,
        match_id=match_id,
        match_title="Sabah FK vs SK Slavia Praha"
    )
    
    print("\nVisualizations successfully created!")
