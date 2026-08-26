# SofaScore & Sportradar Live 3D Tracking Architektúra Elemzés

Ez a dokumentum összefoglalja a SofaScore mérkőzésoldalain található 3D Live Match Tracker (`<canvas id="renderCanvas">`, Babylon.js v8.22.0) működését, adatforrásait, WebSocket protokolljait és a kinyerhető koordináták struktúráját.

---

## 1. Architektúra és Adatfolyam Áttekintés

A SofaScore mérkőzéskövető felülete két független valós idejű adatfolyamból épül fel:

```
                          ┌──────────────────────────────┐
                          │    SofaScore Match Page      │
                          └──────────────┬───────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │    SofaScore NATS WS      │                   │   Sportradar VLMT LMT+    │
   │  wss://ws.sofascore.com   │                   │ wss://ws.fn.sportradar.com│
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 ▼                                               ▼
      Meccs események (gól,                        3D Match Tracker (Babylon.js)
      lapok, statisztikák,                         - 4 pontos labda trajektória
      élő eredményváltozás)                        - Akció / támadási zónák
                                                   - Taktikai felállás animáció
```

---

## 2. A 3D Canvas (`renderCanvas` / Babylon.js) Működése

A vizsgálat során elemezett DOM elem:
```html
<canvas id="renderCanvas" touch-action="none" data-engine="Babylon.js v8.22.0" ...></canvas>
```

### Valós Tracking vs. Procedurális Animáció
* **Labda pozíció (Valós idejű)**: A stream másodpercről másodpercre küldi a valós `(X, Y)` pálya koordinátákat és 4-pontos mozgásíveket (`ballcoordinates`). A Babylon.js a 3D labda modellt ezek mentén animálja.
* **Játékosok mozgása (Szimulált/Procedurális)**: A 22 mezőnyjátékos pozíciója **nem** valós idejű 25/50 Hz-es optikai/GPS trackingből jön. A Sportradar widget a `/match_lineup` statikus kezdőfelállásból és a támadás irányából (`situation: dangerous/attack`) számol ki prediktív helyezkedést a 3D játékosmodellekhez.
* **Coverage szint**: A SofaScore a Sportradar **VLMT Level 2 / LMT+** szintjét használja (`"coverage": {"depth": "DEEP", "level": 2}`). A teljes 22 fős egyéni játékos tracking a Sportradar különálló, licencelt Tracking Services terméke (Level 4/5).

---

## 3. WebSocket és REST Végpontok

### A) Sportradar NATS WebSocket (Tracking & Trajektória)
* **URL**: `wss://ws.fn.sportradar.com/wss?T={signed_token}`
* **Kliens Alias**: `acf51edeeec126432707c0bf07673d86`
* **Csatorna feliratkozás (Subscription payload)**:
  ```json
  {"type": "sub", "subjects": {"match_timeline_fn.events.<feed_id>": -10}}
  ```
  *(Ahol `<feed_id>` a mérkőzés belső Sportradar azonosítója, pl. `73394628`)*

### B) SofaScore NATS WebSocket (Score & Incident Feed)
* **URL**: `wss://ws.sofascore.com:9222/`
* **Csatorna feliratkozás**: `SUB event.<event_id> 1`

### C) Kapcsolódó REST Endpointok
| Végpont | Funkció |
| :--- | :--- |
| `/api/v1/event/{id}/live-match-tracker` | A Sportradar LMT+ beágyazó HTML konténere és widget loader konfigurációja |
| `https://lmt.fn.sportradar.com/.../gismo/match_timelinedelta/{feed_id}` | REST fallback a mérkőzés legfrissebb eseményeihez és trajektóriáihoz |
| `https://f3.sportradar.com/api/vlmt/v0/matches?sport_id=sr:sport:1` | Elérhető élő VLMT stream-ek listája és FPS adatai (50 fps) |

---

## 4. Nyers Adatformátum (Payload Minta)

A `match_timelinedelta` eseménycsomagban érkező valós idejű koordináta-adat:

```json
{
  "event": "match_timelinedelta",
  "data": {
    "events": [
      {
        "_id": "dcc74231-ca8b-449a-9be8-4ed4b69190df",
        "type": "matchsituation",
        "name": "matchsituation",
        "situation": "dangerous",
        "team": "away",
        "time": 10,
        "seconds": 548,
        "X": 10,
        "Y": 52,
        "uts": 1787501421,
        "matchid": 73394628
      },
      {
        "_id": "dcc74231-ca8b-449a-9be8-4ed4b69190df-1140",
        "type": "ballcoordinates",
        "name": "Pitch coordinates",
        "coordinates": [
          { "team": "away", "X": 10.0, "Y": 52.0 },
          { "team": "away", "X": 16.0, "Y": 58.0 },
          { "team": "away", "X": 28.0, "Y": 92.0 },
          { "team": "away", "X": 20.0, "Y": 90.0 }
        ],
        "uts": 1787501421,
        "matchid": 73394628
      }
    ]
  }
}
```

* **Koordináta skála**: `X: 0.0 - 100.0`, `Y: 0.0 - 100.0` (normált pálya koordináták).
* **Trajektória**: A `coordinates` tömb 4 pontja írja le a labda kiindulását, ívét és megérkezési pontját.

---

## 5. Kapcsolat a TSDP Projekttel

A projekt [SofaScore_module.py](file:///c:/Users/Adam/Data/TSDP/modules/SofaScore_module.py) moduljában lévő metódusok közvetlenül erre az architektúrára épülnek:

1. **`fetch_live_match_events(event_id, duration_seconds)`**:
   - Automatizált Chrome session segítségével csatlakozik a streamhez.
   - Elcsípi a fenti `match_timelinedelta` üzeneteket, és automatikusan strukturált DataFrame-be rendezi (`x`, `y`, `start_x`, `start_y`, `trajectory_points`, `situation`, `team`, `match_minute`).
2. **`match_live_events_with_player_passes(...)`** ([match_events_with_passes.py](file:///c:/Users/Adam/Data/TSDP/scripts/ss_event_data/match_events_with_passes.py)):
   - A streamelt 3D labda koordinátákat idő és térbeli euklideszi távolság alapján összeköti a SofaScore Opta passz adatbázisával (`player_name`, `action_type`, `outcome`).
