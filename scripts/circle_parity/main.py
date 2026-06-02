import csv
from collections import defaultdict
from datetime import datetime

# -----------------------------
# 1. Graph + edge metadata
# -----------------------------
graph = defaultdict(set)

# (winner -> loser -> (date, week, score))
edge_info = defaultdict(dict)

teams = set()

def parse_date(d):
    return datetime.strptime(d, "%Y-%m-%d")

with open("nb1.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["Home"] != "true":
            continue
        if int(row["Week"]) > 11:
            continue

        home = row["teamFullName"]
        away = row["opponentFullName"]
        result = row["Result"]

        date = row["Date"]
        week = int(row["Week"])
        score = f"{row['score']}-{row['finalScoreOpponent']}"

        teams.add(home)
        teams.add(away)

        if result.startswith("W"):
            graph[home].add(away)
            edge_info[home][away] = (date, week, score)

        elif result.startswith("L"):
            graph[away].add(home)
            edge_info[away][home] = (date, week, score)

teams = list(teams)

# -----------------------------
# 2. Hamilton cycle search
# -----------------------------
def find_hamilton_cycle():
    n = len(teams)

    def backtrack(path, visited):
        if len(path) == n:
            if path[0] in graph[path[-1]]:
                return path + [path[0]]
            return None

        last = path[-1]

        for nei in graph[last]:
            if nei not in visited:
                visited.add(nei)
                res = backtrack(path + [nei], visited)
                if res:
                    return res
                visited.remove(nei)

        return None

    for start in teams:
        res = backtrack([start], {start})
        if res:
            return res

    return None

cycle = find_hamilton_cycle()

# -----------------------------
# 3. Temporal closure analysis
# -----------------------------
if cycle:
    print("\n🔁 CIRCLE OF PARITY FOUND\n")

    events = []

    for i in range(len(cycle) - 1):
        a = cycle[i]
        b = cycle[i + 1]

        date, week, score = edge_info[a][b]
        events.append((parse_date(date), week, a, b, date, score))

        print(f"{a} → {b} | Week {week} | {date} | {score}")

    # záró él
    a = cycle[-2]
    b = cycle[-1]

    date, week, score = edge_info[a][b]
    events.append((parse_date(date), week, a, b, date, score))

    print(f"{a} → {b} | Week {week} | {date} | {score}")

    # -----------------------------
    # 4. FIRST COMPLETION MOMENT
    # -----------------------------
    latest = max(events, key=lambda x: (x[0], x[1]))

    print("\n📅 FIRST TIME CIRCLE COMPLETED:")
    print(f"Week {latest[1]} | {latest[0].strftime('%Y-%m-%d')}")
    print(f"Last edge added: {latest[2]} → {latest[3]} ({latest[4]}, {latest[5]})")

else:
    print("❌ No full parity circle found")

"""

# -----------------------------
# RAW DATA LOAD
# -----------------------------
matches = []

with open("nb1.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["Home"] != "true":
            continue

        matches.append({
            "week": int(row["Week"]),
            "home": row["teamFullName"],
            "away": row["opponentFullName"],
            "result": row["Result"]
        })

teams = set()
for m in matches:
    teams.add(m["home"])
    teams.add(m["away"])

teams = list(teams)

# -----------------------------
# BUILD GRAPH UP TO GIVEN WEEK
# -----------------------------
def build_graph(up_to_week):
    graph = defaultdict(set)

    for m in matches:
        if m["week"] > up_to_week:
            continue

        home = m["home"]
        away = m["away"]
        result = m["result"]

        if result.startswith("W"):
            graph[home].add(away)
        elif result.startswith("L"):
            graph[away].add(home)

    return graph

# -----------------------------
# FIND BEST CYCLE (approx)
# -----------------------------
def longest_cycle(graph):
    best = []

    def dfs(start, node, visited, path):
        nonlocal best

        for nei in graph[node]:
            if nei == start and len(path) > 2:
                if len(path) > len(best):
                    best = path[:]
                continue

            if nei not in visited:
                visited.add(nei)
                dfs(start, nei, visited, path + [nei])
                visited.remove(nei)

    for t in teams:
        dfs(t, t, {t}, [t])

    return best

# -----------------------------
# WEEKLY SIMULATION
# -----------------------------
max_week = max(m["week"] for m in matches)

print("\n📊 PARITY EVOLUTION OVER SEASON\n")

for w in range(1, max_week + 1):
    graph = build_graph(w)
    cycle = longest_cycle(graph)

    print(f"Week {w:2d} → cycle size: {len(cycle)}")
"""