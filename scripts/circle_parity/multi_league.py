import pandas as pd
from collections import defaultdict
import datetime

# -----------------------------
# FILE
# -----------------------------
FILE = r"E:\Data\TSDP\datafiles\gamelogs_raw.xlsx"


# -----------------------------
# BITMASK HAMILTON CHECK (DP)
# -----------------------------
def has_hamilton_cycle_dp(adj, n):
    """
    adj: adjacency list (0..n-1)
    """

    if n == 0:
        return False

    # dp[mask][i] = elérhető-e, hogy mask csúcsokat bejárva i-ben végzünk
    dp = [[False] * n for _ in range(1 << n)]

    for i in range(n):
        dp[1 << i][i] = True

    for mask in range(1 << n):
        for u in range(n):
            if not dp[mask][u]:
                continue

            for v in adj[u]:
                if mask & (1 << v):
                    continue
                dp[mask | (1 << v)][v] = True

    full = (1 << n) - 1

    # ciklus: utolsó vissza tud menni az elsőhöz
    for start in range(n):
        for end in range(n):
            if dp[full][end] and start in adj[end]:
                return True

    return False


# -----------------------------
# LOAD
# -----------------------------
df = pd.read_excel(FILE)
df["Date"] = pd.to_datetime(df["Date"])


results = []


# -----------------------------
# MAIN LOOP
# -----------------------------
for (country, season), g in df.groupby(["Country", "Season"]):

    print(f"\n{country} {season}")

    g = g.sort_values("Wk")

    nodes = set()
    node_to_idx = {}

    adj = defaultdict(set)

    first_cycle_week = None
    first_cycle_date = None

    for wk in sorted(g["Wk"].unique()):
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')}: Week {wk}")

        wk_games = g[g["Wk"] == wk]

        # -------------------------
        # graph update
        # -------------------------
        for _, row in wk_games.iterrows():

            home = row["Home"]
            away = row["Away"]
            score = str(row["Score"])

            h_score, a_score = map(int, score.split("–"))

            nodes.add(home)
            nodes.add(away)

            if h_score > a_score:
                adj[home].add(away)
            elif a_score > h_score:
                adj[away].add(home)

        # indexelés (fontos DP-hez)
        nodes_list = list(nodes)

        node_to_idx = {n: i for i, n in enumerate(nodes_list)}

        n = len(nodes_list)

        # ha túl nagy lenne (biztonság)
        if n > 21:
            print("⚠️ Túl sok csapat a DP-hez, kihagyva")
            continue

        # adjacency list indexelve
        adj_idx = [[] for _ in range(n)]

        for u in nodes_list:
            for v in adj[u]:
                if v in node_to_idx:
                    adj_idx[node_to_idx[u]].append(node_to_idx[v])

        # -------------------------
        # CHECK
        # -------------------------
        if n >= 4:
            if has_hamilton_cycle_dp(adj_idx, n):

                first_cycle_week = wk
                first_cycle_date = wk_games["Date"].max()
                break

    results.append({
        "Country": country,
        "Season": season,
        "Nodes": len(nodes),
        "FirstCycleWeek": first_cycle_week,
        "FirstCycleDate": first_cycle_date
    })


# -----------------------------
# OUTPUT
# -----------------------------
res_df = pd.DataFrame(results)

print("\n📊 PARITY CYCLE TIMING")
print("────────────────────────\n")
print(res_df.sort_values("FirstCycleWeek"))

print("\n📈 SUMMARY")
print("Mean week:", res_df["FirstCycleWeek"].mean())
print("Median week:", res_df["FirstCycleWeek"].median())