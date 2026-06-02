import random
from collections import defaultdict
import numpy as np

N_TEAMS = 12
ROUNDS = 33
GAMES_PER_ROUND = 6
SIMS = 500


teams = list(range(N_TEAMS))


def add_edge(graph, a, b):
    graph[a].add(b)


def has_hamilton_cycle(graph):
    n = N_TEAMS

    def backtrack(path, visited):
        if len(path) == n:
            return path[0] in graph[path[-1]]

        last = path[-1]

        for nei in graph[last]:
            if nei not in visited:
                visited.add(nei)
                if backtrack(path + [nei], visited):
                    return True
                visited.remove(nei)

        return False

    for start in teams:
        if backtrack([start], {start}):
            return True

    return False


def simulate_one():
    graph = defaultdict(set)

    for r in range(1, ROUNDS + 1):

        round_pairs = []

        while len(round_pairs) < GAMES_PER_ROUND:
            a = random.randint(0, N_TEAMS - 1)
            b = random.randint(0, N_TEAMS - 1)

            if a == b:
                continue

            round_pairs.append((a, b))

        for a, b in round_pairs:
            if random.random() < 0.50:
                add_edge(graph, a, b)
            else:
                add_edge(graph, b, a)

        if has_hamilton_cycle(graph):
            return r

    return None


results = []

for i in range(SIMS):
    if i % 125 == 0:
        print(f"Sim {i}/{SIMS}")

    t = simulate_one()
    results.append(t if t is not None else np.nan)


arr = np.array(results, dtype=float)

print("\n📊 RESULTS")
print("────────────")

print("Mean emergence round:", np.nanmean(arr))
print("Median:", np.nanmedian(arr))
print("Success rate:", np.mean(~np.isnan(arr)))

print("\nDistribution (rounded):")
for i in range(1, ROUNDS + 1):
    print(f"Round {i:2d}: {(arr == i).sum()}")