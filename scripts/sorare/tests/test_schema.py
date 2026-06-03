import requests
import json

query = """
query {
  football {
    player(slug: "lionel-andres-messi-cuccittini") {
      id
      displayName
    }
  }
}
"""
query2 = """
query {
  football {
    players(slugs: ["lionel-andres-messi-cuccittini"]) {
      id
      displayName
    }
  }
}
"""
query3 = """
query {
  players(slugs: ["lionel-andres-messi-cuccittini"]) {
    id
    displayName
  }
}
"""

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

print("Trying query3 (players directly):")
res3 = requests.post("https://api.sorare.com/graphql", json={"query": query3}, headers=headers)
print(res3.json())

print("Trying query1 (football { player }):")
res1 = requests.post("https://api.sorare.com/graphql", json={"query": query}, headers=headers)
print(res1.json())
