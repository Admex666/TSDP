import json
from api_client import SorareApiClient

client = SorareApiClient()

query = """
query GetClubPlayers($slug: String!) {
  football {
    club(slug: $slug) {
      name
      activePlayers {
        nodes {
          slug
          displayName
        }
      }
    }
  }
}
"""

variables = {"slug": "real-madrid-madrid"}
res = client.execute_query(query, variables)
print(json.dumps(res, indent=2))
