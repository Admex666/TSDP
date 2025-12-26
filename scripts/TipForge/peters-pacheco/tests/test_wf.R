library(worldfootballR)

cat("--- Testing with 5-second delay ---\n")
Sys.sleep(5)

cat("\n--- Fetching Kevin De Bruyne Match Logs (Summary) ---\n")
tryCatch({
    kdb_logs <- fb_player_match_logs("https://fbref.com/en/players/e46012d4/Kevin-De-Bruyne", season_end_year = 2024, stat_type = "summary")
    cat("Loaded columns: ", length(colnames(kdb_logs)), "\n")
    print(colnames(kdb_logs))
    print(head(kdb_logs[, 1:15]))
}, error = function(e) {
    cat("Error caught during KDB fetch: ", e$message, "\n")
})


cat("\n--- Testing simple team stats (might be less protected) ---\n")
tryCatch({
    team_stats <- fb_team_match_log_stats(team_url = "https://fbref.com/en/squads/b8c47410/Manchester-City-Stats", stat_type = "shooting")
    print(head(team_stats))
}, error = function(e) {
    cat("Error caught during Team stats fetch: ", e$message, "\n")
})

