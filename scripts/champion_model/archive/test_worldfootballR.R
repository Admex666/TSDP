# worldfootballR - Összes elérhető load függvény és stat típus
library(worldfootballR)
library(dplyr)

output_file <- "c:/Users/Adam/Data/TSDP/scripts/champion_model/worldfootballR_results.txt"
sink(output_file)

cat("=================================================================\n")
cat("worldfootballR - ÖSSZES ELÉRHETŐ ADAT\n")
cat("=================================================================\n")
cat("Csomag verzió:", as.character(packageVersion("worldfootballR")), "\n")
cat("Időpont:", as.character(Sys.time()), "\n\n")

# Elérhető load_ függvények listázása
cat("ELÉRHETŐ LOAD_ FÜGGVÉNYEK A CSOMAGBAN:\n")
cat("======================================\n")
funs <- ls("package:worldfootballR")
load_funs <- funs[grepl("^load_", funs)]
print(load_funs)

# Összes stat típus lekérdezése
stat_types <- c("shooting", "passing", "passing_types", "gca", "defense", 
                "possession", "playing_time", "misc", "keepers", "keepers_adv")

cat("\n\n=================================================================\n")
cat("MINDEN STAT TÍPUS - PREMIER LEAGUE 2024\n")
cat("=================================================================\n")

for (stat in stat_types) {
  cat("\n--- ", toupper(stat), " ---\n")
  tryCatch({
    data <- load_fb_big5_advanced_season_stats(
      season_end_year = 2024,
      stat_type = stat,
      team_or_player = "team"
    )
    
    # Csak Premier League, csak "for" (nem "against")
    pl <- data %>% 
      filter(grepl("Premier", Comp)) %>%
      filter(Team_or_Opponent == "team" | is.na(Team_or_Opponent))
    
    cat("Sorok:", nrow(pl), "\n")
    cat("Oszlopok (első 15):\n")
    print(head(names(data), 15))
    
  }, error = function(e) {
    cat("HIBA:", e$message, "\n")
  })
}

# Arsenal részletes adatok minden stat típusból
cat("\n\n=================================================================\n")
cat("ARSENAL - RÉSZLETES SZEZON STATISZTIKÁK 2024\n")
cat("=================================================================\n")

cat("\n--- SHOOTING ---\n")
shooting <- load_fb_big5_advanced_season_stats(2024, "shooting", "team")
arsenal_shooting <- shooting %>% filter(Squad == "Arsenal", Team_or_Opponent == "team")
print(t(arsenal_shooting))

cat("\n--- PASSING ---\n")
passing <- load_fb_big5_advanced_season_stats(2024, "passing", "team")
arsenal_passing <- passing %>% filter(Squad == "Arsenal", Team_or_Opponent == "team")
print(t(arsenal_passing))

cat("\n--- DEFENSE ---\n")
defense <- load_fb_big5_advanced_season_stats(2024, "defense", "team")
arsenal_defense <- defense %>% filter(Squad == "Arsenal", Team_or_Opponent == "team")
print(t(arsenal_defense))

cat("\n--- POSSESSION ---\n")
possession <- load_fb_big5_advanced_season_stats(2024, "possession", "team")
arsenal_possession <- possession %>% filter(Squad == "Arsenal", Team_or_Opponent == "team")
print(t(arsenal_possession))

cat("\n--- KEEPERS (xGA, védések) ---\n")
keepers <- load_fb_big5_advanced_season_stats(2024, "keepers", "team")
arsenal_keepers <- keepers %>% filter(Squad == "Arsenal", Team_or_Opponent == "team")
print(t(arsenal_keepers))

sink()
cat("Eredmények mentve ide:", output_file, "\n")
