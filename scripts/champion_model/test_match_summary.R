# Teszt: fb_match_summary()
library(worldfootballR)

# Wolves vs Arsenal (2024-04-20)
match_url <- "https://fbref.com/en/matches/fc8ab8b2/Brighton-and-Hove-Albion-Manchester-United-August-24-2024-Premier-League"

output_file <- "c:/Users/Adam/Data/TSDP/scripts/champion_model/match_summary_result.txt"
sink(output_file)

cat("=== fb_match_summary() Teszt ===\n")
cat("Időpont:", as.character(Sys.time()), "\n")
cat("URL:", match_url, "\n")

tryCatch(
  {
    # A kért függvényhívás
    summary_data <- fb_match_urls(country = "AUS", gender = "F", season_end_year = 2021, tier = "1st")

    # df <- fb_match_summary(match_url = summary_data)

    cat("SIKERES LEKÉRDEZÉS!\n")
    cat("Sorok száma:", nrow(df), "\n")
    cat("Oszlopok:\n")
    # print(names(df))

    cat("\nADATOK (első 10 sor):\n")
    # print(head(summary_data, 10))

    # Nézzünk bele az esemény típusokba
    if ("Event_Type" %in% names(summary_data)) {
      cat("\nEsemény típusok eloszlása:\n")
      # print(table(summary_data$Event_Type))
    }
  },
  error = function(e) {
    cat("HIBA TÖRTÉNT:\n")
    cat(e$message, "\n")
    if (!is.null(e$parent)) {
      cat("Részletek:", e$parent$message, "\n")
    }
  }
)

sink()
cat("Eredmény mentve:", output_file, "\n")
