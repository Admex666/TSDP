# FBref Kapcsolat Diagnosztika
# Cél: Kideríteni, hogy az URL elérhető-e vagy blokkolva van (HTTP 403/429)

library(httr)
library(rvest)

url <- "https://fbref.com/en/matches/c7512760/Wolves-Arsenal-April-20-2024-Premier-League"
output_file <- "c:/Users/Adam/Data/TSDP/scripts/champion_model/connection_debug.txt"
sink(output_file)

cat("=== KAPCSOLAT DIAGNOSZTIKA ===\n")
cat("Időpont:", as.character(Sys.time()), "\n")
cat("Cél URL:", url, "\n\n")

# 1. Kísérlet: Sima GET kérés (mint egy böngésző)
cat("1. KÍSÉRLET: httr::GET kérés User-Agent beállítással\n")
tryCatch(
    {
        # User-Agent beállítása fontos, hogy böngészőnek tűnjünk
        response <- GET(
            url,
            add_headers(`User-Agent` = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        )

        status <- status_code(response)
        cat("HTTP Státusz Kód:", status, "\n")

        if (status == 200) {
            cat("Sikeres kapcsolat! (OK)\n")
            cat("Tartalom típusa:", headers(response)$`content-type`, "\n")
            content_text <- content(response, "text", encoding = "UTF-8")
            cat("Tartalom hossza:", nchar(content_text), "karakter\n")

            # Ellenőrizzük, hogy benne van-e a "Match Summary" szöveg
            has_summary <- grepl("Match Summary", content_text)
            cat("Tartalmaz 'Match Summary' szöveget?:", has_summary, "\n")

            # Ellenőrizzük, hogy nincs-e bot check szöveg
            is_bot_check <- grepl("automated requests", content_text) || grepl("Too Many Requests", content_text)
            cat("Bot ellenőrzés gyanú?:", is_bot_check, "\n")
        } else if (status == 403) {
            cat("HIBA: 403 Forbidden - A szerver blokkolja a kérést.\n")
            cat("Ez a leggyakoribb ok: Az FBref érzékelte, hogy scriptből jön a kérés.\n")
        } else if (status == 429) {
            cat("HIBA: 429 Too Many Requests - Túl sok kérés ment ki rövid idő alatt.\n")
            cat("Várni kell (kb. 1 órát) és növelni a 'time_pause' értékét.\n")
        } else {
            cat("Egyéb hiba történt. Státusz:", status, "\n")
        }
    },
    error = function(e) {
        cat("KIVÉTEL TÖRTÉNT:", e$message, "\n")
    }
)

cat("\n------------------------------------------------\n")

# 2. Kísérlet: rvest olvasás (ez használja a worldfootballR is a háttérben)
cat("2. KÍSÉRLET: rvest::read_html (alapértelmezett, amit a csomag használ)\n")
tryCatch(
    {
        # Itt NEM állítunk be extra header-öket, hogy lássuk, hogyan viselkedik natívan
        page <- read_html(url)
        cat("Sikerült beolvasni az oldalt rvest-tel!\n")

        # Táblázatok keresése
        tables <- page %>% html_nodes("table")
        cat("Talált táblázatok száma:", length(tables), "\n")
    },
    error = function(e) {
        cat("HIBA (rvest):", e$message, "\n")
        if (grepl("403", e$message)) {
            cat("-> MEGERŐSÍTVE: A worldfootballR által használt módszer 403-as hibát kap.\n")
        }
    }
)

sink()
cat("Eredmény mentve:", output_file, "\n")
