def get_coordinates(city, postal_code=None):
    #pip install geopy
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="city_geocoder")
    
    # Ha irányítószámot is adunk meg, akkor azt is beállítjuk
    if postal_code:
        location = geolocator.geocode(f"{city}, {postal_code}")
    else:
        location = geolocator.geocode(city)
    
    if location:
        return location.latitude, location.longitude
    else:
        return None
    
def get_hourly_weather(latitude, longitude, start_date, end_date):
    # pip install openmeteo-requests
    # pip install requests_cache
    # pip install retry-requests
    
    import openmeteo_requests
    import requests_cache
    import pandas as pd
    from retry_requests import retry
    
    # Open-Meteo API kliens beállítása gyorsítótárral és hibakezeléssel
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    # API hívás paramétereinek beállítása
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,  # Budapest szélességi fok
        "longitude": longitude,  # Budapest hosszúsági fok
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,wind_speed_10m,weathercode"
    }
    
    # API hívás végrehajtása
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Óránkénti adatok feldolgozása
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
    hourly_weathercode = hourly.Variables(2).ValuesAsNumpy()
    
    # Időbélyegek létrehozása
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=False),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=False),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly_temperature_2m,
        "wind_speed_10m": hourly_wind_speed_10m,
        "weathercode": hourly_weathercode
    }
    
    # DataFrame létrehozása
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    
    # Időjárás kódok értelmezése
    weather_descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Intense drizzle",
        56: "Light freezing drizzle",
        57: "Intense freezing drizzle",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Light snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Hail",
        80: "Light showers",
        81: "Moderate showers",
        82: "Heavy showers",
        85: "Light snow showers",
        86: "Heavy snow showers",
        95: "Light or moderate thunderstorm",
        96: "Light thunderstorm with hail",
        99: "Severe thunderstorm with hail"
    }
    
    # Az időjárás kódok leírásának hozzáadása a DataFrame-hez
    hourly_dataframe["weather_description"] = hourly_dataframe["weathercode"].map(weather_descriptions)
    # Átváltás m/s-ról km/h-ra
    hourly_dataframe["wind_speed_kmh"] = hourly_dataframe["wind_speed_10m"] * 3.6
    
    # Eredmény kiíratása
    output_df = hourly_dataframe[["date", "temperature_2m", "wind_speed_kmh", "weather_description", "weathercode"]]
    output_df = output_df.copy()
    output_df.rename(columns={"temperature_2m": "temp_celsius"}, inplace=True)
        
    return output_df
