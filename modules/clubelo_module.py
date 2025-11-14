def clubelo_fixtures_to_df():
    import requests
    import pandas as pd
    import io  # Add this line to import the 'io' module

    # URL of the CSV file
    url = "http://api.clubelo.com/Fixtures"

    # Sending a GET request to download the CSV file
    response = requests.get(url)

    # Checking if the request was successful (status code 200)
    if response.status_code == 200:
        # Saving the content of the response (CSV data)
        csv_data = response.content

        # Using pandas to read the CSV data into a DataFrame
        df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
        return df
    else:
        print("Failed to download the CSV file.")