import requests
import pandas as pd
from datetime import datetime


def fetch_data(year, month):
    """Fetch data for a specific year and month."""
    day = '01'
    filename = f"{year}-{month:02}-{day}-as.csv"
    url = f'https://resources.potaroo.net/iso3166/archive/{year}/{month:02}/{filename}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching data for {year}-{month}: {e}")
        return None


def parse_data(html_content):
    """Parse HTML content into a DataFrame."""
    to_start_saving = False
    to_column = False
    data = []
    columns_line = []
    for line in html_content.split('\n'):
        if to_start_saving:
            if len([value.strip() for value in line.split(',')]) > 27:
                print('?!',len([value.strip() for value in line.split(',')]))
                continue
            data.append([value.strip() for value in line.split(',')])
        if to_column:
            columns_line = line.split(',')
            columns_line.append('World')
            columns_line.append('Country Name')
            to_start_saving = True
            to_column = False
        if line.startswith('Country Table'):
            to_column = True
        if line.startswith('Regional Table'):
            break

    return pd.DataFrame(data, columns=columns_line)


def process_data(year_range):
    """Process data for given year range."""
    for year in year_range:
        for month in range(1, 13):
            # if year == 2021 and month >= 8:
            #     break  # Stop early for 2021 after July

            print(f"Processing {year}-{month}")
            html_content = fetch_data(year, month)
            if html_content:
                df = parse_data(html_content)
                df.to_csv(f'../data_potaroo/{year}-{month:02}-01-as.csv', index=False)


def analyze_country_data(file_path):
    """Analyze country data from a file and store results."""
    try:
        data = pd.read_csv(file_path)
        print(data)
    except pd.errors.EmptyDataError:
        print("No data to display.")


def main():
    # Range of years to process
    year_range = range(2024, 2025)
    process_data(year_range)
    # analyze_country_data('../data_potaroo/2020-01-01-as.csv')


if __name__ == "__main__":
    main()