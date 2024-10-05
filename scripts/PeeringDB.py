import json
import pandas as pd
from collections import defaultdict
from geopy.geocoders import Nominatim
from tqdm import tqdm
from statistics import mean
import os
import time

# Directory where the data will be stored
DATA_DIR = '../data/PeeringDB/'


# Function to download PeeringDB data
def download_peeringdb_data(year, month):
    file_url = f"http://data.caida.org/datasets/peeringdb-v2/{year}/{month}/peeringdb_2_dump_{year}_{month}_01.json"
    file_path = f"{DATA_DIR}peeringdb_2_dump_{year}_{month}_01.json"
    os.system(f"wget {file_url} -O {file_path}")


# Function to generate AS-level information from PeeringDB
def generate_as_peeringdb(year, month):
    file_path = f"{DATA_DIR}peeringdb_2_dump_{year}_{month}_01.json"

    with open(file_path) as json_file:
        data = json.load(json_file)

    unbundled = {}
    logical_connectors = defaultdict(list)

    for key in data:
        if key == 'api':
            continue
        unbundled[key] = pd.DataFrame.from_dict(data[key]['data'])

        for col in unbundled[key].columns:
            if col == 'id':
                logical_connectors[f"{key}_id"].append(f"{key}_{col}")
                unbundled[key][f"{key}_id"] = unbundled[key]['id']
            elif 'id' in col:
                logical_connectors[f"{key}_id"].append(col)

        if 'id' in unbundled[key].columns:
            del unbundled[key]['id']

    # Merge AS and Org data
    merged_data = pd.merge(unbundled['net'], unbundled['org'], on='org_id')

    # Save the merged data to CSV
    output_file = f"{DATA_DIR}PeeringDB_AS_level_info_{year}-{month}-01.csv"
    merged_data.to_csv(output_file)


# Function to generate IXP data
def generate_ixp_data(year, month):
    file_path = f"{DATA_DIR}peeringdb_2_dump_{year}_{month}_01.json"

    with open(file_path) as json_file:
        data = json.load(json_file)

    ixp_df = pd.DataFrame.from_dict(data['ix']['data'])

    # Adding latitude and longitude based on city
    geolocator = Nominatim(user_agent='burdantes')
    bounds = {}

    city_list = list(set(ixp_df['city']))

    for city in tqdm(city_list):
        location = geolocator.geocode(city, timeout=100)
        try:
            bounding = list(map(float, location.raw['boundingbox']))
            lat, long = mean(bounding[:2]), mean(bounding[2:])
            bounds[city] = [lat, long]
        except:
            bounds[city] = [0, 0]

    ixp_df['lat'] = ixp_df['city'].map(lambda city: bounds[city][0])
    ixp_df['lon'] = ixp_df['city'].map(lambda city: bounds[city][1])

    # Save IXP data with lat/lon
    ixp_df.to_csv(f"{DATA_DIR}IX_PeeringDB_{year}_{month}.csv")


# Function to generate IXP networks data
def generate_ixp_networks(year, month):
    file_path = f"{DATA_DIR}peeringdb_2_dump_{year}_{month}_01.json"

    with open(file_path) as json_file:
        data = json.load(json_file)

    ixp_network_df = pd.DataFrame.from_dict(data['netixlan']['data'])
    ixp_network_df.to_csv(f"{DATA_DIR}IXnet_PeeringDB_{year}_{month}.csv")


# Function to generate facility data
def generate_facilities_as(year, month):
    file_path = f"{DATA_DIR}peeringdb_2_dump_{year}_{month}_01.json"

    with open(file_path) as json_file:
        data = json.load(json_file)

    fac_df = pd.DataFrame.from_dict(data['fac']['data'])
    fac_df.set_index('clli', inplace=True)
    fac_df = fac_df[['address1', 'country', 'city', 'latitude', 'longitude']]

    fac_df.to_csv(f"{DATA_DIR}CLLI_PDB_{year}-{month}.csv")


if __name__ == "__main__":
    # Get today's date for generating the year and month if needed
    today = time.strftime("%Y-%m-%d")
    year = '2024'  # Fixed year for demonstration
    month = '04'  # Fixed month for demonstration

    download_peeringdb_data(year, month)
    generate_facilities_as(year, month)
    generate_ixp_data(year, month)
    generate_ixp_networks(year, month)
    generate_as_peeringdb(year, month)
