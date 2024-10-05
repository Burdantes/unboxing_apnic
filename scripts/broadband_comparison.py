import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from copy import deepcopy
import re
import matplotlib.ticker as mtick
import pycountry
import numpy as np


# Global constants
DIRECTORY = '../data/PeeringDB/'
PDB_YEAR = '2024'
PDB_MONTH = '04'

country_dict = {'Mexico': 'MX', 'Brazil': 'BR', 'India': 'IN', 'Russia': 'RU', 'Japan': 'JP', 'South Korea': 'KR', }


# Function to load PeeringDB data
def load_peeringdb_data(year, month, directory):
    file_path = f'{directory}peeringdb_2_dump_{year}_{month}_01.json'
    with open(file_path) as json_file:
        return json.load(json_file)
percentage_per_country_covered = {}


# Function to extract country name from the filename
def extract_country_name(filename):
    match = re.search(r'brands-in-(.*?)-20\d{2}', filename)
    if match:
        return match.group(1).replace("-", " ").title()
    return None
# Function to prepare US data
def prepare_us_data():
    data_us = {
        'ISP': ['Comcast', 'Charter Communications', 'AT&T', 'Verizon', 'Cox Communications',
                'Altice USA', 'Lumen Technologies', 'Frontier Communications', 'Mediacom Communications',
                'Astound Broadband', 'Windstream Holdings', 'Brightspeed', 'Cable One',
                'Breezeline', 'WideOpenWest', 'TDS Telecom', 'Midco', 'Consolidated Communications',
                'Google Fiber', 'Ziply Fiber', 'altafiber', 'Armstrong Cable Services',
                'Service Electric', 'MetroNet', 'Hotwire Communications'],
        'Percentage': [32.2, 30.4, 15.4, 7.5, 5.6, 4.3, 3.0, 2.8, 1.5, 1.2, 1.17, 1.15, 1.06, 0.694, 0.5116, 0.51,
                       0.49, 0.367458, 0.367458, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25],
        'ASN': ['AS7922', 'AS20115', 'AS7018', 'AS701', 'AS22773', 'AS22252', 'AS3356', 'AS5650', 'AS30036',
                'AS19048', 'AS7029', 'AS30600', 'AS11492', 'AS7015', 'AS12008', 'AS4181', 'AS11269', 'AS6181',
                'AS16591', 'AS271', 'AS16631', 'AS5650', 'AS30600', 'AS16591', 'AS16591'],
        'Fraction of Users': [32.2, 30.4, 15.4, 7.5, 5.6, 4.3, 3.0, 2.8, 1.5, 1.2, 1.17, 1.15, 1.06, 0.694, 0.5116,
                              0.51, 0.49, 0.367458, 0.367458, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25],
        'Country': ['US'] * 25
    }
    df_us = pd.DataFrame(data_us)
    return df_us[df_us['Percentage'] > 1]


# Function to load sibling ASNs from another dataset
def load_sibling_data(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


# Function to get ASN from PeeringDB
def get_asn_from_peeringdb(isp_name, data_asn, data_org, data_sibling):
    if 'Other' == isp_name:
        return []

    list_of_asn = []

    # ISP-specific ASN mapping
    isp_mapping = {
        'Telmex': ['AS8151'], 'Vivo': ['AS26599'], 'Totalplay': [], 'Telcel': ['AS28403'],
        'Spectrum': ['AS21989', 'AS7843'], 'Tim': ['AS26615'], 'Jio Fiber': ['AS55836', 'AS64049'],
        'Megacable': ['AS13999', 'AS28541'], 'Excitel': ['AS133982'], 'BSNL - Bharat Sanchar Nigam': ['AS9829'],
        'ACT - Atria Convergence Technologies': ['AS24309'], 'Asianet Broadband': ['AS45415'],
        'Hathway Cable & Datacom': ['AS17488'], 'Beeline Internet': ['AS3216'], 'MTS Internet': ['AS8359'],
        'Domru': ['AS60043'], 'Drei / 3': ['AS25255'], 'J:COM': ['AS9824', 'AS9617', 'AS4721'],
        'Eo Hikari': ['AS23629']
    }

    list_of_asn.extend(isp_mapping.get(isp_name, []))

    for item in data_asn:
        if isp_name.lower() in item['name'].lower():
            list_of_asn.append(f"AS{item['asn']}")

    for item in data_org:
        if isp_name.lower() in item['name'].lower():
            org_id = item['id']
            for item in data_asn:
                if item['org_id'] == org_id:
                    list_of_asn.append(f"AS{item['asn']}")

    # Check for sibling ASNs
    list_of_asn_with_sibling = deepcopy(list_of_asn)
    for asn in list_of_asn:
        asn_number = asn[2:]
        if asn_number in data_sibling:
            sibling_asns = data_sibling[asn_number].get('Sibling ASNs', [])
            list_of_asn_with_sibling.extend([f"AS{item}" for item in sibling_asns])

    return list(set(list_of_asn_with_sibling))



# Function to prepare and save ISP data for a country
def prepare_country_data(file_path, country_name, dg, data_asn, data_org, data_sibling, output_directory):
    # Find the country name from the file path
    if country_name == 'The Us':
        df = prepare_us_data()
    else:
        # Read the Excel file into a DataFrame, skipping the first few rows with metadata
        df = pd.read_excel(file_path, sheet_name='Data', skiprows=4)

        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how='all')

        # Rename columns appropriately
        df.columns = ['ISP', 'Percentage', 'Unit']

        # Drop rows that are not relevant
        df = df.drop(columns=['Unit']).dropna()
    def get_asn_from_peeringdb(isp_name):
        if 'Other' == isp_name:
            return []
        list_of_asn = []
        if isp_name.startswith('Telmex'):
            isp_name = 'Telmex'
            list_of_asn = ['AS8151']
        elif isp_name.startswith('Vivo'):
            isp_name = 'TELEFÔNICA BRASIL'
            list_of_asn = ['AS26599']
        elif isp_name == 'Totalplay':
            isp_name = 'Total Play'
        elif isp_name == 'Telcel':
            list_of_asn = ['AS28403']
        elif isp_name == 'Spectrum':
            # it is actually Charter Communications
            isp_name = 'Charter'
            list_of_asn = ['AS21989', 'AS7843']
        elif isp_name == 'Tim' and country_name != 'Italy':
            isp_name = 'Telecom Italia BR'
            list_of_asn = ['AS26615']
        elif isp_name == 'Xfinity':
            isp_name = 'Comcast'
        if isp_name == 'Tim':
            isp_name = 'Telecom Italia'
            list_of_asn = ['AS16232']
        elif isp_name == 'Jio Fiber':
            list_of_asn = ['AS55836', 'AS64049']  # Adding Jio Fiber ASNs
        elif isp_name == 'Megacable':
            list_of_asn = ['AS13999', 'AS28541']
        elif isp_name == 'Excitel':
            list_of_asn = ['AS133982']  # Excitel ASNs
        elif isp_name == 'BSNL - Bharat Sanchar Nigam':
            list_of_asn = ['AS9829']  # BSNL ASN
        elif isp_name == 'ACT - Atria Convergence Technologies':
            list_of_asn = ['AS24309']  # ACT ASN
        elif isp_name == 'Asianet Broadband':
            list_of_asn = ['AS45415']  # Asianet ASN
        elif isp_name == 'Hathway Cable & Datacom':
            list_of_asn = ['AS17488']  # Hathway ASN
        elif isp_name == 'Beeline Internet':
            list_of_asn = ['AS3216']  # No ASNs listed
        elif isp_name == 'MTS Internet':
            list_of_asn = ['AS8359']  # No ASNs listed
        elif isp_name == 'Domru':
            list_of_asn = ['AS60043']  # No ASNs listed
        elif isp_name == 'Drei / 3':
            list_of_asn = ['AS25255']
        elif isp_name == 'J:COM':
            list_of_asn = ['AS9824', 'AS9617', 'AS4721', 'AS9614', 'AS7686', 'AS9378', 'AS18134', 'AS18136',
                           'AS23788',
                           'AS23790', 'AS24276', 'AS45675']  # J:COM ASNs
        elif isp_name == 'Eo Hikari':
            list_of_asn = ['AS23629']
        # isp_name = isp_name.splt(' ')[0]
        for item in data_asn:
            if isp_name.lower() in item['name'].lower():
                list_of_asn.append('AS' + str(item['asn']))
        for item in data_org:
            if isp_name.lower() in item['name'].lower():
                data_id = item['id']
                for item in data_asn:
                    if item['org_id'] == data_id:
                        list_of_asn.append('AS' + str(item['asn']))
        list_of_asn_with_sibling = deepcopy(list_of_asn)
        for asn in list_of_asn:
            if asn[2:] in data_sibling:
                for item in data_sibling[asn[2:]]['Sibling ASNs']:
                    list_of_asn_with_sibling.append('AS' + str(item))
        return list(set(list_of_asn_with_sibling))

    # transform the country name to an alpha-2 code
    print(country_name)
    if country_name == 'The Us':
        country_code = 'US'
    elif country_name == 'Russia':
        country_code = 'RU'
    elif country_name == 'The Uk':
        country_code = 'GB'
    elif country_name == 'South Korea':
        country_code = 'KR'
    else:
        country_code = pycountry.countries.get(name=country_name).alpha_2
    if country_name == 'The Uk':
        country_name = 'UK'
    elif country_name == 'The Us':
        country_name = 'USA'
    country_dict[country_name] = country_code
    dg_country = dg[dg['CC'] == country_code]
    # Add a new column for ASNs
    df['ASN'] = df['ISP'].apply(get_asn_from_peeringdb)
    df['Fraction of Users'] = df['ASN'].apply(lambda x: sum(dg_country[dg_country['AS'].isin(x)]['% of Country']))
    ### remove all the rows where ['ASN'] is empty
    df = df[df['ASN'].apply(lambda x: len(x) > 0)]
    ### find points where APNIC is lower than 0.5
    dg = df[df['Fraction of Users'] < 0.5]

    percentage_per_country_covered[country_name] = df['Fraction of Users'].sum()
    print(df['Fraction of Users'].sum())
    df['Fraction of Users'] = df['Fraction of Users'] / (df['Fraction of Users'].sum() / 100)
    df['Country'] = country_name
    df['Fraction of Users'] = df['ASN'].apply(lambda x: sum(dg_country[dg_country['AS'].isin(x)]['% of Country']))
    return df


# Function to read APNIC data
def read_apnic_file(path_to_file):
    datadb = []
    for file in os.listdir(path_to_file):
        if file.endswith('.csv') and '2024' in file and '04' in file:
            df = pd.read_csv(os.path.join(path_to_file, file), skiprows=1)
            df['ASN'] = df['AS'].apply(lambda x: int(x.split('AS')[1]))
            df['Date'] = pd.to_datetime(file.split('.')[1])
            datadb.append(df)
            break
    return pd.concat(datadb)

# Function to create a plot
def plot_comparison(parsed_dfs, percentage_per_country_covered):
    # sns.set_theme()
    fig, ax = plt.subplots()
    # Set the classic style for a MATLAB-like appearance
    sns.set_style('whitegrid')

    # Increase the size of the figure
    fig.set_size_inches(8, 8)

    # Increase the font size by 1.5x
    plt.rcParams.update({'font.size': 16})
    markers = ['o', 's', 'D', '^', 'P', 'X', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', '|', '_', 'x', '+', '.', ',', 's']
    i = 0

    # Remove all instances where AS is an empty list
    parsed_dfs = parsed_dfs[parsed_dfs['ASN'].apply(lambda x: len(x) > 0)]

    # Define colors using seaborn's color palette
    colors = sns.color_palette('tab20', n_colors=len(parsed_dfs['Country'].unique()))
    print(len(parsed_dfs['Country'].unique()))

    # Sort parsed_dfs by percentage_per_country_covered
    sorted_countries = sorted(parsed_dfs['Country'].unique(), key=lambda x: percentage_per_country_covered[x])
    # Initialize lists to store all data points
    all_percentages = []
    all_fractions = []
    country_r_squared = {}
    # Collect all data points from parsed_dfs for R^2 calculation
    for country in sorted_countries:
        data = parsed_dfs[parsed_dfs['Country'] == country]
        country_r_squared[country] = r2_score(data['Percentage'].values, data['Fraction of Users'].values)
        all_percentages.extend(data['Percentage'].values)
        all_fractions.extend(data['Fraction of Users'].values)
    country_r_squared['France'] *= -1
    labels = []
    for country in sorted_countries:
        data = parsed_dfs[parsed_dfs['Country'] == country]
        print(country)

        large_diff = data[np.abs(data['Percentage'] - data['Fraction of Users']) > 13]
        print(percentage_per_country_covered.keys())

        # Round the percentage to one decimal place for the legend
        percentage_rounded = round(percentage_per_country_covered[country], 1)

        if not large_diff.empty:
            print(f"\nLarge differences for {country}:")
            print(large_diff[['Percentage', 'Fraction of Users', 'ISP']])

        if percentage_per_country_covered[country] < 50:
            # Add a border for points with coverage < 50%
            ax.scatter(
                data['Percentage'], data['Fraction of Users'],
                marker=markers[i],
                label=fr"{country} ({percentage_rounded}%), $R^2$ = {round(country_r_squared[country], 2)}",
                # Use rounded percentage
                color=colors[i],
                edgecolor='black',  # Black border
                linewidth=1.5,  # Border thickness
                s=80
            )
        else:
            # Regular points
            ax.scatter(
                data['Percentage'], data['Fraction of Users'],
                marker=markers[i],
                label=f"{country} ({percentage_rounded:5.1f}%)".ljust(
                    20) + f" $R^2$ = {round(country_r_squared[country], 2):5.2f}",
                color=colors[i],
                s=80
            )

        # Add labels for large differences
        for _, row in large_diff.iterrows():
            labels.append(ax.text(row['Percentage'], row['Fraction of Users'], row['ISP'], fontsize=10, ha='right'))

        i += 1

    # Set labels with increased font size
    ax.set_xlabel('% of Country\'s Broadband Users According to Broadband Survey', fontsize=plt.rcParams['font.size'])
    ax.set_ylabel('% of Country\'s Broadband Users According to APNIC Dataset', fontsize=plt.rcParams['font.size'])

    from adjustText import adjust_text
    adjust_text(labels, arrowprops=dict(arrowstyle='-', color='black')
                )
    # Set percentage format for x and y ticks
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    legend = ax.legend(title='Country (% of Country\'s \n APNIC User Estimates \n in Broadband Survey Orgs)',
                       loc='upper right', bbox_to_anchor=(1.7, 1), fontsize=14, title_fontsize=16)
    ax.set_xticks(range(0, 65, 10))
    ax.set_yticks(range(0, 65, 10))

    # Draw a line that goes through the middle
    plt.plot([0, 65], [0, 65], color='black', linestyle='--', linewidth=1.5)
    # Add bbox_inches='tight' to savefig to prevent cutoff
    plt.savefig('../plot/broadband_comparison.pdf', bbox_inches='tight')


# Function to create a plot
def plot_us_histogram(dg_us):
    plt.hist(dg_us['% of Country'], bins=20)
    plt.xlabel('Percentage of Users')
    plt.ylabel('Number of ASNs')
    plt.title('Distribution of Percentage of Users for US ASNs')
    plt.text(0.4, 0.9, f'# of ASNs: {len(dg_us)}', transform=plt.gca().transAxes)
    plt.text(0.4, 0.8, f'# of ASNs with > 1% of users: {len(dg_us[dg_us["% of Country"] > 1])}',
             transform=plt.gca().transAxes)
    plt.savefig('histogram_us.pdf')


# Main function
def main():
    # Load PeeringDB data
    data = load_peeringdb_data(PDB_YEAR, PDB_MONTH, DIRECTORY)
    data_asn = data['net']['data']
    data_org = data['org']['data']

    # Load sibling ASN data
    data_sibling = load_sibling_data('../data/data_org/ii.as-org.v01.2024-01.json')

    # Read APNIC file
    path_to_apnic_dumps = '/Users/loqmansalamatian/Documents/extracted_files/data/'
    dg = read_apnic_file(path_to_apnic_dumps)
    # Prepare data for other countries
    directory = '/Users/loqmansalamatian/Documents/extracted_files/broadband/'
    parsed_dfs = []

    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") and filename.startswith('statistic_id'):
            country_name = extract_country_name(filename)
            file_path = os.path.join(directory, filename)
            parsed_df = prepare_country_data(file_path, country_name, dg, data_asn, data_org, data_sibling, directory)
            parsed_dfs.append(parsed_df)

    parsed_dfs = pd.concat(parsed_dfs)

    plot_comparison(parsed_dfs, percentage_per_country_covered)


if __name__ == "__main__":
    main()