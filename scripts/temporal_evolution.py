import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
from bs4 import BeautifulSoup
import json
from collections import defaultdict
from itertools import combinations
from scipy.stats import ks_2samp
from tqdm import tqdm
import numpy as np
import re
from ast import literal_eval
import matplotlib.pyplot as plt
from adjustText import adjust_text
from plotly.subplots import make_subplots
from datetime import datetime
import gzip
import os
import sys
import zstandard as zstd
import tarfile
import os
import warnings
import seaborn as sns
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def decompress_zst(file_path, output_path):
    # Decompress .zst to .tar
    with open(file_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with open(output_path, 'wb') as decompressed:
            dctx.copy_stream(compressed, decompressed)

def extract_tar(tar_path, extract_path):
    # Extract .tar file
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_path)

def main_tar():
    zst_file = '../Populations/data.tar.zst'  # Path to the .tar.zst file
    tar_file = '../Populations/data.tar'      # Path where the .tar will be saved
    extract_dir = '../extracted_files'  # Directory to extract the files

    # Create the directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Decompress .zst to .tar
    decompress_zst(zst_file, tar_file)

    # Extract files from .tar
    extract_tar(tar_file, extract_dir)

    print(f'Files have been extracted to {extract_dir}')


# Function to parse year and month from filename
def parse_year_month(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if match:
        date_part = match.group(0)
        year, month, _ = date_part.split('-')
        return (int(year), int(month))
    return None

def extract_data_from_html(html_content):

    # Find the JavaScript array using regex
    match = re.search(r"arrayToDataTable\((\[.*?\])\)", html_content, re.DOTALL)
    if match:
        # Extract and clean the data array
        data_string = match.group(1)
        data_string = data_string.replace('\n', '')  # Remove newlines
        data = literal_eval(data_string)  # Safely evaluate string as a Python literal

        # Convert to DataFrame
        columns = data[0]  # First row as column headers
        rows = data[1:]    # Remaining rows as data
        df = pd.DataFrame(rows, columns=columns)
        return df
    else:
        raise ValueError("Data array not found in HTML")
# Path to your credentials JSON file
CREDENTIALS_FILE = '/Users/loqmansalamatian/Documents/GitHub/event_detections/private_api/gdrive_api.json'

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_google_drive():
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    return build('drive', 'v3', credentials=creds)

def list_files(service, folder_id,timespan):
    query = f"'{folder_id}' in parents"
    results = []
    page_token = None

    while True:
        response = service.files().list(q=query,
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name)',
                                        pageToken=page_token,
                                        pageSize=1000  # You can adjust this value as needed
                                       ).execute()
        results.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    # Group files by year and month
    files_by_month = defaultdict(list)
    for entry in results:
        ym = parse_year_month(entry['name'])
        if ym:
            files_by_month[ym].append(entry)

    # Select one sample per month
    if timespan == 'monthly':
        monthly_samples = []
        for ym, files in files_by_month.items():
            # Sort files by date to get the last one of each month, you could also sort and get the first
            files_sorted = sorted(files, key=lambda x: x['name'])
            # monthly_samples[ym] = files_sorted[-1]  # Change to [0] if you want the first of the month
            monthly_samples.append(files_sorted[-1])
        return monthly_samples
    # elif timespan == 'yearly':
    #     yearly_samples = []
    #     for ym, files in files_by_month.items():
    #         # Sort files by date to get the last one of each month, you could also sort and get the first
    #         files_sorted = sorted(files, key=lambda x: x['name'])
    #         # monthly_samples[ym] = files_sorted[-1]  # Change to [0] if you want the first of the month
    #         yearly_samples.append(files_sorted[-1])
    #     return yearly_samples
    elif timespan == 'weekly':
        weekly_samples = []
        year_month_pair = '2024'
        for entry in results:
            if year_month_pair in entry['name']:
                weekly_samples.append(entry)
        return weekly_samples
    else :
        return results



def download_file(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    fh.seek(0)
    with open(file_name, 'wb') as f:
        f.write(fh.read())
    print(f"File {file_name} downloaded.")



def load_data_to_dataframe(data_array):
    if not data_array:
        return None

    # The first element of data_array contains column headers
    columns = data_array[0]
    # The rest are the data rows
    data = data_array[1:]
    df = pd.DataFrame(data, columns=columns)
    return df

def load_file_into_memory(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    fh.seek(0)

    # Load content into pandas DataFrame if it's a CSV
    if file_name.endswith('.db'):
        df = extract_data_from_html(fh.read().decode('utf-8'))
        df['ASN'] = df['ASN'].apply(lambda x: int(x.split('AS')[1]))
        df['CC'] = df['CC'].apply(lambda x: x.split('>')[1].split('<')[0] if '>' in x else x)
        df['Date'] = pd.to_datetime(('-').join(file_name.split('-')[1:4]).split('.')[0])
    return df

def check_years_in_folder(service, folder_id):
    files = list_files(service, folder_id)
    years = [int(file['name'].split('-')[1]) for file in files if file['name'].endswith('.db')]
    return years

def downloading_all_files():
    service = authenticate_google_drive()
    folder_id = '1tyK-uf0qUFBdN08XVCFMluLCMsh2WTwi'  # Folder ID from your URL
    files = list_files(service, folder_id, timespan = '')
    # print(check_years_in_folder(service, folder_id))
    print(len(files))
    datadb = []
    for file in files:
        if file['name'].endswith('.db'):
            try:
                datadb.append(load_file_into_memory(service, file['id'], file['name']))
            except:
                continue
    pd.concat(datadb).to_csv('data_march_2024.csv', index=False)
    # for file in files:
    #     if file['name'].endswith('.db'):
    #         download_file(service, file['id'], file['name'])

def parsing_all_files():
    datadb = []
    path_to_file = '/Users/loqmansalamatian/Documents/extracted_files/data/'
    files = os.listdir(path_to_file)
    for file in tqdm(files):
        if file.endswith('.csv') and any(year in file for year in ['2019', '2020', '2021', '2022', '2023']):
                # skip the first line of the file
                df = pd.read_csv(os.path.join(path_to_file,file), skiprows=1)
                df['ASN'] = df['AS'].apply(lambda x: int(x.split('AS')[1]))
                df['Date'] = pd.to_datetime(file.split('.')[1])
                df['Date'] = df['Date']
                datadb.append(df)
    dg = pd.concat(datadb)
    return dg


def studying_stats(df):
    from statsmodels.stats.anova import AnovaRM
    print(df.groupby(['ASN', 'Date']).size().unstack(fill_value=0))
    # Check how many samples each ASN has for each Date
    count_per_asn_date = df.groupby('ASN')['Date'].nunique()
    print("Count of Dates per ASN:\n", count_per_asn_date)

    # Identify the maximum number of unique dates available
    max_dates = df['Date'].nunique()
    print("Maximum number of unique dates:", max_dates)

    # Filter ASNs that have the same number of unique dates as the maximum found
    complete_asns = count_per_asn_date[count_per_asn_date == max_dates].index
    print("ASNs with complete data:", complete_asns)
    # Filter the DataFrame to include only complete ASNs
    filtered_df = df[df['ASN'].isin(complete_asns)]
    print("Filtered DataFrame:\n", filtered_df.shape)
    print('Count of Unique Dates:\n', filtered_df['Date'].nunique())
    print('The Unique Dates:\n', filtered_df['Date'].unique())
    print('Count of Dates per ASN:\n', filtered_df.groupby('ASN')['Date'].nunique())
    print('Count of unique ASN:\n', filtered_df['ASN'].nunique())
    print('Count of Countries:\n', filtered_df['CC'].nunique())

    # Identify top 10 ASNs in each country on each date.
    def top_asns_by_date(df):
        # Assuming 'Score' or a similar metric determines the top ASNs.
        return df.nlargest(10, 'Users (est.)')

    # Group by date and country, then apply the function.
    top_asns = filtered_df.groupby(['Date', 'CC']).apply(top_asns_by_date)
    top_asns.reset_index(drop=True, inplace=True)
    print("Top ASNs per Country per Date:\n", top_asns)

    # Focus on ASNs that appear at least once in the top 10.
    top_unique_asns = top_asns['ASN'].unique()
    filtered_df = filtered_df[filtered_df['ASN'].isin(top_unique_asns)]
    print('Count of unique ASN after Filtering:\n', filtered_df['ASN'].nunique())

    # Can we identify the ASN that appear at least in one date point in the top 10 of their countries and focus on that subset?
    # Fit the Repeated Measures ANOVA model
    # model = AnovaRM(filtered_df, depvar='Samples', subject='ASN', within=['Date'], aggregate_func='mean')
    # results = model.fit()
    # print(results.summary())
    # Get unique dates
    unique_dates = filtered_df['Date'].unique()

    # Prepare for KS Test
    ks_results = pd.DataFrame(columns=['Date1', 'Date2', 'KS Statistic', 'P-Value'])

    # Perform KS test on each pair of dates
    for date1, date2 in tqdm(combinations(unique_dates, 2)):
        data1 = filtered_df[filtered_df['Date'] == date1]['Users (est.)']
        data2 = filtered_df[filtered_df['Date'] == date2]['Users (est.)']

        ks_stat, p_value = ks_2samp(data1, data2)
        ks_results = ks_results.append({
            'Date1': date1,
            'Date2': date2,
            'KS Statistic': ks_stat,
            'P-Value': p_value
        }, ignore_index=True)

    print("KS Test Results between each pair of dates:")
    print(ks_results)

from copy import deepcopy
from datetime import datetime
granularities = {'Days': 1, 'Weeks': 7, 'Months': 30, 'Years': 365}

def studying_stats_country(df, renew= True, granularity = 'Weeks'):
    # check if the ks_results exists
    if os.path.exists(f'../ks_results_{granularity}.csv') and not(renew):
        ks_results = pd.read_csv(f'../ks_results_{granularity}.csv', index_col = 0)
    else:
        print(df.groupby(['ASN', 'Date']).size().unstack(fill_value=0))

        # count_per_asn_date = df.groupby('ASN')['Date'].nunique()
        # max_dates = df['Date'].nunique()
        # complete_asns = count_per_asn_date[count_per_asn_date == max_dates].index
        # filtered_df = df[df['ASN'].isin(complete_asns)]
        filtered_df = df
        ### Trivial
        def top_asns_by_date(df):
            return df.nlargest(25, 'Users (est.)')

        # top_asns = filtered_df.groupby(['Date', 'CC']).apply(top_asns_by_date)
        # top_asns.reset_index(drop=True, inplace=True)
        # top_unique_asns = top_asns['ASN'].unique()
        # filtered_df = filtered_df[filtered_df['ASN'].isin(top_unique_asns)]

        # Prepare KS Test DataFrame
        ks_results = pd.DataFrame(columns=['Country', 'Date1', 'Date2', 'KS Statistic', 'P-Value'])

        # Get unique countries
        countries = filtered_df['CC'].unique()
        # Perform KS test on each pair of dates within each country
        for country in tqdm(countries):
            # if country != 'US':
            #     continue
            df_country = filtered_df[filtered_df['CC'] == country]
            df_country = df_country.drop_duplicates(subset=['ASN', 'Date'])
            unique_dates = sorted(df_country['Date'].unique())  # Sort the dates to ensure they are consecutive
            # current_date = 0
            for i in range(len(unique_dates) - 1):
                date1 = unique_dates[i]
                date2 = unique_dates[i + 1]
            # for date1, date2 in combinations(unique_dates, 2):
            # for date1, date2 in combinations(unique_dates, 2):
                data1 = df_country[df_country['Date'] == date1].set_index('ASN').sort_index()
                data2 = df_country[df_country['Date'] == date2].set_index('ASN').sort_index()

                if data1.shape[0] < 5 or data2.shape[0] < 5:
                    continue

                # subtracting the overlapping rows between data1 and data2
                data_diff = data2.copy()
                data_diff['Users (est.)'] = data2['Users (est.)'].subtract(data1['Users (est.)'], fill_value=0)
                # data_diff = data_diff.dropna()
                data_diff['Date Before'] = data1['Date']
                # Convert the dates to datetime
                data_diff['Date'] = pd.to_datetime(data_diff['Date'])
                data_diff['Date Before'] = pd.to_datetime(data_diff['Date Before'])
                # Interval of time between both in days
                data_diff['Interval'] = (data_diff['Date'] - data_diff['Date Before']).dt.days
                ks_stat, p_value = ks_2samp(data1['Users (est.)'], data2['Users (est.)'])
                data_diff['Ratio'] = data_diff['Users (est.)'] / data1['Users (est.)']
                ### plot the distribution of data1 and data2
                regularized_max = max(data_diff['Users (est.)']/data1['Users (est.)'])
                regularized_min = min(data_diff['Users (est.)']/data1['Users (est.)'])
                regularized_mean = data_diff['Ratio'].mean()
                regularized_median = data_diff['Ratio'].median()
                # Assuming ks_results is defined somewhere above as a DataFrame
                new_row = pd.DataFrame({
                    'Country': [country],
                    'Date1': [date1],
                    'Date2': [date2],
                    'KS Statistic': [ks_stat],
                    'P-Value': [p_value],
                    'Max Diff': max(data_diff['Users (est.)']),
                    'Min Diff': min(data_diff['Users (est.)']),
                    'Mean Diff': data_diff['Users (est.)'].mean(),
                    'Median Diff': data_diff['Users (est.)'].median(),
                    'Max Relative Diff': regularized_max,
                    'Min Relative Diff': regularized_min,
                    'Mean Relative Diff': regularized_mean,
                    'Median Relative Diff': regularized_median
                })
                # if data_diff['Interval'].min() >= granularities[granularity]:
                #     current_date = deepcopy(i)
                # else:
                #     continue
                # data_diff = data_diff[abs(data_diff['Ratio'])> 1.5]

                # data_diff.sort_values(by='Users (est.)', ascending=False, inplace=True)
                # if data_diff.shape[0] == 0:
                #     continue
                if ks_stat > 0.86 and p_value < 0.05:
                    # print('?!')
                    # ### print a bunch of stats to understand the data
                    # print(data_diff)
                    # print(f'KS Statistic: {ks_stat}, P-Value: {p_value}')
                    # print(f'Max Diff: {max(data_diff["Users (est.)"])}')
                    # print(f'Min Diff: {min(data_diff["Users (est.)"])}')
                    # print(f'Mean Diff: {data_diff["Users (est.)"].mean()}')
                    # print(f'Median Diff: {data_diff["Users (est.)"].median()}')
                    # print(f'Max Relative Diff: {regularized_max}')
                    # print(f'Min Relative Diff: {regularized_min}')
                    # print(f'Mean Relative Diff: {regularized_mean}')
                    # date1 = pd.to_datetime(str(date1)).strftime('%Y.%m.%d')
                    # date2 = pd.to_datetime(str(date2)).strftime('%Y.%m.%d')
                    # print(f'Date 1: {date1}')
                    # print(f'Date 2: {date2}')
                    plt.figure(figsize=(10, 6))

                    # # Get the range of data and set bins
                    # all_data = np.concatenate((data1['Users (est.)'], data2['Users (est.)']))
                    # bins = np.linspace(min(all_data), max(all_data), 30)
                    #
                    # # Histogram for data1
                    # counts1, _, _ = plt.hist(data1['Users (est.)'], bins=bins, alpha=0.5, label='data1', color='blue')
                    #
                    # # Histogram for data2
                    # counts2, _, _ = plt.hist(data2['Users (est.)'], bins=bins, alpha=0.5, label='data2', color='red')
                    #
                    # # Clear the previous histograms and plot side by side
                    # plt.clf()
                    #
                    # # Calculate the width of each bar
                    # width = np.diff(bins)[0] / 2
                    #
                    # data_diff.to_csv(f'../data_diff/data_diff_{country}_{date1}_{date2}_{granularity}.csv')
                    #
                    # # Bar plots
                    # plt.bar(bins[:-1], counts1, width=width, align='center', alpha=0.5, label={date1}, color='blue')
                    # plt.bar(bins[:-1] + width, counts2, width=width, align='center', alpha=0.5, label={date2},
                    #         color='red')
                    #
                    # # Add titles and labels
                    # plt.title(f'Distribution of Eyeballs in {country}')
                    # plt.xlabel('Users (est.)')
                    # plt.ylabel('Frequency')
                    # plt.legend()
                    # plt.savefig(f'../plot/distribution_of_eyeballs/distribution_of_eyeballs_{country}_{date1}_{date2}_{granularity}.png')
                    # KDE plot for data1
                    sns.kdeplot(data1['Users (est.)'], color='blue', label=f'{date1} data', shade=True, clip=(0, None))

                    # KDE plot for data2
                    sns.kdeplot(data2['Users (est.)'], color='red', label=f'{date2} data', shade=True,clip=(0, None))

                    # Save the diff data to CSV
                    data_diff.to_csv(f'../data_diff/data_diff_{country}_{date1}_{date2}_{granularity}.csv')

                    # Add titles and labels
                    plt.title(f'Distribution of Eyeballs in {country}')
                    plt.xlabel('Users (est.)')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.savefig(
                        f'../plot/distribution_of_eyeballs/distribution_of_eyeballs_{country}_{date1}_{date2}_{granularity}.png')
                    plt.show()
                ks_results = pd.concat([ks_results, new_row], ignore_index=True)
        ks_results.to_csv(f'../ks_results_{granularity}.csv')
    # Plotting the CDF of the KS Statistic column
    ks_statistics = ks_results['KS Statistic']
    print(ks_results.columns)
    # Count instances where the p-value is less than 0.05
    significant_results = ks_results[(ks_results['P-Value'] < 0.05) | (ks_results['KS Statistic'] > 0.3)]
    print("Significant Results:\n", significant_results)
    # Calculate the CDF
    print('Countries in Significant Results:', significant_results['Country'].unique())
    values, base = np.histogram(ks_statistics, bins=40, density=True)
    cumulative = np.cumsum(values)
    cumulative = cumulative / cumulative[-1]  # Normalize
    plt.figure(figsize=(10, 6))
    # Create the CDF plot
    plt.plot(base[:-1], cumulative, c='blue')
    plt.title(f'CDF of KS Statistics in {granularity}')
    plt.xlabel('KS Statistic')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.savefig(f'../plot/ks_statistics/cdf_ks_statistics_{granularity}.png')
    val  = ks_results.nlargest(10, 'Max Diff')
    # Remove the top 1st percentile and top 99th percentile
    ks_results = ks_results
    # [(ks_results['Max Diff'] > ks_results['Max Diff'].quantile(0.01)) &
    #                         (ks_results['Max Diff'] < ks_results['Max Diff'].quantile(0.99)) & (ks_results['Min Diff'] > ks_results['Min Diff'].quantile(0.01)) &
    #                         (ks_results['Min Diff'] < ks_results['Min Diff'].quantile(0.99)) & (ks_results['Mean Diff'] > ks_results['Mean Diff'].quantile(0.01))]
    ### print top-10 differences
    print(val)
    # Also plot the CDF of mean-median-max-min differences using different colors and diffeent markers
    fig, ax = plt.subplots()
    for i, (statistic, color, hatch) in enumerate(zip(['Median Relative Diff', 'Median Diff'],
                                                        ['red', 'green', 'purple', 'orange'],
                                                        ['v', '*', '.', 'x'])):
            values, base = np.histogram(ks_results[statistic], bins=40, density=True)
            cumulative = np.cumsum(values)
            cumulative = cumulative / cumulative[-1]
            ax.plot(base[:-1], cumulative, c=color, label=statistic, linestyle='-', marker=hatch, markersize=5)
    ax.set_title(f'{granularity}')
    ax.legend()
    plt.savefig('../plot/ks_statistics/diff_metrics.png')

    # print("KS Test Results between each pair of dates for each country:")
    # print(ks_results)

    return ks_results

def mapping_as_to_org(year):
    filename = f"../as_to_org/{year}0101.as-org2info.txt.gz"
    date = datetime.strptime(filename, '../as_to_org/%Y%m%d.as-org2info.txt.gz')
    with gzip.open(filename, "r") as fin:
        content = fin.readlines()
        index_org = \
        [x for x in range(len(content)) if "# format:org_id|changed|org_name|country|source" in str(content[x])][0]
        index_asn = \
        [x for x in range(len(content)) if "# format:aut|changed|aut_name|org_id|opaque_id|source" in str(content[x])][
            0]
    print(index_org, index_asn, index_asn - index_org - 2)
    org_df = pd.read_csv(filename, delimiter="|", skiprows=index_org, nrows=index_asn - index_org - 2).rename(
        columns={"# format:org_id": "org_id"})[["org_id", "org_name", "country", "source"]]
    if year == 2021 or year == 2020 or year == 2019:
        index_asn = index_asn - 1
    asn_df = pd.read_csv(filename, delimiter="|", skiprows=index_asn).rename(
        columns={"# format:aut": "ASN", "aut_name": "asn_name"})[["ASN", "asn_name", "org_id"]]
    asn_df.merge(org_df, on="org_id").to_csv(str(date.date()) + '_org2info.csv', index=False)
    return asn_df

def studying_number_of_as_per_country_to_reach_n(df,n=0.95):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date']>'2019-01-01']
    df['Users (est.)'] = df['Users (est.)'].astype(float)
    df['Year'] = df['Date'].dt.year  # Extract year once

    # Return the Year
    years = df['Date'].dt.year.unique()

    # Do AS-to-Org mapping for that specific date
    all_asn_to_org = pd.DataFrame()
    for year in years:
        df_as_to_org = mapping_as_to_org(year)
        df_as_to_org['Year'] = year  # Add year to the mapping DataFrame
        all_asn_to_org = pd.concat([all_asn_to_org, df_as_to_org], ignore_index=True)

    # Merge df with the complete ASN-to-Org DataFrame based on 'ASN' and 'Year'
    df = pd.merge(df, all_asn_to_org, how='left', on=['ASN', 'Year'], suffixes=(None, '_org'))

    # grouped = df.groupby(['CC', 'Date', 'Org-ID'])['Users (est.)'].sum().reset_index()
    df = df.drop(columns = ['ASN', 'AS Name', 'asn_name'])
    # Ensure no zero values to avoid log(0) issues
    df['Samples'] = df['Samples'].replace(0, np.nan)
    df['Users (est.)'] = df['Users (est.)'].replace(0, np.nan)

    # Compute log(Samples)/log(User Estimates) per day and country
    df['log_samples'] = np.log(df['Samples'])
    df['log_user_estimates'] = np.log(df['Users (est.)'])

    # Calculate the ratio of log(Samples) to log(User Estimates)
    df['elasticity_coefficient'] = df['log_samples'] / df['log_user_estimates']

    # Drop the temporary columns if you don't need them anymore
    df.drop(columns=['log_samples', 'log_user_estimates'], inplace=True)

    # Find all country-year pairs where elasticity coefficient < 1.2
    filtered_df = df[df['elasticity_coefficient'] <= 1.1].dropna(subset=['elasticity_coefficient'])

    # Get unique (Country, Year) pairs
    country_year_pairs = filtered_df[['CC', 'Year']].drop_duplicates()

    # Count the number of countries per year
    country_year_pairs_count = country_year_pairs.groupby('Year').size().reset_index()
    print(country_year_pairs_count)
    # Group by 'CC', 'Date', and 'ASN' then sum the 'Users (est.)'
    grouped = df.groupby(['CC', 'Date', 'org_id'])['Users (est.)'].sum().reset_index()

    # Sort within each group by 'Users (est.)' in descending order
    grouped.sort_values(by=['CC', 'Date', 'Users (est.)'], ascending=[True, True, False], inplace=True)

    # Calculate the cumulative sum of 'Users (est.)' within each country and date
    grouped['cumulative_users'] = grouped.groupby(['CC', 'Date'])['Users (est.)'].cumsum()

    # Calculate the total users per country and date
    total_users = grouped.groupby(['CC', 'Date'])['Users (est.)'].sum().reset_index()
    total_users.rename(columns={'Users (est.)': 'total_users'}, inplace=True)

    # Merge cumulative sums with total users
    merged = pd.merge(grouped, total_users, on=['CC', 'Date'])

    # Calculate the cumulative percentage
    merged['cumulative_percentage'] = merged['cumulative_users'] / merged['total_users']

    print(merged[merged['CC']== 'BR'].head())
    # Determine the first instance where cumulative percentage exceeds n
    def get_threshold_reach(df):
        return df[df['cumulative_percentage'] >= n]['org_id'].iloc[0] if any(df['cumulative_percentage'] >= n) else None

    result = merged.groupby(['CC', 'Date']).apply(get_threshold_reach).reset_index()
    result.rename(columns={0: 'Threshold ASN'}, inplace=True)
    print(result.head())
    # Count how many ASes are needed to reach 95% per country and date
    final_res = merged.groupby(['CC', 'Date']).apply(lambda x: (x['cumulative_percentage'] < n).sum() + 1)
    print(result.shape)
    return final_res, country_year_pairs

def studying_specific_as_per_country_to_reach_n(df, n=0.95, metric = 'Users (est.)'):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Users (est.)'] = df[metric].astype(float)

    # Group by 'CC', 'Date', and 'ASN' then sum the 'Users (est.)'
    grouped = df.groupby(['CC', 'Date', 'ASN', 'AS Name'])[metric].sum().reset_index()

    # Sort within each group by 'Users (est.)' in descending order
    grouped.sort_values(by=['CC', 'Date', metric], ascending=[True, True, False], inplace=True)

    # Calculate the cumulative sum of 'Users (est.)' within each country and date
    grouped['cumulative_users'] = grouped.groupby(['CC', 'Date'])[metric].cumsum()

    # Calculate the total users per country and date
    total_users = grouped.groupby(['CC', 'Date'])[metric].sum().reset_index()
    total_users.rename(columns={metric: 'total_users'}, inplace=True)

    # Merge cumulative sums with total users
    merged = pd.merge(grouped, total_users, on=['CC', 'Date'])

    # Calculate the cumulative percentage
    merged['cumulative_percentage'] = merged['cumulative_users'] / merged['total_users']

    # Filter out ASes that contribute to the first 95% of the user base
    filtered_as = merged[merged['cumulative_percentage'] <= n]

    return filtered_as

def plot_time_series(df, country_code, metric = 'Users (est.)', percentile = 95):
    """ Plot the time-series for user population of each AS for a specific country. """
    # country_data = df[df['CC'] == country_code]
    #
    # plt.figure(figsize=(14, 7))
    # for asn in country_data['ASN'].unique():
    #     asn_data = country_data[country_data['ASN'] == asn]
    #     plt.plot(asn_data['Date'], asn_data['Users (est.)'], marker='o', label=f'ASN {asn}')
    #
    # plt.title(f"User Population Over Time for ASes in {country_code} Contributing to 95%")
    # plt.xlabel('Date')
    # plt.ylabel('User Population')
    # plt.legend(title='ASN', loc='upper left', bbox_to_anchor=(1,1))
    # plt.grid(True)
    # plt.show()

    """ Plot the interactive time-series for user population of each AS for a specific country using Plotly. """
    country_data = df[df['CC'] == country_code]

    fig = go.Figure()
    sum_data = country_data.groupby('Date')[metric].sum().reset_index()

    # Add a trace for each ASN
    for asn in country_data['ASN'].unique():
        asn_data = country_data[country_data['ASN'] == asn]
        ### count the number of unique ASes per date
        total_asn = asn_data.groupby('Date')['ASN'].nunique().reset_index()
        res = total_asn.sort_values(by = ['ASN']).head(1)['ASN'].iloc[0]
        if res > 1:
            print('?!')
        print(asn_data.columns)
        fig.add_trace(go.Scatter(
            x=asn_data['Date'],
            y=asn_data[metric],
            mode='markers',
            name=f'ASN {asn} + {asn_data["AS Name"].iloc[0]}',
        ))
    # Add a line to the sum of all ASes
    fig.add_trace(go.Scatter(
        x=sum_data['Date'],
        y=sum_data[metric],
        mode='lines',
        name=metric,
        line=dict(color='black', width=2, dash='dash')
    ))



    # Update plot layout
    fig.update_layout(
        title=f'{metric} Over Time for ASes in {country_code} Contributing to {percentile}%',
        xaxis_title='Date',
        yaxis_title=metric,
        legend_title='ASN',
        legend=dict(
            x=3,
            xanchor='right',
            y=1
        ),
        hovermode='closest'
    )
    ## save the html file
    fig.write_html(f'../plot/time_series_{country_code}_{metric}_{percentile}.html')
    # fig.show()


import plotly.graph_objs as go
import pandas as pd
import pycountry
import pycountry_convert as pc
from collections import defaultdict


def country_to_continent(country_alpha2):
    try:
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name
    except:
        return "Unknown"

def plotting_ks():
    plt.figure(figsize=(8, 4))  # Set the figure size for better readability

    # Assuming 'granularities' is defined; if not, you should define it.
    for granularity in granularities.keys():
        # Load the data
        file_path = f'../ks_results_{granularity}.csv'

        print(file_path)
        df = pd.read_csv(file_path, index_col=0)

        # Extract the KS Statistics column
        ks_statistics = df['KS Statistic'].values

        # Calculate the CDF values
        sorted_data = np.sort(ks_statistics)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        # Plot the CDF
        plt.plot(sorted_data, yvals, label=f'{granularity}')

    plt.xlabel('KS Statistic')
    plt.ylabel('Fraction of ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../plot/ks_statistics/cdf_ks_statistics.pdf')


def plot_yearly_worldheatmap(df, start_year, end_year, country_year_pairs, ratio=True):
    # Converting country codes from alpha-2 to alpha-3
    countries = {country.alpha_2: country.alpha_3 for country in pycountry.countries}
    # Calculate the base values for each country in 2019
    base_year_data = df[df.index.get_level_values('Date').year == 2019].groupby(level=0).median()
    # Store results for mapping from year to percentage change
    percentage_changes = {}
    # Prepare a subplot layout
    # fig = make_subplots(
    #     rows=1, cols=3,
    #     subplot_titles=[f"{year}" for year in range(end_year - 2, end_year + 1)],
    #     specs=[[{"type": "choropleth"}] * 3]  # Specify the type of plots for each subplot
    # )
    # col_index = 1
    hatch_color = '#000000'  # Color to represent missing countries

    for year in range(start_year + 1, end_year + 1):
        yearly_data = df[df.index.get_level_values('Date').year == year].groupby(level=0).median()
        if yearly_data.empty:
            continue
        if ratio:
            # Calculate the percentage change compared to the base year (2019)
            yearly_percentage_change = yearly_data.div(base_year_data).subtract(1).multiply(100)

            # Handling countries not present in 2019 or with no data
            yearly_percentage_change.fillna(0, inplace=True)  # Assuming 0% change if no data available

            # Transforming country codes to ISO-3
            yearly_percentage_change.index = [countries.get(index, 'Unknown') for index in yearly_percentage_change.index]
            yearly_percentage_change = yearly_percentage_change.reset_index()
            yearly_percentage_change.columns = ['country', 'percentage_change']

            # Store for plotting later
            percentage_changes[year] = yearly_percentage_change
        else:
            yearly_data.index = [countries.get(index, 'Unknown') for index in yearly_data.index]
            percentage_changes[year] = yearly_data.reset_index()
            percentage_changes[year].columns = ['country', '95th_percentile']

        # # Adding to subplot
        # fig.add_trace(
        #     go.Choropleth(
        #         locations=yearly_percentage_change['country'],
        #         z=yearly_percentage_change['percentage_change'],
        #         colorscale=[
        #             [0.0, "red"],  # decrease
        #             [0.5, "white"],  # neutral
        #             [1.0, "green"],  # increase
        #         ],
        #         locationmode='ISO-3',
        #         text=yearly_percentage_change['country'],  # showing country codes on hover
        #         colorbar=dict(title="Percentage Change"),
        #         showscale=(col_index == 3)  # Only show scale for last plot
        #     ),
        #     row=1, col=col_index
        # )
        # col_index += 1
    # Plotting the results
    for year, data in percentage_changes.items():
        if ratio:
            # Filtering out extreme values for better visualization by mapping elements above 300 to 300
            data['percentage_change'] = data['percentage_change'].clip(upper=300)
            data['percentage_change'] = data['percentage_change'].clip(lower=-100)
            # Get the countries that appear in the country_year_pairs for the current year (in ISO-2)
            countries_in_year_iso2 = country_year_pairs[country_year_pairs['Year'] == year]['CC'].tolist()
            # Convert the ISO-2 codes to ISO-3 for plotting
            countries_in_year_iso3 = [countries.get(cc, 'Unknown') for cc in countries_in_year_iso2]

            # Identify missing countries by subtracting the present ones from all available countries
            all_countries = set(countries.values())
            missing_countries = list(all_countries - set(countries_in_year_iso3))
            print(missing_countries)
            fig = go.Figure(
                data=[
                    go.Choropleth(
                        locations=data['country'],
                        z=data['percentage_change'],
                        zmin=-100,
                        zmax=300,
                        colorscale=[
                            [0.0, "red"],
                            [0.25, "white"],
                            [1.0, "blue"]
                        ],
                        locationmode='ISO-3',
                        text=data['country'],  # showing country codes on hover
                        colorbar=dict(title="Percentage Change")
                    ),
                    go.Choropleth(
                        locations=missing_countries,
                        z=[0] * len(missing_countries),  # Assign no data to missing countries
                        colorscale=[[0, hatch_color], [1, hatch_color]],  # Set color for missing countries
                        locationmode='ISO-3',
                        showscale=False,  # No color scale for missing countries
                        hoverinfo='none',  # No hover info for missing countries
                    )
                ],
                layout=go.Layout(
                    title=f"World Heatmap for Year {year} (% Change Since 2019)",
                    geo=dict(
                        scope='world',
                        projection=dict(type='natural earth'),
                        showlakes=True,
                        lakecolor='rgb(255, 255, 255)',
                        landcolor='lightgray',
                        showland=True,
                        showcountries=True,
                        countrycolor='gray'
                    )
                )
            )
            # fig = go.Figure(
            #     data=[go.Choropleth(
            #         locations=data['country'],
            #         z=data['percentage_change'],
            #         zmin = -100,
            #         zmax = 300,
            #         colorscale=[
            #             [0.0, "red"],  # Minimum value, -100
            #             [0.25, "white"],  # 0
            #             [1.0, "green"]  # Maximum value, 300
            #         ],
            #         locationmode='ISO-3',
            #         text=data['country'],  # showing country codes on hover
            #         colorbar=dict(title="Percentage Change")
            #     )],
            #     layout=go.Layout(
            #         title=f"World Heatmap for Year {year} (% Change Since 2019)",
            #         geo=dict(
            #             scope='world',
            #             projection=dict(type='natural earth'),
            #             showlakes=True,
            #             lakecolor='rgb(255, 255, 255)',
            #             landcolor='lightgray',
            #             showland=True,
            #             showcountries=True,
            #             countrycolor='gray'
            #         )
            #     )
            # )
            fig.update_layout(
                margin={"r": 0, "t": 50, "l": 0, "b": 0},
                font=dict(family="Arial", size=25)
            )

            # fig.write_html(f'../worldmap_{year}_evolution_95th.html')
            fig.write_image(f'../worldmap_{year}_evolution_95th.pdf', width=1800,
                            height=600)  # Adjust dimensions as needed

        else:
            print(data)
            fig = go.Figure(
                data=[go.Choropleth(
                    locations=data['country'],
                    z=data['95th_percentile'],
                    # colorscale=,
                    locationmode='ISO-3',
                    text=data['country'],  # showing country codes on hover
                    colorbar=dict(title="Number of ASes to reach 95th percentile")
                )],
                layout=go.Layout(
                    title=f"World Heatmap for Year {year}",
                    geo=dict(
                        scope='world',
                        projection=dict(type='natural earth'),
                        showlakes=True,
                        lakecolor='rgb(255, 255, 255)',
                        landcolor='lightgray',
                        showland=True,
                        showcountries=True,
                        countrycolor='gray'
                    )
                )
            )
            fig.update_layout(
                margin={"r": 0, "t": 50, "l": 0, "b": 0},
                font=dict(family="Arial", size=25)
            )
            fig.write_html(f'../worldmap_{year}_95th.html')

        # fig.show()
        # fig.write_image(f'../worldmap_{year}_evolution_95th.pdf', width=1800, height=600)  # Adjust dimensions as needed

# def studying_min_samples():

def studying_what_is_the_linear_relationship_per_country(df):
    # Group by 'Date' and 'CC' (Country Code) directly to avoid redundant filtering
    grouped = df.groupby(['Date', 'CC'])

    def calculate_ratio(group):
        if len(group) > 0:
            return (np.log(group['Users (est.)']) / np.log(group['Samples']).to_list()[0])
        else:
            return None

    result = grouped.apply(calculate_ratio)
    # Optionally, convert the result to a DataFrame for easier handling and visualization
    result_df = result.reset_index(name='Ratio')
    ### plot the ratio for a specific date
    # result_df = result_df[result_df['Date'] == '2021-01-01']
    print(result_df)
    result_df.to_csv('../ratio_total.csv')





if __name__ == '__main__':
    # df = parsing_all_files()
    # main_tar()
    # downloading_all_files()
    df= pd.read_csv('all_data.csv')
    # # find the dates in each year
    # df = df[df['Date'].str.contains('2024|2023|2022|2021|2020')]
    # # print(df['Date'].unique())
    # # ### print a list of unique ASes that appears per country
    # dict_as_per_country = df.groupby('CC')['ASN'].unique().to_dict()
    # for key in df.groupby('CC')['ASN'].unique().to_dict().keys():
    #     dict_as_per_country[key] = list(map(int, dict_as_per_country[key]))
    # # save the dictionary in a json
    # with open('dict_as_per_country.json', 'w') as f:
    #     json.dump(dict_as_per_country, f)
    df['Users (est.)'] = pd.to_numeric(df['Users (est.)'], errors='coerce')
    df['Samples'] = pd.to_numeric(df['Samples'], errors='coerce')
    print(df.shape)
    df = df.dropna(subset=['Users (est.)'])
    df = df.dropna(subset=['Samples'])
    # print(df.shape)
    # plotting_ks()
    # studying_what_is_the_linear_relationship_per_country(df)
    # # ### print ASes per country
    # # dg = df.groupby(['CC', 'ASN']).size().unstack(fill_value=0)
    # # number_of_ASes_per_country = dg.apply(lambda x: '1' if x[x!=0].count().sum() == 1 else '2-5' if x[x!=0].count().sum() >= 2 and x[x!=0].count().sum() <= 5 else '5+')
    # # print(number_of_ASes_per_country)
    # # ### Count the number of unique non 0 ASes per country
    # # number_of_ASes_per_country = dg.apply(lambda x: x[x != 0].count(), axis=1)
    # # print(number_of_ASes_per_country)
    # ### Count the number of ASes present in only one country, between 2 and 5 countries and more than countries
    # # print(number_of_ASes_per_country)
    ######
    # for granul in ['Years', 'Months','Days']:
    #     # Select one date per granularity
    #     if granul == 'Years':
    #         # Selecting the first date of each year present in the dataset
    #         dg = df[df['Date'].dt.is_month_start & (df['Date'].dt.month == 1)]
    #         ### sort by date
    #         print(dg['Date'].unique())
    #     elif granul == 'Months':
    #         # Selecting the first day of each month for all available years
    #         dg = df[df['Date'].dt.is_month_start]
    #     elif granul == 'Weeks':
    #         # Selecting the first day of each week for all available years
    #         dg = df[df['Date'].dt.isocalendar().day == 1]
    #     elif granul == 'Days':
    #         # Selecting every day
    #         dg = df.copy()  # This effectively selects all days in the dataset
    #         dg = dg[dg['Date'] > '2019-01-01']
    #     dg = dg.sort_values(by='Date')
    #     # Call the function studying_stats_country with selected data
    #     studying_stats_country(dg, renew=True, granularity=granul)
    ########

    # # df = df[df['Date'].str.contains('2024|2023|2022|2021')]
    # studying_stats_country(df, renew=True, granularity='Days')

    # # studying_stats_country(df, renew=True, granularity = 'Weeek')
    # # print(unqiue_dates)
    # # Print the unique dates
    # # print(df['Date'].unique())
    # # print(df.head())
    # # # df = df[df['CC'] == 'US']
    # result = studying_specific_as_per_country_to_reach_n(df, n=0.75)
    # result_samples = studying_specific_as_per_country_to_reach_n(df, n=0.75, metric='Samples')
    # # # # print(result)
    # for cc in ['FR', 'CN', 'US', 'RU', 'TR', 'UA', 'IQ', 'BR']:
    #     plot_time_series(result, cc, percentile=75)  # Replace 'US' with the desired country code
    #     plot_time_series(result_samples, cc, metric='Samples', percentile=75)
    # # result = studying_specific_as_per_country_to_reach_n(df, n=0.9, metric='Samples')
    # # plot_time_series(result, 'IR', metric= 'Samples')
    #
    df, country_year_pairs = studying_number_of_as_per_country_to_reach_n(df,n=0.95)
    print(country_year_pairs)
    # # # # df = df.reset_index()
    # # # # print(df.head())
    # # # # print(df.reset_index())
    # # # # Example usage assuming df is your DataFrame
    # # # # df['Date'] = pd.to_datetime(df['Date'])  # Ensuring the 'Date' column is of datetime type
    # # # df.to_csv('../number_of_as_per_country_to_reach_95.csv', index=True)
    # df = pd.read_csv('../number_of_as_per_country_to_reach_95.csv')
    plot_yearly_worldheatmap(df, 2019, 2024, country_year_pairs, ratio=True)