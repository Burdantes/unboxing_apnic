import re
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import pycountry
from scipy.stats import kendalltau, linregress
import statsmodels.api as sm
import seaborn as sns
import numpy as np
# Import matplotlib for color mapping
import matplotlib.colors as mcolors
from adjustText import adjust_text
def extract_data_from_html(file_path):
    with open(file_path, 'r') as file:
        html_content = file.read()

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

kind = 'traffic'
# Example usage
file_path = '../data_CloudFlare/apnicDatabaseBackup-2024-08-09.html'
df = extract_data_from_html(file_path)
df.columns = ['Rank', 'asn', 'AS Name', '#cc', 'Users (est.)', '% of country',
       '% of Internet', 'Samples']
print(df.columns)
df['#cc'] = df['#cc'].apply(lambda x: x.split('>')[1])
# print(df_apnic['#cc'].head())
df['#cc'] = df['#cc'].apply(lambda x: x.split('<')[0])
# print(df.head())
# df.to_csv('../data_CloudFlare/apnicDatabaseBackup-2023-10-19.csv')
# path_to_file = '/Users/loqmansalamatian/Documents/extracted_files/data/aspop.2024-04-01.csv'
# df = pd.read_csv(path_to_file, skiprows=1)
# df.columns = ['Rank', 'asn', 'AS Name', '#cc', 'Users (est.)', '% of country',
#        '% of Internet', 'Samples']
# file_path = '../data_CloudFlare/-2024-05-03.csv'

if kind == 'user_agents':
    # file_path_cloudflare = '../data_CloudFlare/userAgent-fractions-perASN-perCountry_2024-05-03.csv'
    # df_cloudflare = pd.read_csv(file_path_cloudflare)
    # df_cloudflare.columns = ['asn', '#cc', 'fraction']
    # df_cloudflare.columns = ['asn', '#cc', 'fraction']
    # # Create a DataFrame
    #
    # # Sorting by country code and fraction in descending order
    # df_cloudflare.sort_values(by=['#cc', 'fraction'], ascending=[True, False], inplace=True)
    # df_cloudflare['fraction'] *= 100  # Convert fraction to percentage
    # # Ranking within each country
    # df_cloudflare['rank'] = df_cloudflare.groupby('#cc')['fraction'].rank(ascending=False, method='min').astype(int)
    file_path_cloudflare = '../data_CloudFlare/userAgents_per_cc_asn_0808.txt'
    df_cloudflare = pd.read_csv(file_path_cloudflare, sep = '\t')
    # transform the cumulative back to fraction by subtracting by the previous element
    df_cloudflare['fraction'] = df_cloudflare.groupby(['#cc'])['cumulative'].diff().fillna(
        df_cloudflare['cumulative'])
else:
    # file_path_cloudflare = '../data_CloudFlare/asn_human_bytes_2023-10-19_cf.txt'
    # df_cloudflare = pd.read_csv(file_path_cloudflare, sep=',')

    file_path_cloudflare = '../data_CloudFlare/traffic_per_cc_asn_0808.txt'
    df_cloudflare = pd.read_csv(file_path_cloudflare, sep = '\t')
    # transform the cumulative back to fraction by subtracting by the previous element
    df_cloudflare['fraction'] = df_cloudflare.groupby(['#cc'])['cumulative'].diff().fillna(
        df_cloudflare['cumulative'])

print(df_cloudflare)


def calculate_statistics(df_cloudflare, df_apnic):
    df_apnic['asn'] = df_apnic['asn'].apply(lambda x: x.split('AS')[1])
    df_apnic['asn'] = df_apnic['asn'].astype(int)

    print(df_cloudflare['#cc'].dtype, df_cloudflare['asn'].dtype)
    print(df_apnic['#cc'].dtype, df_apnic['asn'].dtype)
    print(df_apnic.head())
    # Merge the two dataframes on 'cc' and 'asn'
    merged_df = pd.merge(df_cloudflare, df_apnic, on=['#cc', 'asn'], suffixes=('_cloudflare', '_apnic'), how='inner')
    # for missing ASName, fill them with the ASN
    merged_df['AS Name'].fillna(merged_df['asn'], inplace=True)
    merged_df.fillna(0, inplace=True)
    # Create the AS Name -> ASN mapping
    as_name_to_asns = merged_df.groupby('AS Name')['asn'].apply(lambda x: list(x)).to_dict()
    # Aggregate different ASes with the same name by summing their users
    merged_df = merged_df.groupby(['#cc','AS Name'], as_index=False).sum()
    merged_df.drop(columns = ['rank', 'asn', 'Rank'], inplace=True)
    # Save the mapping to a pickle file
    import pickle
    with open('../data_org/as_name_to_asns_mapping.pkl', 'wb') as f:
        pickle.dump(as_name_to_asns, f)

    for cc,dg in merged_df.groupby('#cc'):
        apnic_ranked = dg['Users (est.)'].rank(method='dense', ascending = False).astype(int)
        dg['rank'] = apnic_ranked
        merged_df.loc[dg.index, 'rank_apnic'] = apnic_ranked
        cloudflare_ranked = dg['fraction'].rank(method='dense', ascending = False).astype(int)
        dg['rank'] = cloudflare_ranked
        merged_df.loc[dg.index, 'rank_cloudflare'] = cloudflare_ranked
        merged_df_res = merged_df.loc[dg.index]
        if cc == 'AL':
            print('?')
            print(merged_df.loc[dg.index])
    ##### FIGURE 3
    # Perform the outer join
    merged_df_outjoin = pd.merge(df_cloudflare, df_apnic, on=['#cc', 'asn'], suffixes=('_cloudflare', '_apnic'), how='outer')

    # Count the number of rows with NaN values in columns from df1 (unmatched in df_cloudflare)
    unmatched_in_df_cloudflare = merged_df_outjoin.filter(like='_cloudflare').isna().all(axis=1).sum()

    # Count the number of rows with NaN values in columns from df_apnic (unmatched in df_apnic)
    unmatched_in_df_apnic = merged_df_outjoin.filter(like='_apnic').isna().all(axis=1).sum()

    # Count the number of rows with non-Nan values
    unmatched_in_df3 = merged_df_outjoin.shape[0] - (unmatched_in_df_cloudflare + unmatched_in_df_apnic)

    # Calculate fractions
    unmatched_in_df_cloudflare_fraction = unmatched_in_df_cloudflare / merged_df_outjoin.shape[0]
    unmatched_in_df_apnic_fraction = unmatched_in_df_apnic / merged_df_outjoin.shape[0]
    matching_in_both_df_cloudflare_df_apnic_fraction = unmatched_in_df3 /  merged_df_outjoin.shape[0]
    # Display results
    print('Only in APNIC', 'Only APNIC fraction', 'Only in CDN', 'Only in CDN fraction', 'Both in CDN and APNIC', 'Both in CDN and APNIC fraction')
    print(unmatched_in_df_cloudflare, unmatched_in_df_cloudflare_fraction, unmatched_in_df_apnic, unmatched_in_df_apnic_fraction, unmatched_in_df3, matching_in_both_df_cloudflare_df_apnic_fraction)
    # For rows where both df_cloudflare and df_apnic have matching entries, use the 'User Estimated' from df_cloudflare
    weighted_overlap_sum = merged_df_outjoin.loc[
        merged_df_outjoin.filter(like='_cloudflare').notna().all(axis=1) & merged_df_outjoin.filter(like='_apnic').notna().all(
            axis=1),
        'Users (est.)'
    ].sum()

    # For rows present only in df_cloudflare, use the 'User Estimated' from df_cloudflare
    weighted_df_cloudflare_only_sum = merged_df_outjoin.loc[
        merged_df_outjoin.filter(like='_cloudflare').notna().all(axis=1) & merged_df_outjoin.filter(like='_apnic').isna().all(
            axis=1),
        'Users (est.)'
    ].sum()
    print(weighted_df_cloudflare_only_sum)
    print(weighted_overlap_sum)
    # Group by 'cc' and calculate Kendall Tau for each group
    correlations_kendall = {}
    correlations_pearson = {}
    coefficients = {}
    print(merged_df.columns)
    merged_df.to_csv('merged_df_traffic.csv')
    # List of country codes
    country_codes = ['RU', 'NO']
                     # 'TH', 'KH' ,'ZA', 'AZ', 'MM', 'SG', 'HK']
    full_names = {'RU': 'Russia', 'MD': 'Moldova', 'IR': 'Iran', 'TH': 'Thailand', 'ZA': 'South Africa', 'AZ': 'Azerbaijan', 'MM': 'Myanmar',
                  'SG': 'Singapore', 'HK': 'Hong Kong', 'KH': 'Cambodia', 'NO' : 'Norway', 'AL': 'Albania', 'QA': 'Qatar', 'IN' : 'India'}
    colors = {'RU': 'blue', 'MD': 'red', 'IR': 'green', 'TH': 'purple', 'ZA': 'orange', 'AZ': 'brown', 'MM': 'pink', 'SG': 'purple', 'AL': 'cyan'
              ,'KH': 'black', 'NO': 'red', 'QA': 'yellow', 'HK': 'grey', 'IN' : 'green'}
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    # Flatten the array of axes for easy iterating
    axs = axs.flatten()

    # Iterate over each country code and its corresponding axis
    for idx, cc in enumerate(country_codes):

        ax = axs[idx]

        # Filter the data for the current country code
        group = merged_df[merged_df['#cc'] == cc]
        # Calculate the global min and max for x and y across all subplots
        x_min = group['% of country'].min()
        x_max = group['% of country'].max()
        y_min = group['fraction'].min()
        y_max = group['fraction'].max()
        # Create scatter plot
        ax.scatter(group['% of country'], group['fraction'], s=10, c=colors[cc], alpha=1)

        # Add a linear regression with a dashed line with an intercept
        model = sm.OLS(group['fraction'], sm.add_constant(group['% of country']))  # Notice we use just 'x' without adding a constant
        model = sm.OLS(group['fraction'], group['% of country'])
        results = model.fit()
        slope = results.params[0]
        # slope = results.params[1]
        # intercept = results.params[0]
        # x = group['% of country']
        # # y = slope * x + intercept
        # y = slope*x
        # Set x-values for the line to span the entire x-axis range
        line_x = np.linspace(min(x_min,y_min), max(x_max, y_max), 100)  # Generate 100 points across the x-axis range
        # line_y = slope * line_x + intercept
        line_y = slope* line_x
        ax.plot(line_x, line_y, color='black', alpha=0.5, linestyle='--', label='Linear Regression')
        # Set the same x and y limits for all subplots (to keep them square)
        min_limit = min(x_min, y_min)
        max_limit = max(x_max, y_max)
        ax.set_xlim(min_limit, max_limit)
        ax.set_ylim(min_limit, max_limit)
        print(min_limit, max_limit)
        # ax.plot(x, y, color='black', alpha = 0.5, linestyle='--', label='Linear Regression')
        # Add the linear fit to the plot next to the line
        x_text = group['% of country'].max() - 1
        # y_text = slope * x_text + intercept + 6
        y_text = slope* x_text + 2
        ax.text(x_text, y_text, f'$\\rho$ = {slope:.2f}', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        # Add subplot title
        # ax.set_title(f'ASes in {full_names[cc]}', fontsize=20)
        # Add labels for top 10 ASes
        group1 = group.sort_values(by='rank_cloudflare', ascending=True).head(5)['AS Name'].tolist()
        group2 = group.sort_values(by='rank_apnic', ascending=True).head(5)['AS Name'].tolist()
        texts = []
        # for i, label in enumerate(group['AS Name']):
        #     if label in group1 or label in group2:
        #         texts.append(ax.text(group['% of country'].iloc[i], group['fraction'].iloc[i], label.split(' ')[0], fontsize=10))
        ax.yaxis.set_tick_params(labelsize=14)
        ax.xaxis.set_tick_params(labelsize=14)
        ### change the ticks to have the percentage
        # ax.set_xticks(list(ax.get_xticks()) + [0])
        # ax.set_yticks(list(ax.get_xticks()) + [0])
        # ax.set_xticklabels([f'{int(x)}%' for x in ax.get_xticks()])
        # ax.set_yticklabels([f'{int(y)}%' for y in ax.get_yticks()])
        # Set the tick values to be the same for both x and y axes
        # Define a range of ticks based on the minimum and maximum of the axis limits
        ticks = np.arange(np.floor(min_limit / 5) * 5, np.ceil(max_limit / 5) * 5 + 1, 10)

        # Set the same tick values for both x and y axes
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # Set the tick labels to percentages for both axes
        ax.set_xticklabels([f'{int(x)}%' for x in ticks])
        ax.set_yticklabels([f'{int(y)}%' for y in ticks])
        # Adjust text to prevent overlap
        # adjust_text(texts, arrowprops=dict(arrowstyle='-', color='green'), ax=ax)
        ax.xaxis.set_label_text(f'{full_names[cc]} APNIC User Estimates (%)', fontsize=14)
        ax.yaxis.set_label_text(f'{full_names[cc]} CDN User-Agents (%)', fontsize=14)
        # Set the aspect ratio to be equal to ensure square plots
        ax.set_aspect('equal', 'box')
    # Set common labels
    # fig.text(0.5, -0.02, 'User Estimates According to APNIC', ha='center', va='center', fontsize=20)
    # fig.text(-0.04, 0.5, 'Traffic Volume According to CDN', ha='center', va='center', rotation='vertical', fontsize=20)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'comparison_plot_{kind}.pdf', bbox_inches='tight')

    for cc, group in merged_df.groupby('#cc'):
        if group.shape[0] > 2:
            # group = group.sort_values(by='rank_cloudflare', ascending=True)
            # group = group.head(10)
            print(group.shape)
            # if cc == 'IR':
            #     print('?')
            #     plt.title(f'ASes in Iran')
            #     plt.scatter(group['% of country'], group['fraction'], s=10, c='blue', alpha=0.5)  # s is the marker size
            #     plt.xlabel('% of country according to APNIC')
            #     plt.ylabel('% of country according to CDN')
            #     ### add labels of the top 10 ASes according to APNIC and Cloudflare
            #     group1 = group.sort_values(by='rank_cloudflare', ascending=True)
            #     group1 = group1.head(5)['AS Name'].tolist()
            #     group2 = group.sort_values(by='rank_apnic', ascending=True)
            #     group2 = group2.head(5)['AS Name'].tolist()
            #     texts = []
            #     for i, label in enumerate(group['AS Name']):
            #         if label in group1 or label in group2:
            #             texts.append(plt.text(group['% of country'].iloc[i], group['fraction'].iloc[i], label, fontsize=7))
            #     texts = adjust_text(texts, arrowprops=dict(arrowstyle='-', color='green'))
            #     plt.savefig('IR.png', bbox_inches='tight')

            # print(group['fraction'].sum())
            # tau, p_value = kendalltau(group['rank_cloudflare'], group['rank_apnic'])
            # correlations_kendall[cc] = tau
            # correlation = group['% of country'].corr(group['fraction'], method='pearson')
            # correlations_pearson[cc] = correlation
            # Sort the DataFrame by 'rank_cloudflare'
            group_sorted = group.sort_values(by='rank_cloudflare')

            # # Calculate the cumulative sum of 'User (est.)'
            # group_sorted['cumulative_sum'] = group_sorted['Users (est.)'].cumsum()
            #
            # # Calculate the total sum of 'User (est.)'
            # total_sum = group_sorted['Users (est.)'].sum()
            #
            # # Select rows where the cumulative sum is less than or equal to 95% of the total sum
            # group_restricted = group_sorted[group_sorted['cumulative_sum'] <= 0.95 * total_sum]
            #
            # # Drop the 'cumulative_sum' column if no longer needed
            # group_restricted = group_restricted.drop(columns=['cumulative_sum'])
            group_restricted = group_sorted[group_sorted['% of country'] > 0.5]

            # Compute Kendall's tau correlation
            tau, p_value = kendalltau(group_restricted['rank_cloudflare'], group_restricted['rank_apnic'])
            correlations_kendall[cc] = tau

            # Compute Pearson correlation
            correlation = group['% of country'].corr(group['fraction'], method='pearson')
            correlations_pearson[cc] = correlation


            # Compute linear regression

            model = sm.OLS(group['fraction'], group['% of country'])  # Notice we use just 'x' without adding a constant
            results = model.fit()

            print(results.summary())
            # slope, intercept, r_value, p_value, std_err = linregress(group['% of country'],
            #                                                                      group['fraction'])
            slope = results.params[0]
            coefficients[cc] = slope
    # print(f"Linear regression slope: {slope}")
            print(f"Linear regression slope: {slope}")
            print(f"Correlation: {correlation}")
            ### plot the linear regression
            # if cc =='IN':
            # if correlation < 0.8:
            #     plt.figure(figsize=(8,8))
            #     plt.scatter(group['% of country'], group['fraction'], s=10, c='blue', alpha=0.5)
            #     plt.xlabel('% of country according to APNIC')
            #     plt.ylabel('% of country according to CDN')
            #     limit_range = [min(group['% of country'].min(), group['fraction'].min()), max(group['% of country'].max(), group['fraction'].max())]
            #     plt.xlim(limit_range)
            #     plt.ylim(limit_range)
            #     country_name = pycountry.countries.get(alpha_2=cc).name.split(',')[0]
            #     plt.title(f'ASes in {country_name}')
            #     x = group['% of country']
            #     # y = slope * x
            #     # plt.plot(x, y, color='red', label='Linear Regression')
            #     plt.legend()
            #     plt.show()


            ## count
    ### plot count
    import seaborn as sns
    df_data = pd.read_csv('../data_CloudFlare/all.csv')
    continent_mapping = df_data[['alpha-2', 'region']].drop_duplicates().set_index('alpha-2')['region'].to_dict()
    df['Continent'] = df['#cc'].map(continent_mapping)

    # Count the occurrences of each country code
    country_code_counts = df['#cc'].value_counts()

    # Filter to top N countries for better readability
    top_n = 50
    top_country_codes = country_code_counts.head(top_n).index
    top_df = df[df['#cc'].isin(top_country_codes)]

    # Group by country code and continent for plotting
    grouped = top_df.groupby(['#cc', 'Continent']).size().reset_index(name='counts')

    # Sort the values in descending order
    grouped = grouped.sort_values(by='counts', ascending=False)

    # Create a bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='counts', y='#cc', hue='Continent', data=grouped, dodge=False)

    # Adding labels and title
    plt.xlabel('Number of Records')
    plt.ylabel('Country Code')
    plt.title(f'Top {top_n} Countries by Number of Records')

    # Show the plot
    plt.savefig('../top_50_countries.pdf', bbox_inches='tight')
    # Assuming 'df' is your DataFrame
    # Count the occurrences of each country code
    # country_code_counts = df['#cc'].value_counts()
    #
    # # Sort the counts (optional, remove if not needed)
    # country_code_counts = country_code_counts.sort_values(ascending=False)
    #
    # # Create a bar plot
    # plt.figure(figsize=(12, 6))  # Adjust the size as needed
    # # Choose top N countries
    # top_n = 50
    # top_country_codes = country_code_counts.head(top_n)
    #
    # # Create a bar plot
    # sns.barplot(x=top_country_codes.index, y=top_country_codes.values)
    #
    # # Adding labels and title
    # plt.xlabel('Country Code')
    # plt.ylabel('Number of ASes')
    # plt.title(f'Top {top_n} Countries by Number of Records')
    #
    # # Show the plot
    # plt.show()

    return correlations_kendall,correlations_pearson, coefficients

# Example usage
correlations_kendall,correlations_pearson, coefficients  = calculate_statistics(df_cloudflare, df)
labels = correlations_pearson.keys()
consistent_countries = 0
rank_agreement = 0
major_ases_agreement = 0
perfect_alignment_agreement = 0
for cc in correlations_kendall.keys():
    print(f"Kendall Tau for {cc}: {correlations_kendall[cc]}")
    print(f"Pearson for {cc}: {correlations_pearson[cc]}")
    print(f"Linear coefficient for {cc}: {coefficients[cc]}")
    if correlations_pearson[cc] > 0.8 and coefficients[cc] > 0.5:
        major_ases_agreement += 1
    if correlations_kendall[cc] > 0.8:
        rank_agreement += 1
    if  correlations_pearson[cc] > 0.8 and coefficients[cc] > 0.8 and correlations_kendall[cc] > 0.8:
        perfect_alignment_agreement += 1
    if correlations_pearson[cc] > 0.8 or correlations_kendall[cc] > 0.8 and coefficients[cc] > 0.9:
        print(cc)
        consistent_countries+= 1
print('Ratio of Perfect Alignment')
print(perfect_alignment_agreement/ len(correlations_pearson))
print('Ratio of Rank Agreement')
print(rank_agreement/len(correlations_pearson))
print('Ratio of Major ASes Agreement')
print(major_ases_agreement/len(correlations_pearson))
country_categories = {}


for cc in correlations_kendall.keys():
    if correlations_pearson[cc] > 0.8 and coefficients[cc] > 0.8 and correlations_kendall[cc] > 0.8:
        category = 'Perfect Alignment'
    elif correlations_kendall[cc] > 0.8:
        category = 'Rank Agreement'
    elif correlations_pearson[cc] > 0.8 and coefficients[cc] > 0.5 and correlations_kendall[cc] > 0.7:
        category = 'Major ASes Agreement (Strong)'
    elif correlations_pearson[cc] > 0.8 and coefficients[cc] > 0.5 and correlations_kendall[cc] > 0.6:
        category = 'Major ASes Agreement (Moderate)'
    elif correlations_pearson[cc] > 0.8 and coefficients[cc] > 0.5 and correlations_kendall[cc] > 0.4:
        category = 'Major ASes Agreement (Weak)'
    elif correlations_pearson[cc] > 0.8 and coefficients[cc] > 0.5 and correlations_kendall[cc] < 0.4:
        category = 'Major ASes Agreement (None)'
    else:
        category = 'No Significant Agreement'

    country_categories[cc] = category

# Add all countries as 'Not Enough Information' if not already categorized
for country in pycountry.countries:
    if country.alpha_2 not in country_categories:
        country_categories[country.alpha_2] = 'Not Enough Information'
# Convert country categories to a list of dicts with ISO alpha-3 codes and categories
data = []
for cc, category in country_categories.items():
    if len(str(cc)) != 2:
        continue
    try:
        print(cc)
        country = pycountry.countries.get(alpha_2=cc)
        data.append({
            'iso_alpha_3': country.alpha_3,  # Plotly expects ISO-3 country codes
            'category': category
        })
    except KeyError:
        # Handle countries not found in pycountry
        continue


df_country_category = pd.DataFrame(data)

# Color mapping for categories, adding grey for 'Not Enough Information'
category_colors = {
    'Perfect Alignment': 'green',
    'Rank Agreement': 'royalblue',
    'Major ASes Agreement (Strong)': 'indigo',
    'Major ASes Agreement (Moderate)' : 'darkviolet',
    'Major ASes Agreement (Weak)': 'violet',
    'Major ASes Agreement (None)': 'plum',
    'No Significant Agreement': 'red',
    'Not Enough Information': 'grey'
}
import plotly.express as px

# Plot using Plotly Express
fig = px.choropleth(df_country_category,
                    locations="iso_alpha_3",
                    color="category",
                    color_discrete_map=category_colors,
                    # title="Country Classification Based on Agreement Categories",
                    category_orders={"category": ["Perfect Alignment", "Rank Agreement", "Major ASes Agreement", "No Significant Agreement", "Not Enough Information"]},
                    )

# Show the plot
# fig.show()

# Step 1: Adjust the layout to remove extra elements
fig.update_layout(
    title=None,           # Remove the title
    showlegend=False,     # Hide the legend
    margin=dict(l=0, r=0, t=0, b=0),  # Remove margins around the plot
    coloraxis_showscale=False  # Hide the color scale if present
)

# Step 2: Save the map using kaleido
fig.write_image(f"plotly_map_only_{kind}.pdf", format="pdf")

### plot the results in a scatter plot with the x-axis being the pearson and the y-axis being the kendall tau
#
# Create a scatter plot
# Convert correlations_pearson and correlations_kendall into lists for plotting
# remove Gilbratar
correlations_pearson.pop('GI')
correlations_kendall.pop('GI')
# correlations_kendall.pop('CM')
# correlations_pearson.pop('CM')
plt.figure(figsize=(12, 8))

pearson_values = list(correlations_pearson.values())
kendall_values = list(correlations_kendall.values())

# Create a JointGrid for scatter plot with marginal distributions
g = sns.JointGrid(x=pearson_values, y=kendall_values, space=0, height=10)

# Scatter plot in the center
g.plot_joint(plt.scatter, color="black", alpha=0.5, s=100, marker='^')

# Add marginal histograms
g.plot_marginals(sns.histplot, bins=30, kde=False, color="snow", alpha=0.5)

# Add lines at 0.8 for both axes
g.ax_joint.axhline(0.8, color='red', linestyle='--')
g.ax_joint.axvline(0.8, color='red', linestyle='--')

# Define custom purple shades
dark_purple = mcolors.to_rgba('indigo', alpha=0.5)  # Most purple
medium_purple = mcolors.to_rgba('darkviolet', alpha=0.5)  # Medium purple
light_purple = mcolors.to_rgba('violet', alpha=0.5)  # Light purple
very_light_purple = mcolors.to_rgba('plum', alpha=0.5)  # Very light purple

# Define custom blue shades
dark_blue = mcolors.to_rgba('royalblue', alpha=0.5)  # Most blue
medium_blue = mcolors.to_rgba('cornflowerblue', alpha=0.5)  # Medium blue
light_blue = mcolors.to_rgba('lightsteelblue', alpha=0.5)  # Light blue
very_light_blue = mcolors.to_rgba('aliceblue', alpha=0.5)  # Very light blue

# Calculate the relative xmin for x=0.8 when xlim is (-0.1, 1)
relative_xmin = (0.8 - (-0.1)) / (1 - (-0.1))
# Apply the gradient colors to different spans for purple
g.ax_joint.axhspan(0.7, 0.8, xmin=relative_xmin, xmax=1, color=dark_purple)    # Dark purple region
g.ax_joint.axhspan(0.6, 0.7, xmin=relative_xmin, xmax=1, color=medium_purple)  # Medium purple region
g.ax_joint.axhspan(0.4, 0.6, xmin=relative_xmin, xmax=1, color=light_purple)     # Light purple region
g.ax_joint.axhspan(0, 0.4, xmin=relative_xmin, xmax=1, color=very_light_purple)   # Very light purple region

# Apply the gradient colors to different spans for blue
g.ax_joint.axvspan(0.7, 0.8, ymin=0.8, ymax=1, color=dark_blue)    # Dark blue region
g.ax_joint.axvspan(0.6, 0.7, ymin=0.8, ymax=1, color=medium_blue)  # Medium blue region
g.ax_joint.axvspan(0.4, 0.6, ymin=0.8, ymax=1, color=light_blue)   # Light blue region
g.ax_joint.axvspan(-0.1, 0.4, ymin = 0.8, ymax = 1, color=very_light_blue)  # Very light blue region
# Green region remains unchanged
g.ax_joint.axvspan(0.8, 1, ymin=0.8, ymax=1, color='green', alpha=0.5)  # Green region
# Add text labels to points where Pearson and Kendall are both < 0.8
texts = []
for label in labels:
    if correlations_pearson[label] < 0.8 and correlations_kendall[label] < 0.8:
        country_name = pycountry.countries.get(alpha_2=label).name.split(',')[0]
        if country_name.startswith('United K'):
            country_name = 'UK'
        texts.append(g.ax_joint.text(correlations_pearson[label], correlations_kendall[label], country_name, fontsize=17))

# Adjust text to avoid overlapping
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))

# Set labels
g.set_axis_labels('Pearson Correlation', 'Kendall Tau Correlation', fontsize=18)

# Set x and y limits
g.ax_joint.set_xlim(-0.1, 1)
g.ax_joint.set_ylim(0, 1)

# Set tick sizes
g.ax_joint.tick_params(axis='both', labelsize=15)

# Add grid
g.ax_joint.grid(True)

# Save the plot
plt.tight_layout()
print(kind)
plt.savefig(f'kendall_tau_coefficient_pearson_correlation_traffic_with_marginals_{kind}.pdf', bbox_inches='tight')

# plt.show()
# Create the figure
# plt.figure(figsize=(10, 6))  # Increase figure size

# Create a hexbin plot
# plt.hexbin(correlations_pearson.values(), correlations_kendall.values(), gridsize=30, cmap='Blues', alpha=0.5)
#
# # Add a color bar to show the count in the bins
# cb = plt.colorbar()
# cb.set_label('Count in bin')
#
# # Add grid, title, and labels
# plt.grid(True)
# plt.title('Hexbin Comparison of Pearson and Kendall Tau Correlations per Country Code', fontsize=16)
# plt.xlabel('Pearson Correlation', fontsize=14)
# plt.ylabel('Kendall Tau Correlation', fontsize=14)
# plt.xticks(fontsize=11)
# plt.yticks(fontsize=11)
#
# # Improve layout and show/save plot
# plt.tight_layout()
# plt.savefig('kendall_tau_pearson_hexbin.pdf', bbox_inches='tight')
# plt.show()
# Set figure size
