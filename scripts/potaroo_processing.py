import pandas as pd
import os
import glob

# Set the directory containing your CSV files
directory = '../data_potaroo'

# Create an empty list to store the data from each file
data = []

# Load each CSV file and append the data to the list
for file in glob.glob(os.path.join(directory, "*.csv")):
    # Extract the date from the filename (assuming format YYYY-MM-DD-as.csv)
    date = os.path.basename(file).split('-')[0]  # Get the year part

    # Read the CSV file
    df = pd.read_csv(file)

    # Add a new column for the year (extracted from the filename)
    df['Year'] = pd.to_datetime(date).year
    df['Date'] = pd.to_datetime(date)
    # Remove any rows where 'Allocated ASNs' is not a number (in case of repeated headers)
    df = df[pd.to_numeric(df['Allocated ASNs'], errors='coerce').notna()]

    # Append the dataframe to the list
    data.append(df)

# Combine all the CSV data into a single dataframe
combined_df = pd.concat(data, ignore_index=True)

# Convert 'Allocated ASNs' to numeric (ensure all values are numbers)
combined_df['Advertised ASNs'] = pd.to_numeric(combined_df['Advertised ASNs'], errors='coerce')
combined_df['Allocated ASNs'] = pd.to_numeric(combined_df['Allocated ASNs'], errors='coerce')
combined_df['Users'] = pd.to_numeric(combined_df['Users'], errors='coerce')

# Filter the data to include only the years from 2019 to 2023
# combined_df = combined_df[combined_df['Year'].between(2014, 2023)]

# Filter data for 2019 and 2024
df_filtered = combined_df[combined_df['Year'].isin([2019, 2024])]

# Group by 'Year' and 'Region' to get the total allocated and announced ASNs for each year and region
asn_totals = df_filtered.groupby(['Year', 'Region'])[['Allocated ASNs', 'Advertised ASNs', 'Users']].sum().reset_index()

# Pivot the data to have separate columns for 2019 and 2024 for both Allocated and Advertised ASNs
asn_totals_pivot = asn_totals.pivot(index='Region', columns='Year', values=['Allocated ASNs', 'Advertised ASNs', 'Users'])

# Calculate the percentage increase between 2019 and 2024 for both Allocated and Advertised ASNs
asn_totals_pivot['Allocated ASN Increase (%)'] = ((asn_totals_pivot['Allocated ASNs'][2024] - asn_totals_pivot['Allocated ASNs'][2019]) / asn_totals_pivot['Allocated ASNs'][2019]) * 100
asn_totals_pivot['Advertised ASN Increase (%)'] = ((asn_totals_pivot['Advertised ASNs'][2024] - asn_totals_pivot['Advertised ASNs'][2019]) / asn_totals_pivot['Advertised ASNs'][2019]) * 100
asn_totals_pivot['Users Increase (%)'] = ((asn_totals_pivot['Users'][2019] - asn_totals_pivot['Users'][2024]) / asn_totals_pivot['Users'][2024]) * 100
# Reset the index for easier viewing
asn_totals_pivot.reset_index(inplace=True)

# Display the percentage increase
print(asn_totals_pivot[['Region', 'Allocated ASN Increase (%)', 'Advertised ASN Increase (%)']])

# Select the relevant columns for the LaTeX table
latex_table_df = asn_totals_pivot[['Region', 'Allocated ASN Increase (%)', 'Advertised ASN Increase (%)', 'Users Increase (%)']]

# Sort by the Advertised ASN Increase in descending order
latex_table_df = latex_table_df.sort_values('Advertised ASN Increase (%)', ascending=False)

# Convert the DataFrame to a LaTeX table
latex_table = latex_table_df.to_latex(index=False, float_format="%.2f", caption="Percentage Increase in Allocated and Advertised ASNs (2019-2024)", label="tab:asn_increase", column_format="|l|c|c|", bold_rows=True)

# Print the LaTeX table
print(latex_table)

# Optionally, save the LaTeX table to a file
with open('asn_increase_table.tex', 'w') as f:
    f.write(latex_table)

# Group the data by Year and Region (continent) and calculate the average number of ASNs
# average_asns_per_continent = combined_df.groupby(['Year', 'CC Code'])['Advertised ASNs'].mean().reset_index()

# Display the average number of ASNs per continent each year
# print(average_asns_per_continent)

# Optionally, plot the average evolution of ASNs per continent
import matplotlib.pyplot as plt
import seaborn as sns

# Find all latin american countries
latin_america = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PY', 'SV', 'UY', 'VE']
# average_asns_per_continent = average_asns_per_continent[average_asns_per_continent['CC Code'].isin(latin_america)]
combined_df = combined_df[combined_df['CC Code'].isin(latin_america)]
plt.figure(figsize=(10, 6))
sns.lineplot(data=combined_df, x='Date', y='Advertised ASNs', hue='CC Code')
plt.title('Average Evolution of ASNs per Country in Each Continent (2019-2023)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of ASNs', fontsize=12)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()