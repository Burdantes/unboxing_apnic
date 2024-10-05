import pickle
import pandas as pd

df = pickle.load(open('/Users/loqmansalamatian/Downloads/drive-download-20240108T200553Z-001/Nov.p', 'rb'))
### transform the index to columns
df.reset_index(inplace=True)
# check the dtype of each column
print(df.dtypes)
df['in_bytes'] = df['in_bytes'].astype('float64')
df['out_bytes'] = df['out_bytes'].astype('float64')


### generate distribution per ['nflows', 'in_bytes', 'out_bytes']
# df = df.groupby(['nflows', 'in_bytes', 'out_bytes']).size().reset_index(name='counts')
# ### generate the total number of flows
# df['total_flows'] = df['nflows'] * df['counts']
# ### generate the total number of bytes
# df['total_bytes'] = df['in_bytes'] * df['counts']
# ### generate the total number of bytes
# df['total_bytes'] = df['in_bytes'] * df['counts']
# ### generate the total number of bytes
# df = df.groupby('as_owner').sum().reset_index()
df = df.groupby('service')[['nflows', 'out_bytes', 'in_bytes']].sum().reset_index()

### transform all the nans to 0
df.fillna(0, inplace=True)
df['nflows'] = df['nflows']/ df['nflows'].sum()
df['in_bytes'] = df['in_bytes']/ df['in_bytes'].sum()
df['out_bytes'] = df['out_bytes']/ df['out_bytes'].sum()

df.to_csv('df_November_service_normalized.csv')
print(df.sort_values(by='nflows', ascending=False).head(10))
print(df.columns)