
## Prepare the virtual environment 


```python
# !which python # using python2 from virtualenv
# !pip install pandas
# !pip install datetime
# !pip install scipy
# !pip install NumPy
# !pip install Matplotlib
# !pip install scikit-learn
```

## Import and loading data

Global import declarations and loading.


```python
import json
import time
import copy
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsRegressor


# importing the csv data about the store sales, 906 stores with 11936 timestamps.
raw_csv = pd.read_csv('sales_granular.csv', index_col=0)

# importing the surroundings information about some 546 stores, each with 89 possible types of surroundings.
raw = json.load(open('Surroundings.json'))

# manual inspection of JSON file by writing a single items at a time to disk.
# commented out due to possible permission issues on different computers.
# with open('extract.json', 'w') as outfile:
#     json.dump(raw[0], outfile)

```

## Helper functions

Helper functions which move complexity outside the core phases of data science: data understanding (previous step), exploration, transformation (cleaning), modeling and assessment.


```python
def group_by_month(timestamp_str):
    """
    Groups the sparse timestamps of each store inside a monthly group. 
    This is relevant since the number of timestamps in not consistent across periods in a month.
    Furthermore more sophisticated models such as AMRA / ARIMA modeling require lag times of weeks or months.
    
    :param: timestam_str -> a string representing the hourly timestamp.
    :return: month_key -> the year-month timestamps which can be used for time-series analysis.
    """
    space_idx = timestamp_str.find(' ')
    parse_date_string = timestamp_str[:space_idx]
    
    date_datetime = datetime.strptime(parse_date_string, '%m/%d/%y')

    if len(str(date_datetime.month)) == 1:
        mm = '0' + str(date_datetime.month)
    else:
        mm = str(date_datetime.month)
        
    yyyy = str(date_datetime.year)
    
    month_key = '{0}-{1}'.format(yyyy, mm)
    return month_key
```


```python
def get_series_stats(series, store_id):
    """
    Generates a dictionary of relevant statistics for a given time-series -like dataset.
    
    :param: series -> the monthly sales of a store.
    :param: store_id -> the unique identifier of a store.
    :return: dict -> a dictionary containing statistics about each store, identifiable by the store_id key.
    """
    sales_points_sum = 0
    sales_points_valid = []

    for _, val in enumerate(series):
        if math.isnan(val):
            continue
        else:
            sales_points_sum += val
            sales_points_valid.append(val)

    series_mean = float(sales_points_sum) / max(len(sales_points_valid), 1)
    series_stdev = max(round(math.sqrt(float(reduce(lambda x, y: x + y, map(lambda x: (x - series_mean) ** 2, sales_points_valid))) / len(sales_points_valid)), 2), 1)
    
    return dict({
        'mean': round(series_mean, 2),
        'months_of_data_count': len(sales_points_valid),
        'store_id': store_id,
        'total_products_sold': sales_points_sum,
        'stdev': series_stdev
        })
    
```


```python
def normalize_df(df):
    """
    Creates a normalized version of the dataframe used in training and testing.
    
    :param: df -> a pandas dataframe which is not normalized.
    :return: norml -> a pandas dataframe with normalized values between (0,1).
    """
    normalized_df = copy.deepcopy(df)
    norm1 = normalize(normalized_df, axis=0, norm='max')
    return(norm1)
```


```python
# borrowed shamelessly from https://plot.ly/python/polygon-area/
def PolygonSort(corners):
    n = len(corners)
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    cornersWithAngles = []
    for x, y in corners:
        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        cornersWithAngles.append((x, y, an))
    cornersWithAngles.sort(key = lambda tup: tup[2])
    return map(lambda (x, y, an): (x, y), cornersWithAngles)

def PolygonArea(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

## example of area calculation below | as reference: https://www.mathsisfun.com/geometry/polygons-interactive.html
# plotting the points in this order: N, E, S, W
corners = [(2.2, 5.4), (5.4, 4.4), (4.7, 1.7), (1.4, 2.5)]
corners_sorted = PolygonSort(corners)
area = PolygonArea(corners_sorted)
```


```python
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

def plot_monthly_series(monthly_series):
    """
    Creates the scatter plot of a time series for easy of exploration.
    """
    plt.plot(monthly_series)
    plt.show()
```

## Data Transformation

The data is given on an hourly basis. After some assessment it made the most sense group the hourly sales data into monthly sales data, making the months much easier to compare.

The following section transforms the column names (the hourly timestamps) into chunks of monthly data, aggregating all the sales within a month period.

Initially the idea was to create a sorted list of year-month timestamps and use it at a store level to predict / forecast monthly sales.

In the end the most useful feature was the ability to plot the time series of a store and try to identify trends.


```python
# get the column names of the original csv file
atomic_timestamps = list(raw_csv)

# create a container dictionary
month_dict = {}

for idx, col_timestamp in enumerate(atomic_timestamps):
    
    # translate column string into year-month: yyyy-mm format    
    current_month = group_by_month(col_timestamp)
    
    # store new timestamp into monthly dictionary.
    if month_dict.has_key(current_month):
        month_dict[current_month].append(idx)
    else:
        month_dict[current_month] = []
        month_dict[current_month].append(idx)


sorted_key_list = sorted(month_dict.keys())

store_id_list = list(raw_csv.index)

# generate a dictionary of store_id with their appropriate series values (interpretable by month)
store_dict = {}

# Store the monthly sales per each store. 
for i in range(0, len(raw_csv.index)):
    monthly_series = []
    for _, key in enumerate(sorted_key_list):
        monthly_series.append(raw_csv.iloc[i][month_dict[key][0]:month_dict[key][-1]].sum(skipna=True))
        
    store_dict[store_id_list[i]] = monthly_series
```

## Create table of relevant statistics in a time-series manner

The next steps are identifying a target variable in order to produce a supervised model.

For simplicity the 'total_products_sold' of each store were selected. This is of course a simple metric and prone to bias (as we shall see shortly) however it makes the most sense since we are trying to analyze what features of the surrounding area can affect the sales of a store.


```python
# generate empty placeholder pandas dataframe
stats_df = pd.DataFrame(columns= ['store_id', 'total_products_sold', 'mean', 'stdev', 'months_of_data_count'])

# append row by row to the stats dataframe
for key in store_dict.keys():
    row_dict = get_series_stats(store_dict[key], key) 
    row_df = pd.DataFrame.from_records(row_dict, index=[0])[['store_id', 'total_products_sold', 'mean', 'stdev', 'months_of_data_count']]
    stats_df = pd.concat([stats_df, row_df])

stats_df.reset_index(drop=True, inplace=True)

# create a view of the stats data frame for investigation. Sorting the table by number of months, total products sold and standard deviation.
result = stats_df.sort_values(['months_of_data_count', 'total_products_sold', 'stdev'], ascending=[0, 0, 1])

```

## Preparing the surroundings dataset for analysis

The goal of the project is to identify important attributes in the surroundings. At this point I have already investigated the structure of the JSON dataset by exporting different store properties, by use of a 'extract.json' file, as can be seen in the import section of this notebook.

It contained 89 different forms of surroundings, each an array of 0 or more objects. 


```python
# preparing the surroundings data for analysis
amenities_array = raw[0]['surroundings'].keys()

column_names_amenities = copy.deepcopy(amenities_array)
column_names_amenities = ['store_id'] + column_names_amenities

full_feature_amenities_df = pd.DataFrame(columns = column_names_amenities, index=[0])

# create a feature vector per store id which contains the number of a certain surrounding type.
for _, surroundings_obj in enumerate(raw):
    
    amenities_feature_dict = {}
    store_id = surroundings_obj['store_code']
    amenities_feature_dict['store_id'] = store_id
    
    for _, key in enumerate(amenities_array):
        amenities_feature_dict[key] = len(surroundings_obj['surroundings'][key]) 
        
    feature_amenities_row = pd.DataFrame(data = amenities_feature_dict, columns = column_names_amenities, index=[0])
    
    full_feature_amenities_df = pd.concat([full_feature_amenities_df, feature_amenities_row])

full_feature_amenities_df = full_feature_amenities_df[1:]
full_feature_amenities_df.reset_index(drop=True, inplace=True)
```

## Preparing fitting and validation data sets

Need to join the 2 pandas dataframes on their store_ids in order to append the target variable per feature vector.


```python
# select the store ids that are only in the surrounding dataset
ops_df = result.loc[result['store_id'].isin(full_feature_amenities_df['store_id'])]

# join on store_id key to append the total_products_sold
merged_df = pd.merge(full_feature_amenities_df, ops_df[['store_id', 'total_products_sold']], on='store_id').dropna()
merged_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>subway_station</th>
      <th>department_store</th>
      <th>embassy</th>
      <th>beauty_salon</th>
      <th>police</th>
      <th>courthouse</th>
      <th>cemetery</th>
      <th>pharmacy</th>
      <th>local_government_office</th>
      <th>...</th>
      <th>zoo</th>
      <th>train_station</th>
      <th>jewelry_store</th>
      <th>laundry</th>
      <th>insurance_agency</th>
      <th>plumber</th>
      <th>pet_store</th>
      <th>bakery</th>
      <th>travel_agency</th>
      <th>total_products_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10055</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>33780.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10077</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3900.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10079</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>270210.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10086</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>33810.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10111</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>18000.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10377</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>14</td>
      <td>70800.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10441</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>9</td>
      <td>172050.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10545</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>20280.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10548</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>102510.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10672</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>30390.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10814</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>42330.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10820</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9150.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10871</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>135330.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10883</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13200.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10928</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>520380.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>10962</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>15390.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10975</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>5</td>
      <td>2970.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10992</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>12</td>
      <td>450.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>11007</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>4</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>11013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>10380.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>11028</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>12</td>
      <td>539040.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11028</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>3</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>12</td>
      <td>539040.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>11233</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>360.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>11564</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>16</td>
      <td>273300.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>11570</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>18</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>20</td>
      <td>117810.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>11582</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>11790.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>11603</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>15</td>
      <td>537420.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>11607</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>18</td>
      <td>308910.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>11736</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25290.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>11954</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>524610.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>510</th>
      <td>36151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5580.0</td>
    </tr>
    <tr>
      <th>511</th>
      <td>36154</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2700.0</td>
    </tr>
    <tr>
      <th>512</th>
      <td>3622</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2430.0</td>
    </tr>
    <tr>
      <th>513</th>
      <td>3626</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1800.0</td>
    </tr>
    <tr>
      <th>514</th>
      <td>3655</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>515</th>
      <td>36570</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2940.0</td>
    </tr>
    <tr>
      <th>516</th>
      <td>36589</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8100.0</td>
    </tr>
    <tr>
      <th>517</th>
      <td>36934</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2520.0</td>
    </tr>
    <tr>
      <th>518</th>
      <td>3745</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>519</th>
      <td>39228</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1710.0</td>
    </tr>
    <tr>
      <th>520</th>
      <td>3933</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>521</th>
      <td>3958</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>20</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>522</th>
      <td>4013</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>420.0</td>
    </tr>
    <tr>
      <th>523</th>
      <td>4032</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>524</th>
      <td>40407</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3780.0</td>
    </tr>
    <tr>
      <th>525</th>
      <td>4068</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5610.0</td>
    </tr>
    <tr>
      <th>526</th>
      <td>4069</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>10500.0</td>
    </tr>
    <tr>
      <th>527</th>
      <td>425</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>810.0</td>
    </tr>
    <tr>
      <th>528</th>
      <td>4443</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>450.0</td>
    </tr>
    <tr>
      <th>529</th>
      <td>44781</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7800.0</td>
    </tr>
    <tr>
      <th>530</th>
      <td>4565</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>15510.0</td>
    </tr>
    <tr>
      <th>531</th>
      <td>45940</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>330.0</td>
    </tr>
    <tr>
      <th>532</th>
      <td>46147</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1830.0</td>
    </tr>
    <tr>
      <th>533</th>
      <td>46279</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14160.0</td>
    </tr>
    <tr>
      <th>534</th>
      <td>46361</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>510.0</td>
    </tr>
    <tr>
      <th>535</th>
      <td>46379</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>34740.0</td>
    </tr>
    <tr>
      <th>536</th>
      <td>46468</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>9</td>
      <td>7380.0</td>
    </tr>
    <tr>
      <th>537</th>
      <td>46587</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2130.0</td>
    </tr>
    <tr>
      <th>538</th>
      <td>46610</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>539</th>
      <td>47724</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13380.0</td>
    </tr>
  </tbody>
</table>
<p>540 rows × 91 columns</p>
</div>



## Start modeling with cross validation implementation

This is the modeling step. It fits a variety of models by tweaking the weight type and the number of neighbours to analyze.

The output shows a 10-fold cross-validation mean score for a range of k-nn parameters.


```python
# create range of possible neighbour values for k
neigh_list = list(range(5,50))

# picking only odd neighbour counts. 
# This way we avoid having equal number of neighbours. 
# It is good practice for classification problems with fixed number of labels, however I consider it good convetion for discret models as well.
neighbors = filter(lambda x: x % 2 != 0, neigh_list)

for _, weight_type in enumerate(['uniform', 'distance']):
    
    print('\n #### Starting with {} weight #### \n'.format(weight_type))
    
    for kn in neighbors:

        cross_validation_array = []

        for cv_iteration in range(0, 10):

            # sample random index numbers for splitting the dataset. Will go for 85% training and 15% testing.
            msk = np.random.rand(len(merged_df)) <= 0.6
            train_df = merged_df[msk]

            test_df = merged_df[~msk]

            training_fit_set_columns = [col for col in train_df.columns if col not in ['store_id', 'total_products_sold']]
            training_fit_df = normalize_df(train_df[training_fit_set_columns])
            training_target_df = train_df['total_products_sold']

            test_fit_set_columns = [col for col in test_df.columns if col not in ['store_id', 'total_products_sold']]
            test_fit_df = normalize_df(test_df[test_fit_set_columns])
            test_target_df = test_df['total_products_sold']

            # fit model 
            neigh = KNeighborsRegressor(n_neighbors=kn, weights=weight_type)
            neigh.fit(training_fit_df, training_target_df) 
            neigh.predict(test_fit_df)

            cross_validation_array.append(neigh.score(test_fit_df, test_target_df))

        avg_score = reduce(lambda x, y: x + y, cross_validation_array) / len(cross_validation_array)

        print('Average score for k equals {0} is {1}'.format(kn, avg_score))
```

    
     #### Starting with uniform weight #### 
    
    Average score for k equals 5 is -0.121150842511
    Average score for k equals 7 is 0.00262294538631
    Average score for k equals 9 is -0.022210426836
    Average score for k equals 11 is -0.0140506914239
    Average score for k equals 13 is -0.113167260377
    Average score for k equals 15 is -0.0984276413126
    Average score for k equals 17 is 0.00101560989903
    Average score for k equals 19 is 0.0616209030196
    Average score for k equals 21 is 0.0284349695046
    Average score for k equals 23 is 0.0412615293445
    Average score for k equals 25 is 0.0314242724486
    Average score for k equals 27 is 0.00974787666221
    Average score for k equals 29 is 0.0237376589043
    Average score for k equals 31 is -0.0113732609986
    Average score for k equals 33 is 0.0266037403044
    Average score for k equals 35 is -0.0319197809883
    Average score for k equals 37 is 0.0465279603695
    Average score for k equals 39 is 0.0448521065037
    Average score for k equals 41 is 0.0504223566418
    Average score for k equals 43 is 0.0638921828774
    Average score for k equals 45 is 0.0567485713528
    Average score for k equals 47 is 0.0523237364677
    Average score for k equals 49 is 0.0465609522838
    
     #### Starting with distance weight #### 
    
    Average score for k equals 5 is 0.0187915788622
    Average score for k equals 7 is -0.0673969587722
    Average score for k equals 9 is 0.0395105839697
    Average score for k equals 11 is -0.00679828306554
    Average score for k equals 13 is 0.00242304197865
    Average score for k equals 15 is -0.0613618518391
    Average score for k equals 17 is 0.01399996497
    Average score for k equals 19 is 0.0517900026164
    Average score for k equals 21 is 0.00894265160769
    Average score for k equals 23 is 0.0658495285783
    Average score for k equals 25 is 0.0332102837285
    Average score for k equals 27 is 0.0758501097966
    Average score for k equals 29 is 0.0399910529964
    Average score for k equals 31 is 0.0609545345737
    Average score for k equals 33 is 0.0579877274008
    Average score for k equals 35 is 0.0689876734801
    Average score for k equals 37 is 0.0725677590559
    Average score for k equals 39 is 0.0777651703349
    Average score for k equals 41 is 0.0603003250034
    Average score for k equals 43 is 0.0778407086593
    Average score for k equals 45 is 0.0619539167038
    Average score for k equals 47 is 0.0385661297188
    Average score for k equals 49 is 0.0407148821476


## Findings

The score function gives the coefficient of determination R^2 of the prediction.

This metric is used to explain the amount of variance in the dependent variable based on the feature vector (the independent variables) of the k-nn model.

The k-nn model lends itself well to the type of problem we are presented with. However the results are very poor and random. 

There is clearly bias in the data, mostly attributed to the large difference between the stores sales data.

In this first iteration the number of periods which contribute towards the total sales period is completely ignored. The model results are used as a benchmark against a version where there is a defined constraints on the number of periods of a store.

## Attempts to improve the model.

From here the aim is to try and improve the model by selecting stores with more historical monthly data.

This is attempted by filtering out the stores that have less than 6 months of data, irrelevant of when the sales were recorded.

The average number of periods in the full data set (for which there are surrounding information) is 9.


```python
print("The mean number of periods for the dataset is: {0}".format(str(reduce(lambda x, y: x + y, result['months_of_data_count']) / len(result['months_of_data_count']))))

# filter out stores with less than 6 periods
trimmed_result_df = result.loc[result['months_of_data_count'] > 5]

amenities_array = raw[0]['surroundings'].keys()

column_names_amenities = copy.deepcopy(amenities_array)
column_names_amenities = ['store_id'] + column_names_amenities

full_feature_amenities_df = pd.DataFrame(columns = column_names_amenities, index=[0])

for _, surroundings_obj in enumerate(raw):
    
    amenities_feature_dict = {}
    store_id = surroundings_obj['store_code']
    amenities_feature_dict['store_id'] = store_id
    
    for _, key in enumerate(amenities_array):
        amenities_feature_dict[key] = len(surroundings_obj['surroundings'][key])  
        
    feature_amenities_row = pd.DataFrame(data = amenities_feature_dict, columns = column_names_amenities, index=[0])
    
    full_feature_amenities_df = pd.concat([full_feature_amenities_df, feature_amenities_row])

full_feature_amenities_df = full_feature_amenities_df[1:]
full_feature_amenities_df.reset_index(drop=True, inplace=True)

# select the store ids that are only in the surrounding dataset
ops_df = trimmed_result_df.loc[trimmed_result_df['store_id'].isin(full_feature_amenities_df['store_id'])]

# join on store_id key to append the total_products_sold
# merged_df = pd.merge(full_feature_amenities_df, ops_df[['store_id', 'total_products_sold']], on='store_id')
merged_df = pd.merge(full_feature_amenities_df, ops_df[['store_id', 'total_products_sold']], on='store_id').dropna()
```

    The mean number of periods for the dataset is: 9


## Final fit and analysis


```python
# creating odd list of K for KNN
neigh_list = list(range(5,25))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, neigh_list)

for _, weight_type in enumerate(['uniform', 'distance']):
    
    print('\n #### Starting with {} weight #### \n'.format(weight_type))
    
    
    for kn in neighbors:

        cross_validation_array = []

        for cv_iteration in range(0, 10):

            # sample random index numbers for splitting the dataset. Will go for 85% training and 15% testing.
            msk = np.random.rand(len(merged_df)) <= 0.5

            train_df = merged_df[msk]

            test_df = merged_df[~msk]

            training_fit_set_columns = [col for col in train_df.columns if col not in ['store_id', 'total_products_sold']]
            training_fit_df = normalize_df(train_df[training_fit_set_columns])
            training_target_df = train_df['total_products_sold']

            test_fit_set_columns = [col for col in test_df.columns if col not in ['store_id', 'total_products_sold']]
            test_fit_df = normalize_df(test_df[test_fit_set_columns])
            test_target_df = test_df['total_products_sold']

            # fit model 
            neigh = KNeighborsRegressor(n_neighbors=kn, weights=weight_type)
            neigh.fit(training_fit_df, training_target_df) 
            neigh.predict(test_fit_df)

            cross_validation_array.append(neigh.score(test_fit_df, test_target_df))


        avg_score = reduce(lambda x, y: x + y, cross_validation_array) / len(cross_validation_array)

        print('Average score for k equals {0} is {1}'.format(kn, avg_score))
```

    
     #### Starting with uniform weight #### 
    
    Average score for k equals 5 is -0.271350324112
    Average score for k equals 7 is 0.0135371022582
    Average score for k equals 9 is -0.00469373996343
    Average score for k equals 11 is 0.0378195216296
    Average score for k equals 13 is 0.0137495578115
    Average score for k equals 15 is 0.0158002189937
    Average score for k equals 17 is 0.0254212262177
    Average score for k equals 19 is 0.0402337129143
    Average score for k equals 21 is 0.0330036314994
    Average score for k equals 23 is 0.0206787709384
    
     #### Starting with distance weight #### 
    
    Average score for k equals 5 is 0.0398575403335
    Average score for k equals 7 is 0.0191720951407
    Average score for k equals 9 is -0.0469526865433
    Average score for k equals 11 is 0.0382242870905
    Average score for k equals 13 is 0.0579435435849
    Average score for k equals 15 is 0.0136282332315
    Average score for k equals 17 is 0.0164974875418
    Average score for k equals 19 is 0.0485566022353
    Average score for k equals 21 is 0.0472226269646
    Average score for k equals 23 is 0.0119908965637


## Conclusion

First of all, the number of k-neighbour had to be adjusted since there are less degrees of freedom due to the restriction on the dataset.

Unfortunately there is not visible improvement over the previous iteration.

Different modeling features could be used to create more accurate predictions. As it is now, the model has no power to explain the sales of a store given its surroundings. 

The data is however prepared in a way which could be grouped and pivoted to analyze the coordinates, opening hours of surrounding stores and the labels under the 'type' of surroundings. 

Given more time, the model could surely be improved and insights could be generated based on the feature vectors of the surroundings.
