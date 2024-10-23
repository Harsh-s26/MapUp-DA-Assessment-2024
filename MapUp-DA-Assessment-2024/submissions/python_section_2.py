#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# Question 9: Distance Matrix Calculation
# Create a function named calculate_distance_matrix that takes the dataset-2.csv as input and generates a DataFrame representing distances between IDs.
# 
# The resulting DataFrame should have cumulative distances along known routes, with diagonal values set to 0. If distances between toll locations A to B and B to C are known, then the distance from A to C should be the sum of these distances. Ensure the matrix is symmetric, accounting for bidirectional distances between toll locations (i.e. A to B is equal to B to A).

# In[2]:


df=pd.read_csv(r'C:/Users/jyo14/Downloads/dataset-2.csv')


# In[3]:


df.head()


# In[7]:


def calculate_distance_matrix(file_path):
    locations = pd.concat([df['id_start'], df['id_end']]).unique()
    
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    
    np.fill_diagonal(distance_matrix.values, 0)
  
    for _, row in df.iterrows():
        start_id = row['id_start']
        end_id = row['id_end']
        dist = row['distance']
        distance_matrix.loc[start_id, end_id] = dist
        distance_matrix.loc[end_id, start_id] = dist
    for k in locations:
        for i in locations:
            for j in locations:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]
    
    return distance_matrix

file_path = 'dataset-2.csv'
result_matrix = calculate_distance_matrix(file_path)
print(result_matrix)


# 10)Question 10: Unroll Distance Matrix
# Create a function unroll_distance_matrix that takes the DataFrame created in Question 9. The resulting DataFrame should have three columns: columns id_start, id_end, and distance.
# 
# All the combinations except for same id_start to id_end must be present in the rows with their distance values from the input DataFrame.

# In[8]:


import pandas as pd

def unroll_distance_matrix(distance_matrix):
    unrolled = distance_matrix.stack().reset_index()
    unrolled.columns = ['id_start', 'id_end', 'distance']
    unrolled = unrolled[unrolled['id_start'] != unrolled['id_end']]
    return unrolled
unrolled_df = unroll_distance_matrix(result_matrix)
print(unrolled_df)


# Question 11: Finding IDs within Percentage Threshold
# Create a function find_ids_within_ten_percentage_threshold that takes the DataFrame created in Question 10 and a reference value from the id_start column as an integer.
# 
# Calculate average distance for the reference value given as an input and return a sorted list of values from id_start column which lie within 10% (including ceiling and floor) of the reference value's average.

# In[9]:


import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_value):
    ref_rows = df[df['id_start'] == reference_value]
    ref_avg_distance = ref_rows['distance'].mean()
    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
    within_threshold = avg_distances[
        (avg_distances['distance'] >= lower_bound) & 
        (avg_distances['distance'] <= upper_bound)
    ]
    
    result = within_threshold['id_start'].sort_values().tolist()
    
    return result
reference_value = 1 
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_value)
print(ids_within_threshold)


# Question 12: Calculate Toll Rate
# Create a function calculate_toll_rate that takes the DataFrame created in Question 10 as input and calculates toll rates based on vehicle types.
# 
# The resulting DataFrame should add 5 columns to the input DataFrame: moto, car, rv, bus, and truck with their respective rate coefficients. The toll rates should be calculated by multiplying the distance with the given rate coefficients for each vehicle type:
# 
# 0.8 for moto
# 1.2 for car
# 1.5 for rv
# 2.2 for bus
# 3.6 for truck

# In[10]:


import pandas as pd

def calculate_toll_rate(df):
    rate_coefficients = {
        'moto': 0.05,
        'car': 0.1,
        'rv': 0.2,
        'bus': 0.3,
        'truck': 0.4
    }
    
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate
    
    return df
toll_rate_df = calculate_toll_rate(unrolled_df)
print(toll_rate_df)


# Question 13: Calculate Time-Based Toll Rates
# Create a function named calculate_time_based_toll_rates that takes the DataFrame created in Question 12 as input and calculates toll rates for different time intervals within a day.
# 
# The resulting DataFrame should have these five columns added to the input: start_day, start_time, end_day, and end_time.
# 
# start_day, end_day must be strings with day values (from Monday to Sunday in proper case)
# start_time and end_time must be of type datetime.time() with the values from time range given below.
# Modify the values of vehicle columns according to the following time ranges:
# 
# Weekdays (Monday - Friday):
# 
# From 00:00:00 to 10:00:00: Apply a discount factor of 0.8
# From 10:00:00 to 18:00:00: Apply a discount factor of 1.2
# From 18:00:00 to 23:59:59: Apply a discount factor of 0.8
# Weekends (Saturday and Sunday):
# 
# Apply a constant discount factor of 0.7 for all times.
# For each unique (id_start, id_end) pair, cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span all 7 days of the week (from Monday to Sunday).

# In[12]:


import datetime

def calculate_time_based_toll_rates(df):
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_intervals = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0), 0.8),  
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0), 1.2),
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59), 0.8) 
    ]
    weekend_discount = 0.7
    rows = []
    for idx, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        original_rates = {
            'moto': row['moto'],
            'car': row['car'],
            'rv': row['rv'],
            'bus': row['bus'],
            'truck': row['truck']
        }
    
        for day in days_of_week:
            if day in ['Saturday', 'Sunday']:
                for start_time, end_time, _ in time_intervals:
                    modified_rates = {vehicle: rate * weekend_discount for vehicle, rate in original_rates.items()
                    rows.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        'moto': modified_rates['moto'],
                        'car': modified_rates['car'],
                        'rv': modified_rates['rv'],
                        'bus': modified_rates['bus'],
                        'truck': modified_rates['truck']
                    })
            
            # Weekdays (Monday to Friday)
            else:
                for start_time, end_time, discount_factor in time_intervals:
                    # Apply weekday discount factor
                    modified_rates = {vehicle: rate * discount_factor for vehicle, rate in original_rates.items()}
                    
                    # Append the row data as a dictionary
                    rows.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        'moto': modified_rates['moto'],
                        'car': modified_rates['car'],
                        'rv': modified_rates['rv'],
                        'bus': modified_rates['bus'],
                        'truck': modified_rates['truck']
                    })
    
    expanded_df = pd.DataFrame(rows)
    
    return expanded_df
time_based_toll_rate_df = calculate_time_based_toll_rates(toll_rate_df)
print(time_based_toll_rate_df)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




