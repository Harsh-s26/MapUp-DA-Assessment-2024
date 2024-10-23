#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# Problem Statement:
# 
# 1)Write a function that takes a list and an integer n, and returns the list with every group of n elements reversed. If there are fewer than n elements left at the end, reverse all of them.
# 
# Requirements:
# 
# You must not use any built-in slicing or reverse functions to directly reverse the sublists.
# The result should reverse the elements in groups of size n.

# In[4]:


def rotate_list(lst,position):
    position = position%len(lst)
    rtd_list = lst[position:]+lst[:position]
    return rtd_list
lst=[10,20,30,40,50,60,70]
position = 4
print(rotate_list(lst,position))


# 2)Problem Statement:
# 
# Write a function that takes a list of strings and groups them by their length. The result should be a dictionary where:
# 
# The keys are the string lengths.
# The values are lists of strings that have the same length as the key.
# Requirements:
# 
# Each string should appear in the list corresponding to its length.
# The result should be sorted by the lengths (keys) in ascending order.

# In[8]:


def group_strings_by_length(strings):
    grouped_strings = {}
    for string in strings:
        length = len(string)
        if length not in grouped_strings:
              grouped_strings[length] = []
        grouped_strings[length].append(string)

    return dict(sorted(grouped_strings.items()))
strings = ["one","two",'three','four']
result = group_strings_by_length(strings)
print(result)


# 3) You are given a nested dictionary that contains various details (including lists and sub-dictionaries). Your task is to write a Python function that flattens the dictionary such that:
# 
# Nested keys are concatenated into a single key with levels separated by a dot (.).
# List elements should be referenced by their index, enclosed in square brackets (e.g., sections[0]).
# For example, if a key points to a list, the index of the list element should be appended to the key string, followed by a dot to handle further nested dictionaries.
# 
# Requirements:
# 
# Nested Dictionary: Flatten nested dictionaries into a single level, concatenating keys.
# Handling Lists: Flatten lists by using the index as part of the key.
# Key Separator: Use a dot (.) as a separator between nested key levels.
# Empty Input: The function should handle empty dictionaries gracefully.
# Nested Depth: You can assume the dictionary has a maximum of 4 levels of nesting.

# In[10]:


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for idx, item in enumerate(v):
                list_key = f"{new_key}[{idx}]"
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, list_key, sep=sep).items())
                else:
                    items.append((list_key, item))
        else:
            items.append((new_key, v))
    
    return dict(items)

nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}


flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)


# 4)Problem Statement:
# 
# You are given a list of integers that may contain duplicates. Your task is to generate all unique permutations of the list. The output should not contain any duplicate permutations.

# In[11]:


def permute(nums):
    result = []
    used = [False] * len(nums)
    def backtrack(start, path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(start, len(nums)):
            if not used[i] and (i == start or nums[i] != nums[i - 1] or used[i - 1]):
                used[i] = True
                path.append(nums[i])
                backtrack(start + 1, path)
                path.pop()
                used[i] = False

    nums.sort() 
    backtrack(0, [])
    return result
nums = [1, 1, 2]
permutations = permute(nums)
print(permutations)


# 5)Problem Statement:
# 
# You are given a string that contains dates in various formats (such as "dd-mm-yyyy", "mm/dd/yyyy", "yyyy.mm.dd", etc.). Your task is to identify and return all the valid dates present in the string.
# 
# You need to write a function find_all_dates that takes a string as input and returns a list of valid dates found in the text. The dates can be in any of the following formats:
# 
# dd-mm-yyyy
# mm/dd/yyyy
# yyyy.mm.dd
# You are required to use regular expressions to identify these dates.
# 

# In[18]:


import re

def find_all_dates(text):
    all_dates = []
    date_formats = [
        r"\b(0?[1-9]|[12][0-9]|3[01])-(0?[1-9]|1[0-2])-\d{4}",   
        r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/\d{4}",   
        r"\b\d{4}\.(0?[1-9]|1[0-2])\.(0?[1-9]|[12][0-9]|3[01])" 
    ]
    
    for date_format in date_formats:
        matches = re.findall(date_format, text)
        for match in matches:
            if isinstance(match, tuple):
                all_dates.append("-".join(match))  # Join the tuple to make the full date
            else:
                all_dates.append(match)
    
    return all_dates

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
valid_dates = find_all_dates(text)
print(valid_dates)


# 6)You are given a polyline string, which encodes a series of latitude and longitude coordinates. Polyline encoding is a method to efficiently store latitude and longitude data using fewer bytes. The Python polyline module allows you to decode this string into a list of coordinates.
# 
# Write a function that performs the following operations:
# 
# Decode the polyline string using the polyline module into a list of (latitude, longitude) coordinates.
# Convert these coordinates into a Pandas DataFrame with the following columns:
# latitude: Latitude of the coordinate.
# longitude: Longitude of the coordinate.
# distance: The distance (in meters) between the current row's coordinate and the previous row's one. The first row will have a distance of 0 since there is no previous point.
# Calculate the distance using the Haversine formula for points in successive rows.

# In[20]:


get_ipython().system('pip install polyline')


# In[21]:


import pandas as pd
import polyline
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def decode_polyline_to_df(polyline_str):
    coords = polyline.decode(polyline_str)
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    df['distance'] = 0.0
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, 'latitude'], df.loc[i - 1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)

    return df

polyline_str = 'u{~vFvyys@fAnE'
df = decode_polyline_to_df(polyline_str)
print(df)


# 7)Write a function that performs the following operations on a square matrix (n x n):
# 
# Rotate the matrix by 90 degrees clockwise.
# After rotation, for each element in the rotated matrix, replace it with the sum of all elements in the same row and column (in the rotated matrix), excluding itself.
# The function should return the transformed matrix.

# In[24]:


def rotate_and_transform_matrix(matrix):
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j] 

    return final_matrix
matrix = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
result = rotate_and_transform_matrix(matrix)
print(result)


# 8)Time Check
# You are given a dataset, dataset-1.csv, containing columns id, id_2, and timestamp (startDay, startTime, endDay, endTime). The goal is to verify the completeness of the time data by checking whether the timestamps for each unique (id, id_2) pair cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span all 7 days of the week (from Monday to Sunday).
# 
# Create a function that accepts dataset-1.csv as a DataFrame and returns a boolean series that indicates if each (id, id_2) pair has incorrect timestamps. The boolean series must have multi-index (id, id_2).

# In[ ]:


def check_time_data_completeness(df):
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    df.set_index(['id', 'id_2'], inplace=True)
    def check_completeness(group):
        days_covered = group['start_timestamp'].dt.dayofweek.unique()
        all_days_covered = set(range(7))  
        time_coverage = (group['start_timestamp'].min().time() <= pd.Timestamp('00:00:00').time() and 
                         group['end_timestamp'].max().time() >= pd.Timestamp('23:59:59').time())
        return len(days_covered) == 7 and time_coverage
    completeness = df.groupby(level=[0, 1]).apply(check_completeness)

    return completeness



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




