# ECS171

Link to Dataset: https://www.kaggle.com/datasets/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml?select=test.csv 

# Data Overview
The original dataset contains 99,569 rows and 29 columns. For this project, we loaded a subset of 31876 data records and  20 columns for the following analysis and modeling sections. In the generated dataset, there are 10 categorical variables and 9 numerical variables. As a first step, we will check the scale of each categorical variable and encode categorical data by transforming categorical data into integer classes, then we take k number of categories and assign integer values from 0 to k-1.

If a category has less than 4 unique values, we will convert it to a numerical encoding (e.g. 0, 1, 2, 3).
If a category has more than 100 unique values, we will consider the meaning and relevance to our project topic.
If we want to keep the high-cardinality category, we will extract the most common values as their own classes (e.g. 0 to 4), and assign 5 to more infrequent "other" values.

We have four main sections for our Data Exploration Milestone: 
- Column Descriptions
- Data processing
- Data distributions (Data Visualization)
- Data splitting 
