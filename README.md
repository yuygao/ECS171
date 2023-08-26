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

Data processing
We cleaning dataset by the following four steps: 
1. Checking Missing Values: We examine any potential missing values within the dataset, if there are missing values exist, we count the number of missing values. For our case, the count of missing values is none.
2. Encoding Data: Since we have 10 categorical variables, we need to transform these categorical variables into numerical variables for the next step of modeling. In our case, we translate the 'host_identity_verified' variable from True/False to 1/0. We assigned numerical values to the 'property_type' variable as follows: 'Apartment' is represented by 0, 'House' by 1, 'Townhouse' by 2, 'Hostel' by 3, and 'Other' by 4. The same rule/logic was applied to the rest of the variables as well.
3. Remove Variables with High Correlations: We analyze the relationships between two compared variables using a heatmap. We establish a threshold of 0.7. If the correlation value is greater than 0.7, we decide to remove the variable from the dataset. In our case, we found high correlations among accommodations, beds, and bedrooms. Similarly, longitude, latitude, and zipcode exhibit strong correlations. There's also a high correlation between cleaning fee and host_has_profile_pic. Consequently, we remove 'longitude', 'latitude', 'accommodates', and 'host_has_profile_pic'. Finally, we generate a new heatmap and re-examine the correlations.
4. Recheck the count of missing values to ensure it has been reduced to zero, and our dataset is ready for modeling.
By these processes,we ensure that our dataset is thoroughly cleansed and optimized for analysis. We removed the missing values and Nan values. Categorical variables are encoded into numerical variables.
