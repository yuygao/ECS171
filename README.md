# ECS171
Install Library
```ruby
!pip3 install xgboost
```
Set Up Coding Environment
```ruby
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

Link to Dataset: https://www.kaggle.com/datasets/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml?select=test.csv 

# Data Overview
The original dataset contains 99,569 rows and 29 columns. For this project, we have extracted a subset consisting of **31,876 data** records and **20 columns** for the following analysis and modeling sections. In the generated dataset, there are 11 categorical variables and 9 numerical variables. To enable effective processing, we will encode the categorical data by transforming categorical data into integer classes. This transformation will help to assign integer values ranging from 0 to k-1, where 'k' represents the number of categories.

- If a category has less than 4 unique values, we will convert it to a numerical encoding (e.g. 0, 1, 2, 3).
- If a category has more than 100 unique values, we will consider the meaning and relevance to our project topic.
- If we want to keep the high-cardinality category, we will extract the most common values as their own classes (e.g. 0 to 4), and assign 5 to more infrequent "other" values.

We have four main sections for our Data Exploration Milestone: 
- Column Descriptions
- Data processing
- Data distributions (Data Visualization)
- Data splitting

# Column Descriptions
![Column Descriptions](https://github.com/yuygao/ECS171/assets/112483058/e9818180-3002-4f2b-b133-d026c1145c39)

# Data processing
We cleaning dataset by the following four steps: 
1. Checking Missing Values: We examine any potential missing values within the dataset, if there are missing values exist, we count the number of missing values. For our case, the count of missing values is none.
2. Encoding Data: Since we have 10 categorical variables, we need to transform these categorical variables into numerical variables for the next step of modeling. In our case, we translate the 'host_identity_verified' variable from True/False to 1/0. We assigned numerical values to the 'property_type' variable as follows: 'Apartment' is represented by 0, 'House' by 1, 'Townhouse' by 2, 'Hostel' by 3, and 'Other' by 4. The same rule/logic was applied to the rest of the variables as well.
3. Remove Variables with High Correlations: We analyze the relationships between two compared variables using a heatmap. We establish a threshold of 0.7. If the correlation value is greater than 0.7, we decide to remove the variable from the dataset. In our case, we found high correlations among accommodations, beds, and bedrooms. Similarly, longitude, latitude, and zipcode exhibit strong correlations. There's also a high correlation between cleaning fee and host_has_profile_pic. Consequently, we remove 'longitude', 'latitude', 'accommodates', and 'host_has_profile_pic'. Finally, we generate a new heatmap and re-examine the correlations.
4. Recheck the count of missing values to ensure it has been reduced to zero, and our dataset is ready for modeling.
By these processes,we ensure that our dataset is thoroughly cleansed and optimized for analysis. We removed the missing values and Nan values. Categorical variables are encoded into numerical variables.

# Data distributions (Data Visualization)
To understand the distribution and relationships between variables, we generate several visualizations:

- Five-number Summary Statistics: 
It could provide us an overview of the distribution of our dataset. These five statistics are particularly useful for us to see the spread of the data. The five summary statistics include the smallest value (Minimum), the value marking the first quarter of the data (First Quartile), the middle value (Median), the value marking the third quarter of the data (Third Quartile), and the largest value (Maximum). Below is a table that shows these statistics within our dataset.
![Five-number Summary Statistics](https://github.com/yuygao/ECS171/assets/112483058/e94eaef4-8e96-45c9-81f9-e6fb1428049e)

- Heatmap
This shows the correlation coefficient between every pair of variables. We remove any variables that are highly correlated (correlation > 0.7) to avoid multicollinearity issues. In our case, we found high correlations among accommodations, beds, and bedrooms. Similarly, longitude, latitude, and zipcode exhibit strong correlations. There's also a high correlation between cleaning fee and host_has_profile_pic. Consequently, we remove 'longitude', 'latitude', 'accommodates', and 'host_has_profile_pic'. Finally, we generate a new heatmap and re-examine the correlations. Below are presented two heatmaps:

Before removing the variables with high correlations:
![Heatmap1](https://github.com/yuygao/ECS171/assets/112483058/5ebc51b2-a4b2-479b-96ff-b4f2f0436aa4)
After removing the variables with high correlations:
![Heatmap 2](https://github.com/yuygao/ECS171/assets/112483058/199c985e-9fd8-4b6b-86ce-d51dbecf5f8b)
- Histograms
We plot histograms of each individual variable to see the details of their distributions. This allows us to check for outliers, skewness, and other properties. In our case, since most of our variables are categorical, the histograms plot could show us the frequency distribution and patterns within our data. For example, for property type, where we observe four different types (0-4) representing 'Apartment' as 0, 'House' as 1, 'Townhouse' as 2,  'Hostel' as 3, and ‘Others’ as 4. We observed that 'Apartment' has approximately 20,000 occurrences, 'House' is around 7,600, 'Townhouse' has approximately 1,000, while 'Hostel' does not have any occurrences. Then, the category 'Others' is present in about 2,600 occurrences. 
![histogram](https://github.com/yuygao/ECS171/assets/112483058/22bad2ec-76f5-424c-9a1f-d1a57a91e144)
- Q-Q plot
A QQ plot is used to assess whether our dataset follows a normal distribution. If our data points lie approximately along a straight line, it indicates that the data is approximated by the normal distribution. In our case, we could see that our data points of log price fall along a dashed line, so it suggests a good fit to the normal distribution.

![QQ](https://github.com/yuygao/ECS171/assets/112483058/65fc31fa-2d4c-4d0f-840c-e8e861920be4)
- Pairplots
Using seaborn pairplot, we visualize pairwise relationships between all variables via a matrix of scatterplots. The lower triangle shows the scatter plots while the upper triangle displays the Pearson correlation coefficient. Pairplots provide a quick overview of the correlations and distributions between all variable pairs in one chart.
![pairplot](https://github.com/yuygao/ECS171/assets/112483058/ffb1f30b-57bc-4b74-9d22-dcd15c0028a4)
By combining heatmaps, histograms, and pairplots, we can comprehensively explore the univariate and bivariate relationships in the dataset. By combining heatmaps, histograms, and pairplots, we can comprehensively explore the univariate and bivariate relationships in the dataset.

# Data splitting 
After the data cleaning and processing phases, we divided the dataset into an 8:2 split. We proceed to evaluate variable accuracy, concentrating on key factors such as 'instant_bookable,' 'cleaning fee,' and 'host_has_profile_pic.'

- X_train shape: (25500, 15)
- X_test shape: (6376, 15)
- y_train shape: (25500,)
- y_test shape: (6376,)
