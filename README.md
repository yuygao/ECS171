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

- Link to Dataset: https://www.kaggle.com/datasets/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml?select=test.csv
- Link to our jupyter notebook: https://github.com/yuygao/ECS171/blob/2b4d2be221ccd90ac4a89d584f4aadff8738c84d/Project%20Code%20and%20Dataset/ECS171_Project_Code.ipynb
- Number of Observations: 31,876 data records
- Number of variables: 20
- Missing Values: 0


# Data Overview
The original dataset contains 99,569 rows and 29 columns. For this project, we have extracted a subset consisting of **31,876 data** records and **20 columns** for the following analysis and modeling sections. In the generated dataset, there are 11 categorical variables and 9 numerical variables. To enable effective processing, we will encode the categorical data by transforming categorical data into integer classes. This transformation will help to assign integer values ranging from 0 to k-1, where 'k' represents the number of categories.

- If a category has less than 4 unique values, we will convert it to a numerical encoding (e.g. 0, 1, 2, 3). 
- If a category has more than 100 unique values, we will consider the meaning and relevance to our project topic.
- If we want to keep the high-cardinality category, we will extract the most common values as their own classes (e.g. 0 to 4), and assign 5 to more infrequent "other" values.

We have four main sections for our Data Exploration Milestone: 

**1. Column Descriptions**

**2. Data processing**

**3. Data distributions (Data Visualization)**

**4. Data splitting**

# Column Descriptions
![Column Descriptions](https://github.com/yuygao/ECS171/assets/112483058/e9818180-3002-4f2b-b133-d026c1145c39)

# Data processing
We cleaning dataset by the following four steps: 
1. **Checking Missing Values:** We examine any potential missing values within the dataset, if there are missing values exist, we count the number of missing values. For our case, the count of missing values is 0.
2. **Encoding Data:** Since we have 11 categorical variables, we need to transform these categorical variables into numerical variables for the next step of modeling. In our case, we translated the 'host_identity_verified' variable from True/False to 1/0. We assigned numerical values to the 'property_type' variable as follows: 'Apartment' is represented by 0, 'House' by 1, 'Townhouse' by 2, 'Hostel' by 3, and 'Other' by 4. The same rule (or logic) wad applied to the rest of the variables as well.
3. **Remove Variables with High Correlations:** We analyze the relationships between two compared variables using a heatmap. We establish a threshold of 0.7. If the correlation value is greater than 0.7, we decide to remove the variable from the dataset. In our case, we found high correlations among accommodations, beds, and bedrooms. Similarly, longitude, latitude, and zipcode exhibit strong correlations. There's also a high correlation between 'cleaning fee' and 'host_has_profile_pic'. Consequently, we removed 'longitude', 'latitude', 'accommodates', and 'host_has_profile_pic'. Finally, we generated a new heatmap and re-examine the correlations.
4. Recheck the count of missing values to ensure it has been reduced to zero, and our dataset is ready for modeling.
   
By these processes, we ensure that our dataset is completely cleansed and optimized for analysis. We removed the missing values and Nan values. Categorical variables are encoded into numerical variables.

# Data distributions (Data Visualization)
To understand the distribution and relationships between variables, we generate several visualizations:

- **Five-number Summary Statistics:**

It could provide us an overview of the distribution of our dataset. These five statistics are particularly useful for us to see the spread of the data. The five summary statistics include the smallest value (Minimum), the value marking the first quarter of the data (First Quartile), the middle value (Median), the value marking the third quarter of the data (Third Quartile), and the largest value (Maximum). Below is a table that shows these statistics within our dataset.
![Five-number Summary Statistics](https://github.com/yuygao/ECS171/assets/112483058/e94eaef4-8e96-45c9-81f9-e6fb1428049e)

- **Heatmap**

This shows us the correlation coefficient between every pair of variables. We remove any variables that are highly correlated (correlation > 0.7) to avoid multicollinearity issues. In our case, we found high correlations among accommodations, beds, and bedrooms. Similarly, longitude, latitude, and zipcode exhibit strong correlations. There's also a high correlation between cleaning fee and host_has_profile_pic. Consequently, we removed 'longitude', 'latitude', 'accommodates', and 'host_has_profile_pic'. Finally, we generated a new heatmap and re-examine the correlations. Below are presented two heatmaps:

Before removing the variables with high correlations:
![Heatmap1](https://github.com/yuygao/ECS171/assets/112483058/5ebc51b2-a4b2-479b-96ff-b4f2f0436aa4)
After removing the variables with high correlations:
![Heatmap 2](https://github.com/yuygao/ECS171/assets/112483058/199c985e-9fd8-4b6b-86ce-d51dbecf5f8b)

- **Histograms**

We plot histograms of each individual variable to see the details of their distributions. This allows us to check for outliers, skewness, and other properties. In our case, since most of our variables are categorical, the histograms plot could show us the frequency distribution and patterns within our data. For example, for property type, where we observed four different types (0-4) representing 'Apartment' as 0, 'House' as 1, 'Townhouse' as 2, 'Hostel' as 3, and ‘Others’ as 4. We observed that 'Apartment' has approximately 20,000 occurrences, 'House' is around 7,600, 'Townhouse' has approximately 1,000, while 'Hostel' does not have any occurrences. Then, the category 'Others' is present in about 2,600 occurrences. The remaining variables can be analyzed using a similar logic.
![histogram](https://github.com/yuygao/ECS171/assets/112483058/22bad2ec-76f5-424c-9a1f-d1a57a91e144)

- **Q-Q plot**

A QQ plot is used to assess whether our dataset follows a normal distribution. If data points lie approximately along a straight line, it indicates that the data is approximated by the normal distribution. In our case, we could see that our data points of log price fall along a dashed line, so it shows a good fit to the normal distribution.
![QQ](https://github.com/yuygao/ECS171/assets/112483058/65fc31fa-2d4c-4d0f-840c-e8e861920be4)

- **Pairplots**

Using seaborn pairplot, we visualize pairwise relationships between all variables via a matrix of scatterplots. The lower triangle shows the scatter plots while the upper triangle displays the Pearson correlation coefficient. Pairplots provide a quick overview of the correlations and distributions between all variable pairs in one chart.
![pairplot](https://github.com/yuygao/ECS171/assets/112483058/ffb1f30b-57bc-4b74-9d22-dcd15c0028a4)

By combining Five-number Summary Statistics, heatmaps, histograms, Q-Q plot and pairplots, we can comprehensively explore the univariate and bivariate relationships in the dataset. 

# Data splitting 
After the data cleaning and processing phases, we divided the dataset into an 8:2 split. We proceed to evaluate variable accuracy, concentrating on key factors such as 'instant_bookable' and 'cleaning fee'.
```ruby
X = df.drop('instant_bookable', axis=1)  # Features
y = df['instant_bookable'] # Target value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- X_train shape: (25500, 15)
- X_test shape: (6376, 15)
- y_train shape: (25500,)
- y_test shape: (6376,)



## Assignment 2: Preprocessing & First Model building and evaluation Milestone
For this milestone, we will focus on four key steps:

1. Finish major preprocessing
    - Imputing your data, 
    - Encoding your data,
    - Feature expansion, 
    - Scaling and/or transforming your data,
    - Data Splitting
    
2. Train our first model
3. Evaluate our model compare training vs test error
4. Where does our model fit in the fitting graph?
   
As we approach this milestone, let's take a moment to see the data preprocessing frist:
 - First, in our initial preprocessing step, we started by examining the raw dataset for missing or duplicate values. We found that our dataset doesn't contain any missing values, so there's no need to drop or replace null values with means, medians, or other values.
 - Next, we organized the dataset into two main categories: categorical variables and numerical variables. We've identified 10 categorical variables, which include property_type, room_type, bed_type, cancellation_policy, cleaning_fee, city, host_has_profile_pic, host_identity_verified, host_response_rate, and instant_bookable. Additionally, there are 9 numerical variables, such as log_price1, accommodates, bathrooms, latitude, longitude, number_of_reviews, review_scores_rating, bedrooms, and beds.
 - Once we distinguished between the different variable types, we applied label encoding to the categorical variables within the cleaned datasets. This encoding process allowed us to convert string labels into numerical representations, which is essential for handling categorical variables in our analysis. Then, we normalized our data by standardizing the numerical data. After the dataset was transformed, we divided it into training and testing sets using an 80:20 ratio.
      - For specific categorical variables like property_type, room_type, bed_type, cancellation_policy, and city, we use a hard label encoding technique. This method allows us to create a mapping dictionary and assign unique whole number integers to each category within these variables.
      - For other variables such as host_identity_verified, instant_bookable, and cleaning_fee, we use the LabelEncoder() method for encoding. 'True' is encoded as 1, while 'False' is encoded as 0.
- Then, we ensure that our dataset is appropriately scaled by applying normalization using the MinMaxScaler() method. This step helps us to standardize the numerical features, ensuring they fall within a consistent range for our analysis. Regarding our target variable, "price," it's important to note that it has already a transformation from its original price to the natural logarithm (log of price) in the raw dataset we obtained from Kaggle. As a result, we keep it in its log-transformed state and there's no need for further normalization, as this transformation is essential for our analysis.

- In the last step of our data preprocessing, we divided the normalized dataset into training and testing sets with an 80:20 ratio.
```ruby
# Define features (attributes) and labels
X = normalized_data_merged.drop(['log_price'], axis=1)
y = normalized_data_merged['log_price']

# labels = np.unique(y)
# print("Unique labels:", labels)

# Split the data into training and testing sets with 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
```
- X_train shape: (25500, 15)
- X_test shape: (6376, 15)
- y_train shape: (25500,)
- y_test shape: (6376,)
  
For Exploratory Data Analysis (EDA) step,
 - We plot the dataset with histogram, scatter plot and box plot for data visualization to view the relationship between target variable log_price and other variables.
 - We create correlation matrices to identify columns with high correlations. Based on the basic correlation matrix, we remove a particular column exceeding the threshold (0.7) and recompute the correlation matrix. In our case, we observed that the columns 'bedrooms,' 'beds,' 'accommodates,' 'longitude,' and 'latitude' exhibit high correlations. As a result, we removed the following columns: 'longitude,' 'latitude,' 'accommodates,' and 'host_has_profile_pic.' After this adjustment, we rechecked the correlation values and found that no columns showed high correlations. So it could indicate that our dataset is now better suited for our analysis without strong multicollinearity issues.
   
In the second section, where we focus on training our first model, 
    - For the first model, we choose XGBoost modeling.
      - **Visualization**: we create a scatter plot that visually compares predicted prices with actual log_prices. Additionally, we generate two types of Feature Importance Plots to gain insights into the significance of different features in our model.

![price2_0](https://github.com/yuygao/ECS171/assets/112483058/5231a2c0-938c-4250-9bfd-118da87dc2f5)

![important1_0](https://github.com/yuygao/ECS171/assets/112483058/0a55b7b0-0002-4cbb-804f-1765379ca9ed)

![important2_0](https://github.com/yuygao/ECS171/assets/112483058/14fe3261-a64c-4cb4-88e6-66ca18da1128)

   - **Dataset Used:** We use the normalized dataset for our model. Upon running the model, we obtain the following evaluation metrics:
     -  Mean Squared Error: 0.17742309717426058
     -  Root Mean Squared Error: 0.4212162119081607
     -  R-squared: 0.6199305447777961


For the third section and fourth section, we focus on evaluating our model compare training vs test error and assessing model complexity, we have the following steps:
- Error Calculation: To see the fitness of our model and determine whether it is overfitting or underfitting, we compared training and test errors.
  - Mean Squared Error(train): 0.1723742638274122
  - Mean Squared Error(test): 0.18030066467327774
  - Mean Squared Error(train): 0.1517298671566797
  - Mean Squared Error(test): 0.16809802623341427
  - Mean Squared Error(train): 0.1383754619210304
  - Mean Squared Error(test): 0.16871127829132468
  - Mean Squared Error(train): 0.12179095868127136
  - Mean Squared Error(test): 0.17274424303443245
  - Mean Squared Error(train): 0.1003135315353569
  - Mean Squared Error(test): 0.17570978840673054
    
We computed the Mean Squared Error (MSE) for our model at different complexity levels, ranging from 1 to 5. The results showed a trend where the test MSE reached its lowest point, 0.16378396806262444, at a complexity level of 2. At this complexity level, the gap between the training and testing MSE was smaller compared to the gap at complexity 3. It means that a model complexity of 2 shows a good balance. Beyond complexity 3, the model had signs of overfitting, where it performed exceptionally well on the training data but performed poorly on unseen test data. 

![final](https://github.com/yuygao/ECS171/assets/112483058/2b849a5c-0562-4d2e-8b22-5c1b9ebcbf66)
