# ECS171
Install Library
```ruby
!pip3 install xgboost
!pip3 install imblearn
!pip3 install reshape
!pip3 install graphviz
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, precision_score, recall_score, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import XGBRegressor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.model_selection import learning_curve, validation_curve
```
**Assignment 1: Data Exploration Milestone**
- Link to Dataset: https://www.kaggle.com/datasets/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml?select=test.csv
- Link to our jupyter notebook: https://github.com/yuygao/ECS171/blob/2b4d2be221ccd90ac4a89d584f4aadff8738c84d/Project%20Code%20and%20Dataset/ECS171_Project_Code.ipynb
- Number of Observations: 31,876 data records
- Number of variables: 20
- Missing Values: 0

**Assignment 2: Preprocessing & First Model building and evaluation Milestone**
- Link to our jupyter notebook: https://github.com/yuygao/ECS171/blob/642b1f715fd463407762b3f1dbbc94f31e0c8863/Preprocessing%20%26%20First%20Model%20building%20and%20evaluation%20Milestone.ipynb

**Final Submission**
- Link to our jupyter notebook: https://github.com/yuygao/ECS171/blob/90c7effd71d58d60df346660196a4efa02fbb344/ECS171_Final_Submission%20.ipynb

--- 

## Assignment 1: Data Exploration Milestone 

### Data Overview
The original dataset contains 99,569 rows and 29 columns. For this project, we have extracted a subset consisting of **31,876 data** records and **20 columns** for the following analysis and modeling sections. In the generated dataset, there are 11 categorical variables and 9 numerical variables. To enable effective processing, we will encode the categorical data by transforming categorical data into integer classes. This transformation will help to assign integer values ranging from 0 to k-1, where 'k' represents the number of categories.

- If a category has less than 4 unique values, we will convert it to a numerical encoding (e.g. 0, 1, 2, 3). 
- If a category has more than 100 unique values, we will consider the meaning and relevance to our project topic.
- If we want to keep the high-cardinality category, we will extract the most common values as their own classes (e.g. 0 to 4), and assign 5 to more infrequent "other" values.

We have four main sections for our Data Exploration Milestone: 

**1. Column Descriptions**

**2. Data processing**

**3. Data distributions (Data Visualization)**

**4. Data splitting**

### Column Descriptions
This table shows us the chosen variables, excluding those that have strong correlations above a threshold of 0.7.
![Column Descriptions](https://github.com/yuygao/ECS171/assets/112483058/e9818180-3002-4f2b-b133-d026c1145c39)

### Data processing
We cleaning dataset by the following four steps: 
1. **Checking Missing Values:** We examine any potential missing values within the dataset, if there are missing values exist, we count the number of missing values. For our case, the count of missing values is 0.
2. **Encoding Data:** Since we have 11 categorical variables, we need to transform these categorical variables into numerical variables for the next step of modeling. In our case, we translated the 'host_identity_verified' variable from True/False to 1/0. We assigned numerical values to the 'property_type' variable as follows: 'Apartment' is represented by 0, 'House' by 1, 'Townhouse' by 2, 'Hostel' by 3, and 'Other' by 4. The same rule (or logic) wad applied to the rest of the variables as well.
3. **Remove Variables with High Correlations:** We analyze the relationships between two compared variables using a heatmap. We establish a threshold of 0.7. If the correlation value is greater than 0.7, we decide to remove the variable from the dataset. In our case, we found high correlations among accommodations, beds, and bedrooms. Similarly, longitude, latitude, and zipcode exhibit strong correlations. There's also a high correlation between 'cleaning fee' and 'host_has_profile_pic'. Consequently, we removed 'longitude', 'latitude', 'accommodates', and 'host_has_profile_pic'. Finally, we generated a new heatmap and re-examine the correlations.
4. Recheck the count of missing values to ensure it has been reduced to zero, and our dataset is ready for modeling.
   
By these processes, we ensure that our dataset is completely cleansed and optimized for analysis. We removed the missing values and Nan values. Categorical variables are encoded into numerical variables.

### Data distributions (Data Visualization)
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

### Data splitting 
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

---

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

**Step One - Finish major preprocessing**

As we approach this milestone, let's take a moment to see the data preprocessing frist:
 - **Data Cleaning:** First, in our initial preprocessing step, we started by examining the raw dataset for missing or duplicate values. We found that our dataset doesn't contain any missing values, so there's no need to drop or replace null values with means, medians, or other values.
 - **Column Type Identification for Processed Datase:** Next, we organized the dataset into two main categories: categorical variables and numerical variables. We've identified 10 categorical variables, which include property_type, room_type, bed_type, cancellation_policy, cleaning_fee, city, host_has_profile_pic, host_identity_verified, host_response_rate, and instant_bookable. Additionally, there are 9 numerical variables, such as log_price1, accommodates, bathrooms, latitude, longitude, number_of_reviews, review_scores_rating, bedrooms, and beds.
 - **Label Encoding for Categorical Variable:** Once we distinguished between the different variable types, we applied label encoding to the categorical variables within the cleaned datasets. This encoding process allowed us to convert string labels into numerical representations, which is essential for handling categorical variables in our analysis. 
      - For specific categorical variables like property_type, room_type, bed_type, cancellation_policy, and city, we use a hard label encoding technique. This method allows us to create a mapping dictionary and assign unique whole number integers to each category within these variables.
      - For other variables such as host_identity_verified, instant_bookable, and cleaning_fee, we use the LabelEncoder() method for encoding. 'True' is encoded as 1, while 'False' is encoded as 0.
- **Normalize the numerical data:** Then, we ensure that our dataset is appropriately scaled by applying normalization using the MinMaxScaler() method. This step helps us to standardize the numerical features, ensuring they fall within a consistent range for our analysis.
- **Log multiplication of "Price":** Regarding our target variable, "price," it's important to note that it has already a transformation from its original price to the natural logarithm (log of price) in the raw dataset we obtained from Kaggle. As a result, we keep it in its log-transformed state and there's no need for further normalization, as this transformation is essential for our analysis.

- **Data Splitting:** In the last step of our data preprocessing, we divided the normalized dataset into training and testing sets with an 80:20 ratio.
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
- X_train shape: (25500, 13)
- X_test shape: (6376, 13)
- y_train shape: (25500,)
- y_test shape: (6376,)
  
For Exploratory Data Analysis (EDA) step,
 - We plot the dataset with histogram, scatter plot and box plot for data visualization to view the relationship between target variable log_price and other variables.
 - We create correlation matrices to identify columns with high correlations. Based on the basic correlation matrix, we remove a particular column exceeding the threshold (0.7) and recompute the correlation matrix. In our case, we observed that the columns 'bedrooms,' 'beds,' 'accommodates,' 'longitude,' and 'latitude' exhibit high correlations. As a result, we removed the following columns: 'longitude,' 'latitude,' 'accommodates,' and 'host_has_profile_pic.' After this adjustment, we rechecked the correlation values and found that no columns showed high correlations. So it could indicate that our dataset is now better suited for our analysis without strong multicollinearity issues.

**Step Two - Train our first model**

In the second section, where we focus on training our first model XGBoost modeling,
   
   - **Visualization:** we create a scatter plot that visually compares predicted prices with actual log_prices. Additionally, we generate two types of Feature Importance Plots to gain insights into the significance of different features in our model.

![price2_0](https://github.com/yuygao/ECS171/assets/112483058/5231a2c0-938c-4250-9bfd-118da87dc2f5)

![important1_0](https://github.com/yuygao/ECS171/assets/112483058/0a55b7b0-0002-4cbb-804f-1765379ca9ed)

![important2_0](https://github.com/yuygao/ECS171/assets/112483058/14fe3261-a64c-4cb4-88e6-66ca18da1128)


   - **Dataset Used:** We use the normalized dataset for our model. Upon running the model, we obtain the following evaluation metrics:
     -  Mean Squared Error: 0.17742309717426058
     -  Root Mean Squared Error: 0.4212162119081607
     -  R-squared: 0.6199305447777961


**Step Three & Four - Evaluate our model compare training vs test error and Check if Overfitting**

For the third section and fourth section, we focus on evaluating our model compare training vs test error and assessing model complexity, we have the following steps:

- **Error Calculation:** To see the fitness of our model and determine whether it is overfitting or underfitting, we compared training and test errors.
  
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

Let's analyze these MSE values to understand the model's performance:

- **First Iteration:**
Training MSE: 0.1724
Testing MSE: 0.1803

The initial iteration starts with a relatively close training and testing MSE, indicating that the model's performance on the training and testing data is reasonably similar.

- **Second Iteration:**
Training MSE: 0.1518
Testing MSE: 0.1682

In the second iteration, both training and testing MSE decrease, which is generally a positive sign. The model is improving its fit to the data.

- **Third Iteration:**
Training MSE: 0.1388
Testing MSE: 0.1695

The training MSE continues to decrease, suggesting that the model is fitting the training data better. However, the testing MSE increases slightly, which can be a sign of the model starting to overfit the training data.

- **Fourth Iteration:**
Training MSE: 0.1226
Testing MSE: 0.1705

Similar to the third iteration, the training MSE decreases, and the testing MSE continues to increase slightly. This trend may indicate further overfitting.

- **Fifth Iteration:**
Training MSE: 0.1020
Testing MSE: 0.1774

In the fifth iteration, the training MSE decreases significantly, indicating a very good fit to the training data. However, the testing MSE increases substantially, which suggests a significant overfitting. The model's performance on the testing data has significantly deteriorated compared to earlier iterations.


**Conclusion:** We computed the Mean Squared Error (MSE) for our model at different complexity levels, ranging from 1 to 5. The results showed a trend where the test MSE reached its lowest point, 0.16378396806262444, at a complexity level of 2. At this complexity level of 2, the gap between the training and testing MSE was smaller compared to the gap at complexity 3. It means that a model complexity of 2 shows a good balance (or optimal fit). At this complexity level of 3, 4 and 5, the model had signs of overfitting, where it performed exceptionally well on the training data but performed poorly on unseen test data. 

![final](https://github.com/yuygao/ECS171/assets/112483058/2b849a5c-0562-4d2e-8b22-5c1b9ebcbf66)

---

# Final Submission

## A. Introduction
Our group project is focused on predicting Airbnb listing prices in major US cities by building a robust prediction model. We're employing various machine learning algorithms to explore the incredible potential of data-driven decision-making in the business sector. The research significance of this project is that building a price prediction model will be profitable for many people. For individual hosts and Airbnb platforms, our price prediction models enhance occupancy rates and revenue by optimizing listing prices. This not only improves the user experience but also ensures competitiveness in local markets. Travelers benefit from predictive pricing by gaining quick insights into accommodation costs, helping them budget their trips more efficiently. Furthermore, the industry itself can leverage price modeling to gain insights into the dynamics and performance of the Airbnb rental market, leading to more informed conclusions about future growth. To conduct this project, we've turned to the Airbnb listings dataset available on Kaggle, focusing on major US cities. This dataset contains 31,876 records with 20 columns, encompassing 11 categorical and 9 numerical variables. Our analysis begins with essential data processing, including data cleaning and encoding. We then employ data visualization techniques to understand data distributions, guiding our decisions on data normalization. Following data preprocessing, we split the dataset into an 80% training set and a 20% testing set, designating "log_price" as the target variable. Subsequently, we construct predictive models on the training data, employing four machine learning approaches: XGBoost, LightGBM (LGBM), Random Forest, and the k-NN Algorithm. To evaluate these models, we employ various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R^2). Our model selection is based on a comprehensive comparison of these metrics, ensuring that we choose the most accurate model for predicting listing prices. In summary, our data analysis and machine learning techniques enable us to provide a well-informed solution to a real-world business challenge in the Airbnb market.

## B. Data Visualization

### Variables summary

This table shows us the chosen variables, excluding those that have strong correlations above a threshold of 0.7.

<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/e9818180-3002-4f2b-b133-d026c1145c39" alt="Column Descriptions">
</div>
<p align="center">Figure.1 Variables summary table</p>


### Five-number Summary Statistics
<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/e94eaef4-8e96-45c9-81f9-e6fb1428049e" alt="Five-number Summary Statistics" >
</div>
<p align="center">Figure.2 Five-number Summary Statistics table</p>

### Heatmap
#### Original Heatmap
<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/5ebc51b2-a4b2-479b-96ff-b4f2f0436aa4" alt="Heatmap1">
</div>
<p align="center">Figure.3 Original Heatmap</p>

#### Heatmap After removing the variables with high correlations

Drop a specific column which is greater than a threshold (0.7) and unneccessary variables 

<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/199c985e-9fd8-4b6b-86ce-d51dbecf5f8b" alt="Heatmap 2">
</div>
<p align="center">Figure.4 Heatmap After removing the variables with high correlations</p>

### Histograms
<p align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/22bad2ec-76f5-424c-9a1f-d1a57a91e144" alt="histogram">
</p>
<p align="center">Figure.5 Histograms of individual variable data distributions</p>

### Q-Q plot
<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/65fc31fa-2d4c-4d0f-840c-e8e861920be4" alt="Image Description">
</div>
<p align="center">Figure.6 Q-Q plot of checking normal distribution</p>

### PairPlots
<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/ffb1f30b-57bc-4b74-9d22-dcd15c0028a4" alt="pairplot">
</div>
<p align="center">Figure.7 PairPlots</p>

### Histogram of a numerical feature

<p align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/bd1ea6b9-fdf2-4e88-91d1-c32235d7f894" alt="Histogram">
</p>
<p align="center">Figure.8  Histograms of the frequence of number of bedrooms</p>

### Box plot of log_price by property_type

<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/114623522/600835a0-8ddf-48a7-8daf-7e9c25cf24af" alt="Image Alt Text">
</div>
<p align="center">Figure.9  Box plot of log_price by property_type</p>

### Scatter Plot of the relationship between log_price and the number of bedrooms

<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/114623522/3ed210db-6dc4-4ada-8b12-61a442272c4b" alt="Image Description">
</div>
<p align="center">Figure.10  Scatter Plot of log_price vs. the number of bedrooms</p>


### Scatter plot: The relationship between log_price and the number of beds
<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/114623522/32df85a3-2ce1-4c08-aa4f-285a4d6ff285" alt="Image Description">
</div>
<p align="center">Figure.11  Scatter Plot of the relationship between log_price and the number of beds</p>


### Scatter plot: The relationship between log_price and host_response_rate
<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/114623522/a01b2607-81c2-4f3d-9b67-cdf4d8b4e4b8" alt="Image Description">
</div>
<p align="center">Figure.12  Scatter Plot of log_price vs. host_response_rate</p>


### Scatter plot: The relationship between log_price and review_scores_rating
<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/114623522/3d03680d-58a8-4a34-a0db-bf5861e14a36" alt="Description of the image" />
</div>
<p align="center">Figure.13  Scatter Plot of log_price vs. review_scores_rating</p>

### Scatter plot: The relationship between log_price and number_of_reviews
<p align="center">
  <img src="https://github.com/yuygao/ECS171/assets/114623522/18d75593-5fa2-490e-a44a-b1547b5199ca" alt="Image Description">
</p>
<p align="center">Figure.14  Scatter Plot of log_price vs. number_of_reviews</p>

### Pie Chart of City
<p align="center">
  <img src="https://github.com/yuygao/ECS171/assets/114623522/bb204012-5079-46b5-9ba5-5261953991f8" alt="Image Description">
</p>
<p align="center">Figure.15  Pie Chart of City</p>


---
## C. Methods section 

**Research Question:**
1. How accurately can the factors predict Airbnb listing prices in the United States?
2. Which specific variables have the greatest impact on pricing, as determined by the most suitable prediction model?

**Methods section:**
The method section will include Initial Preprocessing, Data Exploration and Analysis, Data preprocessing, and Preparing for Model Building. 

**[Initial Preprocessing](https://github.com/yuygao/ECS171/blob/90c7effd71d58d60df346660196a4efa02fbb344/ECS171_Final_Submission%20.ipynb)：** 

To understand our dataset and prepare it for modification, we follow these steps:
 - Data Overview:
    - Load Dataset: Load the raw dataset.
    - Check the Information of the Original Dataset: Use the info() function.

 - Data Cleaning:
   - Check for missing values or duplicate values.
   - If any are found, either drop them or aggregate them from the raw dataset.
   - Create a new dataframe called df_cleaned.

In our initial preprocessing step, we began by examining the raw dataset for missing or duplicate values. We found that our dataset does not contain any missing values. Therefore, there is no need to drop or replace null values with means, medians, or other values.

**Data Exploration and Analysis:**

In this step, we include the following steps:
 - Original Data Exploration and Analysis (EDA)
    - Identify the Types of Columns in the Original Dataset: This involves categorizing columns as either Categorical or Numerical.
    - Basic Statistics of Original Data: We display basic statistics using the describe() function.

**Data preprocessing:**

This section covers data preprocessing and includes the following steps:
 - Correlation Analysis of the Original Dataset:
   - Prior to removing variables with high correlations, we conduct a correlation analysis.
   - Specifically, we check for columns with correlation values exceeding a threshold of 0.7 and then rerun the correlation matrix.
   - correlation matrix (before remove)
![8](https://github.com/yuygao/ECS171/assets/112483058/72fbe0cb-880b-48e8-90b0-3c644974f746)
<p align="center">Figure.16  Heatmap of Original Dataset</p>


 - Removal of Highly Correlated Variables:
    - We identify and drop any columns that exhibit correlations greater than 0.7.
    - This step helps us determine which features are most relevant to our problem through correlation analysis.
    - correlation matrix (after remove):
![9](https://github.com/yuygao/ECS171/assets/112483058/cea77673-91fc-40f2-900f-f741f3d811e5)
<p align="center">Figure.17  Heatmap of Dataset aftrer removing Highly Correlated Variables</p>


Data Transformation:
 - Label Encoding for Categorical Variables:
   - We perform label encoding for categorical variables, which includes hardcoding for variables such as property_type, room_type, bed_type, cancellation_policy, and city.
   - We apply label encoding using a label encoder for host_identity_verified, instant_bookable, and cleaning_fee.
 - Combining Dataset with Categorical and Numerical Data
 - Standardization Using StandardScaler:

    - We normalize the data using the StandardScaler() method. 
    - Additionally, we conduct a histogram plot to test for data normality. 
![10](https://github.com/yuygao/ECS171/assets/112483058/fd6e619c-d6bd-4e7e-a786-a1fa36e6a5af)
<p align="center">Figure.18  Histogram of checking data normality by using StandardScaler</p>

```ruby
# Initialize scalers
standard_scaler = StandardScaler()
# Standardize the data
standardized_data = standard_scaler.fit_transform(df_cleaned_new[numerical_columns])
# Create DataFrames for the transformed data
standardized_df = pd.DataFrame(standardized_data, columns=numerical_columns)
print("\nStandardized Data:")
standardized_df.max()
```

 - Normalization Using MinMaxScaler:
    - We normalize the data using the MinMaxScaler() method.
    - A histogram plot is used to test for data normality.
![11](https://github.com/yuygao/ECS171/assets/112483058/917247e9-7085-4b04-9cd6-3084f9bd8c8d)
<p align="center">Figure.19  Histogram of checking data normality by using MinMaxScaler</p>

```ruby
# Initialize scalers
min_max_scaler = MinMaxScaler()
# Normalize the data
normalized_data = min_max_scaler.fit_transform(df_cleaned_new[numerical_columns])
# Create DataFrames for the transformed data
normalized_df = pd.DataFrame(normalized_data, columns=numerical_columns)
# Display the results
print("Normalized Data:")
normalized_df
```


 - Log Transformation of "Price":
    - It's important to note that our target variable, "price," has already been transformed into its natural logarithm (log of price) in the raw dataset we obtained from Kaggle.
    - We retain it in its log-transformed state as this transformation is crucial for our analysis.
 - Merging Data:
    - We merge the data frames containing standardized data, normalized data, and the target variable log_price.

```ruby
# Target variable
log_price = df_cleaned_new['log_price']
# Merge the dataframes for standardized data
standardized_data_merged = pd.concat([categorical_data, standardized_df, log_price], axis=1)
standardized_data_merged
```
```ruby
# Target variable
log_price = df_cleaned_new['log_price']
# Merge the dataframes for normalized data
normalized_data_merged = pd.concat([categorical_data, normalized_df,log_price], axis=1)
normalized_data_merged
```

**Preparing for Model Building**

In this section, we split dataset at an 8:2 ratio with original dataset and describes our models:
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
   1. [Random Forest - as a baseline model](https://github.com/yuygao/ECS171/blob/90c7effd71d58d60df346660196a4efa02fbb344/ECS171_Final_Submission%20.ipynb)
   2. [XGBoost](https://github.com/yuygao/ECS171/blob/90c7effd71d58d60df346660196a4efa02fbb344/ECS171_Final_Submission%20.ipynb)
   3. [LightGBM](https://github.com/yuygao/ECS171/blob/90c7effd71d58d60df346660196a4efa02fbb344/ECS171_Final_Submission%20.ipynb)
   4. [k-NN Algorithm (KNN)](https://github.com/yuygao/ECS171/blob/90c7effd71d58d60df346660196a4efa02fbb344/ECS171_Final_Submission%20.ipynb)



---
## D. Results section
We will focus our attention on four different models: Random Forest, XGBoost, LGBM, and k-NN. Within the scope of these four modeling approaches, we identify seven essential components:

**1. Randomized Hyperparameter Search:**
> > This phase encompasses an extensive exploration of hyperparameter options, ensuring a comprehensive search for optimal settings.

**2. Grid Hyperparameter Search:** 
> > Following the initial exploration, we conduct a targeted search to pinpoint the most effective hyperparameters.

**3. Metric Visualization:** 
> > To evaluate model performance, we employ visualizations of key metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R2) in relation to the number of estimators.

**4. Scatter Plot Analysis:** 
> > Utilizing scatter plots, we visually compare predicted prices against actual prices, offering valuable insights into model accuracy.

**5. Feature Selection Techniques:** 
> > We apply feature selection methods to identify and leverage the most pertinent features, thereby enhancing the predictive capabilities of our models.

**6. Learning Curve Analysis:** 
> > Learning curves help to understand how model performance evolves with increased training data, revealing potential areas for improvement.

**7. Validation Curve Analysis:** 
> > Validation curves play a crucial role in fine-tuning hyperparameters and assessing their impact on overall model performance.

These seven components contribute to a comprehensive evaluation and optimization of our four modeling approaches, ensuring we effectively utilize the full potential of each model while optimizing feature selection to improve results.

---
### Random Forest - as a baseline model
<center>
  <img width="1016" alt="733665364656940689" src="https://github.com/yuygao/ECS171/assets/112483058/ddb6d867-15ad-4516-aefb-d78c6f6fe821">
</center>
<p align="center">Figure.20 Table of RandomizedSearchCV and GridSearchCV</p>

<div style="text-align: center;">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/6f52e04d-e21c-4c21-bc9b-0ebc91b7ab76" alt="Your Image Alt Text">
</div>
<p align="center">Figure.21 Table of Metric Visualization</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/0e98b03a-7742-4dcf-81d9-5f57c857eb38" alt="Image Description" />
</div>
<p align="center">Figure.22 Change in MSE, MAE, RMSE, R2</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/f89cd6ad-beed-46c3-a285-e06da6c97a13" alt="Image Description" />
</div>
<p align="center">Figure.23 Scatter for Predicted vs. Actual Price</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/f9be9247-dba0-493d-a7a9-1b1c607ff2e7" alt="Image Description" />
</div>
<p align="center">Figure.24  Random Forest Feature Importances (Gain)</p>

<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/3b6cbaf4-96b3-40bd-b992-9648f2136eb4" alt="Image Description" />
</div>
<p align="center">Figure.25 Learning Curve</p>

 <div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/c8345537-2121-4aea-a157-3525d1cd23e6)" alt="Image Description" />
</div>
<p align="center">Figure.26 Validation Curve</p>


---
### XGBoost

<center>
  <img width="1016" alt="733665364656940689" src="https://github.com/yuygao/ECS171/assets/112483058/192faa28-e5c6-4430-be02-5933ffbcbc5c">
</center>
<p align="center">Figure.27 Table of RandomizedSearchCV and GridSearchCV</p>

<div style="text-align: center;">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/6d475315-5399-4106-a390-198503416bb0" alt="Your Image Alt Text">
</div>
<p align="center">Figure.28 Table of Metric Visualization</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/16a0b6cd-abc5-4d85-81b6-211ead2829dc" alt="Image Description" />
</div>
<p align="center">Figure.29 Change in MSE, MAE, RMSE, R2</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/56f44763-fcca-425f-b79e-7aea85f26d0f" alt="Image Description" />
</div>
<p align="center">Figure.30 Scatter for Predicted vs. Actual Price</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/b8d537c0-eddb-47a3-b23e-4dbb306b2b85" alt="Image Description" />
</div>
<p align="center">Figure.31  Random Forest Feature Importances (Gain)</p>

<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/a218dca7-3d85-40d0-ac29-136398d2e4c7" alt="Image Description" />
</div>
<p align="center">Figure.32 Learning Curve</p>

 <div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/a0d70425-a1cf-4122-b2fa-788a76354b9e" alt="Image Description" />
</div>
<p align="center">Figure.33 Validation Curve</p>

---
### LightGBM

<center>
  <img width="1016" alt="733665364656940689" src="https://github.com/yuygao/ECS171/assets/112483058/789bdad4-e8f7-47ab-97d4-22decc7f006f">
</center>
<p align="center">Figure.34 Table of RandomizedSearchCV and GridSearchCV</p>

<div style="text-align: center;">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/82daa8e8-ff07-40f1-94ef-d65e996815df" alt="Your Image Alt Text">
</div>
<p align="center">Figure.35 Table of Metric Visualization</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/35432523-9b1d-432e-9f2c-aede5d3f6204" alt="Image Description" />
</div>
<p align="center">Figure.36 Change in MSE, MAE, RMSE, R2</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/64f973dd-edb8-4862-bd15-657dc0230c99" alt="Image Description" />
</div>
<p align="center">Figure.37 Scatter for Predicted vs. Actual Price</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/8ec372fd-fece-48c5-b067-dd0a306bf24b" alt="Image Description" />
</div>
<p align="center">Figure.38  Random Forest Feature Importances (Gain)</p>

<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/32b17496-0220-4120-b735-6bf800e814b1" alt="Image Description" />
</div>
<p align="center">Figure.39 Learning Curve</p>

---
### k-NN Algorithm (KNN)
<center>
  <img width="1016" alt="733665364656940689" src="https://github.com/yuygao/ECS171/assets/112483058/03786470-b442-4cb7-98da-290369359b88">
</center>
<p align="center">Figure.40 Table of RandomizedSearchCV and GridSearchCV</p>

<div style="text-align: center;">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/c21d8d07-4dee-4df5-b33c-efd44e3ef75e" alt="Your Image Alt Text">
</div>
<p align="center">Figure.41 Table of Metric Visualization</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/b56213a1-16a7-4082-8e07-22911c4aefec" alt="Image Description" />
</div>
<p align="center">Figure.42 Change in MSE, MAE, RMSE, R2</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/ce6cff34-0c4d-40d3-89a8-d7fb650b48a3" alt="Image Description" />
</div>
<p align="center">Figure.43 Scatter for Predicted vs. Actual Price</p>


<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/43c6a7fa-195e-435c-b1f0-dd65d94b16e7" alt="Image Description" />
</div>
<p align="center">Figure.44  Random Forest Feature Importances (Gain)</p>

<div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/d2197ea3-c627-434e-aaf9-e5d66e79b132" alt="Image Description" />
</div>
<p align="center">Figure.45 Learning Curve</p>

 <div align="center">
  <img src="https://github.com/yuygao/ECS171/assets/112483058/c22ac14d-d637-4a2d-ab76-e177d04eaebc" alt="Image Description" />
</div>
<p align="center">Figure.46 MSE, R-squared vs. Model Complexity</p>


---
## E. Discussion section

In this discussion, we're going to break down our Airbnb listing price predictive modeling project into different parts. We'll explore the rationale behind our choices, interpret the results, and outline our thought process from start to finish. Furthermore, we'll take a close look at the quality of our results and recognize any areas where we may have fallen short.

**Choice of Models:**

We began our project by selecting four machine learning models: Random Forest, XGBoost, LightGBM (LGBM), and k-NN. Our rationale for this choice was to explore diverse modeling approaches to address the complex nature of predicting Airbnb listing prices. Each of these models had its own distinct advantages, and we're going to break down why we went with them into three important parts.

**1. Why Random Forest as a Baseline Model, Not Decision Tree or Linear Regression:**

We selected for Random Forest as our baseline model over Decision Trees or Linear Regression for several compelling reasons:

- Overfitting Reduction and Higher Predictive Accuracy: Random Forest helps us prevent overfitting, which is essentially when our model fits the training data too closely and struggles to generalize to new, unseen data. This is especially important in the context of predicting Airbnb listing prices, as we're dealing with a lot of variables and complex relationships. Random Forest generally gives us more accurate predictions compared to a single Decision Tree. It does this by combining the predictions from multiple trees, which helps us avoid making predictions that are biased or overly complicated. This is a huge advantage when you consider the factors that can influence Airbnb listing prices.
  
- Non-Linear Flexibility: Random Forest doesn't make assumptions about the data having a linear relationship between its features (like Linear Regression does). This is important because not all real-world datasets, including Airbnb listings, follow these linear patterns. Random Forest's ability to capture nonlinear relationships is a significant advantage in such cases.
  
- Feature Importance: Random Forest could identify the variables with the most significant impact on price predictions. This feature is useful for gaining deeper understanding into the factors that influence listing prices.

**2. Ensemble Models (XGBoost, LGBM, Random Forest):**

We decided to include ensemble models, namely XGBoost, LGBM, and Random Forest, for several compelling reasons:

- Robustness: Ensemble models are known for their robustness. They can handle tricky stuff like complex, not-so-straightforward connections in the data, even when there's noise (data that doesn't quite fit) and outliers (data that's really different). When we're trying to predict Airbnb listing prices, where there's a ton of data going on at once, having models that can handle all this mess is useful.
  
- Feature Importance Insights: Ensemble models could help us identify feature importance. In our case, they show us which parts of our data have the most impact on predicting prices. This helps us make smart decisions based on the data and understand how prices work better.

**3. k-NN Algorithm:**

We also introduced the k-NN algorithm into our model selection for specific purposes:

- Simplicity: k-NN is relatively straightforward to implement and understand. It serves as a starting point for us to figure out how much we can predict just by looking at similar nearby listings.
  
- Interpretability: k-NN offers a high level of interpretability. Predictions are based on the prices of the nearest neighbors, making it easy to explain why a particular price prediction was made.  

In summary, our model selection strategy combines the robustness and predictive power of ensemble models with the simplicity and interpretability of k-NN to comprehensively address the regression tasks of predicting Airbnb listing prices.

---
**Preparing the Dataset for Modeling:**
First, we cleaned up and encoded the data to get it ready for analysis. Then, we took a crucial step by normalizing the data. This was done to ensure that the scales of our features were consistent across all models. We believed this would help our models converge better and boost their predictive accuracy. As a result, we proceeded with the normalized dataset in the following sections, which we labeled as "normalized_data_merged" in the code.

**Hyperparameter Tuning Strategy:**
To fine-tune our models, we adopted a two-step approach. We employed random search for initial exploration of hyperparameters, followed by grid search for fine-tuning. Random search allowed us to efficiently explore a broad range of hyperparameters, while grid search refined the best-performing combinations. We believed this two-step approach balanced efficiency with optimization in our hyperparameter tuning process..

---

**Results and Interpretation:**
In this section, we will discuss the results from each model: Random Forest, XGBoost, LGBM, and KNN. The results of our modeling showed different levels of predictive accuracy for each model. We'll proceed to analyze these results individually.

**1. Random Forest as a baseline model:**

We employed a Random Forest model as our baseline to prevent overfitting and enhance generalization for predicting the 'price' target variable, splitting the dataset into 80% training and 20% testing sets. Utilizing RandomizedSearchCV, we explored a wide range of hyperparameters and identified the best parameters, subsequently fine-tuning them with GridSearchCV to construct our final Random Forest model. We evaluated its performance on the test dataset using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R^2). This evaluation process is critical to assess whether the model has good generalization or shows signs of overfitting (performing exceptionally well on training data but poorly on test data).

Our initial parameter grid spanned n_estimators (ranging from 100 to 1000), max_depth (ranging from 3 to 10), and min_samples_split (ranging from 2 to 10). After Randomized Search, we found the optimal parameters to be 400 estimators, a minimum sample split of 2, and a maximum depth of 8. Grid Search further refined these values to 450 estimators, a minimum sample split of 2, and a maximum depth of 9. 

With these carefully selected parameters, we constructed our final Random Forest model, which exhibited impressive results when tested on the dataset. This included an MSE of 0.170, MAE of 0.317, RMSE of 0.412, and an R^2 of 0.636. On the training set, we observed an MSE of 0.142, MAE of 0.292, RMSE of 0.377, and an R^2 of 0.689. These metrics not only indicated the prevention of overfitting but also demonstrated the model's strong generalization to new data. Additionally, our visualization of the first 10 iterations of MSE, MAE, RMSE, and R^2 showed consistent trends between the training and test metrics. In this case, the testing line exceeded the training line for MSE, MAE, and RMSE, while for R^2, the testing line was below the training line. These observations highlight our model's effective ability to generalize.

Moreover, when we examined the plot comparing predicted prices to actual prices, we noticed that most data points clustered around the residual line. This clustering indicated a strong alignment between our model's predictions and the true prices, a positive sign that our model effectively captured the underlying data patterns. Our learning curve also displayed converging behavior, progressively reducing the gap between the training and test Mean Squared Error (MSE) lines from left to right. This convergence signified that our model was learning from the training data and improving its predictive abilities on unseen data, another positive outcome. 

Finally, in our analysis of feature importance, we observed that the top three significant features affecting price were room type, bathrooms, and city. These observations provide useful information about the factors influencing the 'price' target variable and can guide further model refinement or decision-making.

---
**2. XGBoost:**

We used an XGBoost model to predict the 'price' target variable. We divided the data into an 80% training set and a 20% testing set. We searched for the best model parameters using RandomizedSearchCV and GridSearchCV.

Initially, we explored a range of parameters like the number of estimators (from 100 to 1000), maximum depth (from 3 to 10), and learning rate (from 0.01 to 0.3). After Randomized Search, we found the best parameters to be 121 estimators, a learning rate of 0.195, and a maximum depth of 4. Grid Search further improved these values to 171 estimators, a learning rate of 0.145, and a maximum depth of 4.

With these chosen parameters, we built our final XGBoost model, which performed well when tested on the dataset. We measured its performance using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R^2).

On the testing set, our XGBoost model achieved an MSE of 0.165, MAE of 0.314, RMSE of 0.407, and an R^2 of 0.645. On the training set, we observed an MSE of 0.146, MAE of 0.294, RMSE of 0.382, and an R^2 of 0.680. These metrics indicated that the model didn't overfit and could generalize well to new data.

We also visualized the first 10 iterations of MSE, MAE, RMSE, and R^2, which showed consistent trends between training and test metrics. MSE, MAE, and RMSE were slightly higher on the test set, while R^2 was slightly lower. This suggested the model's ability to generalize effectively.

Furthermore, when we looked at the plot comparing predicted prices to actual prices, we noticed that most data points were close to the residual line. This indicated that our model's predictions aligned well with actual prices, showing that it captured the data patterns effectively. Our learning curve showed that the training and test Mean Squared Error (MSE) lines gradually converged, demonstrating the model's learning and improved predictive abilities on unseen data.  In our analysis of feature importance, we observed that the top three significant features affecting price were room type, bathrooms, and bedrooms. 

In summary, we successfully used an XGBoost model with carefully tuned parameters to avoid overfitting and achieve strong generalization. The model performed excellently on both training and testing data, indicating its effectiveness in capturing the underlying dataset patterns.

---
**3. LightGBM (LGBM):**

We employed a LightGBM model to predict the 'price' target variable, dividing our dataset into an 80% training set and a 20% testing set. To find the best model parameters, we conducted both RandomizedSearchCV and GridSearchCV.

Initially, we explored a range of parameters, including the number of estimators (ranging from 100 to 1000), maximum depth (from 3 to 14), subsample, colsample_bytree (ranging from 0.6 to 1.0 with 5 evenly spaced values), and learning rate (ranging from 0.01 to 0.3 with 30 evenly spaced values). After Randomized Search, we identified the best parameters as follows: 300 estimators, a learning rate of 0.020, a maximum depth of 8, and subsample and colsample_bytree set at 0.60. Subsequently, Grid Search further fine-tuned these values, resulting in 250 estimators, a learning rate of 0.07, a maximum depth of 9, and subsample and colsample_bytree at 0.55.

With these chosen parameters, we constructed our final LightGBM model, which performed admirably during testing. To assess its performance, we utilized metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R^2).

On the testing set, our LightGBM model achieved an MSE of 0.165, MAE of 0.313, RMSE of 0.406, and an R^2 of 0.647. For the training set, we observed an MSE of 0.139, MAE of 0.287, RMSE of 0.373, and an R^2 of 0.695. These metrics indicated that the model avoided overfitting and demonstrated strong generalization to new data.

We also visually analyzed the first 10 iterations of MSE, MAE, RMSE, and R^2, revealing consistent trends between training and test metrics. MSE, MAE, and RMSE were slightly higher on the test set, while R^2 was slightly lower. This suggested that the model's ability to generalize effectively.

Furthermore, when inspecting the plot comparing predicted prices to actual prices, we noticed that most data points closely followed the residual line. This indicated that our model's predictions aligned well with actual prices, showing its effective capture of data patterns. Our learning curve displayed the gradual convergence of the training and test Mean Squared Error (MSE) lines, illustrating the model's learning and improved predictive abilities on unseen data. In our analysis of feature importance, we identified the top three significant features affecting price: the number of reviews, review scores rating, and city.

In summary, we successfully harnessed a LightGBM model with finely-tuned parameters to avoid overfitting and achieve robust generalization. The model performed exceptionally well on both training and testing data, underscoring its efficacy in capturing underlying dataset patterns.


---
**4. K-Nearest Neighbors (k-NN) Algorithm:**

We choose to perform a K-NN algorithm since it’s a non-parametric method and it’s suitable for
regression and predictions. We began by selecting 'price' as our target variable and splitting the dataset into a training set and a testing set (80:20 ratio). Our approach involved finding the best regression model using KNN, fine-tuning its hyperparameters, assessing its performance, and visualizing the results.

To begin, we preprocessed the data, divided it into training and testing sets, and conducted a RandomizedSearchCV to pinpoint the optimal hyperparameters for the KNN regressor. These hyperparameters included the number of neighbors (k), the weights, and the algorithm used. Subsequently, we evaluated the KNN model's performance on the test dataset, providing metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R^2).

During the hyperparameter search, we explored various options, including different weight schemes ('uniform' and 'distance'), the number of neighbors ranging from 1 to 21, and algorithm choices ('auto', 'ball_tree', 'kd_tree', 'brute'). After Randomized Search, we identified the best configuration as using uniform weight, 11 neighbors, and the kd_tree algorithm. Further refinement through Grid Search resulted in uniform weight, 11 neighbors, and the ball_tree algorithm.

When it came to evaluating our KNN model's performance, we discovered that on the testing set, it achieved an MSE of 0.206, MAE of 0.346, RMSE of 0.454, and an R^2 of 0.558. For the training set, the model exhibited an MSE of 0.161, MAE of 0.307, RMSE of 0.402, and an R^2 of 0.647. These metrics provided evidence that the model avoided overfitting and showcased robust generalization to new data.

Additionally, our visual analysis of the initial 10 iterations of MSE, MAE, RMSE, and R^2 revealed consistent patterns between training and test metrics. Although the test set displayed slightly elevated MSE, MAE, and RMSE values, along with a slightly lower R^2, it indicated that the model effectively extended its learning to new data.

Further exploring our model's predictive prowess, the plot comparing predicted prices to actual prices demonstrated a remarkable alignment of data points with the residual line. This underscored the fact that our model's predictions closely mirrored actual prices, highlighting its proficiency in capturing underlying data patterns. Meanwhile, our learning curve demonstrated the gradual convergence of training and test Mean Squared Error (MSE) lines, emphasizing the model's learning process and its enhanced predictive capabilities on unseen data. In our examination of feature importance, we identified the three most influential features affecting price: room type, city, and property type.

---
**Confidence in the Results:**
We have confidence in the results due to the data preparation, hyperparameter fine-tuning, and model selection process. The inclusion of cross-validation and comprehensive evaluation metrics guaranteed a thorough assessment of the models' performance.

**Shortcomings and Future Considerations:**
While our project has provided a wide range of ways to predict Airbnb listing prices, it's essential to recognize its limitations and areas for potential improvement. Firstly, the accuracy of our findings relies on the quality of the data we had access to. To make our predictions even more reliable, we can work on improving how we gather data and make sure it's free from any biases. Secondly, we can consider adding more features or using advanced methods to make our models perform even better. For example, think about adding features like nearby attractions or public transport accessibility. Thirdly, although we tried different models, there are even more advanced techniques like neural networks that we can explore in the future to make our predictions even more accurate. Lastly, we didn't account for time-related changes in our analysis. To understand pricing trends better, we can use time series analysis, which looks at how prices change over time, like when prices tend to be higher or lower during certain seasons. 

---
## F. Conclusion section

After a thorough analysis of our modeling results, it's clear that the LightGBM (LGBM) model stands out as the most effective choice for predicting Airbnb listing prices. While we began with the Random Forest model as a baseline, LGBM consistently outperformed it in predictive accuracy. LGBM consistently achieved lower MSE, RMSE, and higher R^2 values on both training and testing sets. Similarly, although the XGBoost model is powerful, it couldn't match LGBM's predictive power, as LGBM consistently achieved lower MSE and RMSE values, indicating greater pattern-capturing ability. Turning to the K-Nearest Neighbors (KNN) algorithm, LGBM once again excelled in predictive accuracy. With lower MSE, RMSE, and better R^2 values on the testing set, LGBM proved its capacity to generalize effectively. Overall, LGBM consistently outperformed other models, avoiding overfitting and demonstrating robust generalization to new data. Its performance across various evaluation metrics makes it the preferred choice for predicting Airbnb listing prices. In summary, due to its consistent and excellent performance, characterized by lower MSE, RMSE, and higher R^2 values, LightGBM (LGBM) is a reasonable choice for predicting Airbnb listing prices. However, there are areas for improvement and future exploration, such as enhancing data quality, exploring additional features, considering more advanced techniques like neural networks, and incorporating temporal analysis to better understand pricing trends over time. In this way, we could do predictive modeling on the real-time dynamics of Airbnb listings prices to provide more valuable insights for hosts and travelers.

---
## G. Collaboration

<img width="705" alt="Screenshot 2023-09-13 at 8 41 20 PM" src="https://github.com/yuygao/ECS171/assets/114432520/da5688f8-8aa3-4eb9-ab44-c5e8acc4a425">

There are four of us in our group, Yuanyuan Gao, Yonglin Liang, Wenhui Li, and Yayue Song. We all work as a team and everyone is responsible for a different part. Yuanyuan Gao is our group leader, who was mainly responsible for coordinating everyone's time, setting deadlines and the part of building Random Forest, and LightGBM and analyzing them separately. Wenhui Li is our group coding, who is mainly responsible for building XGBoost and KNN models and doing the model analysis. Yonglin Liang is our group coding, who is mainly responsible for data processing  and data visualization. Yayue Song is our group writer, who was mainly responsible for plot formatting and report writing.


## H. Acknowledgments

We would like to thank our TAs, Mayuri and Yifeng, for their guidance and support throughout this project. Additionally, we extend our appreciation to our group mates for their invaluable contributions towards the success of this project.

## I. References
- RudyMizrahi. (2018, March 14). Airbnb listings in major US cities. Kaggle. https://www.kaggle.com/datasets/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml?select=test.csv 









