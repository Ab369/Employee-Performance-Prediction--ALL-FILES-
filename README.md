# Employee-Performance Prediction
- Project involves using employee dataset to create ML models and compare the efficiency of various models

### Here are the steps involved
1. Data-Collection
   - Downloading the dataset 'garment-workers-productivity' to train model.

2. Visualising and analysing the data
   - imported libraries pandas,seaborn,numpy,matplotlib
   - read the dataset using Pandas 'read_csv' function
   - Did correlation analysis using correlation matrix made using matplotlib
   - the correlation matrix gives a view of how different variables of data are related to each other
   - Did descriptive analysis of dataset using various pandas functions like- pd.decribe(),pd.head() to get an statistical overview of data
   - Did exploratory data analysis using histograms and graphs

3. Data preprocessing
   - Checked for Null values and drop column with null values
   - Converted Date column to datetime format for it to be available for numerical operations, then extracted month from date and created a new 'month' column
   - Merged duplicates in 'department' columns
   - Converted categorical features to numerical features using 'MultiColumnLabelEncoder'
   - Splitting Data to Train and Test set using 'train_test_split()' function from sklearn.
  
4. Model Bulding
   1. Linear Regression Model
       - imported LinearRegression Library from sklearn
       - named model as 'model_lr'
       - provinding training data to model for training and creating model
       - prediciting test data on trained model
       - comparing predicted output with original output using various ways like 'Mean Square Error','Mean absolute Error'
       - getting r2_score for model to get its efficiency
       -Linear Regression Model results

|         MSE         | MAE                 | r2_score(%)       |
|:-------------------:|---------------------|-------------------|
| 0.02097307724687107 | 0.10639164268443942 | 29.06317166092659 |

  2. Random Forest Model
     - trained rf_model using same training data as linear regression
     - Model results-
  
|         MSE         | MAE                 | r2_score(%)        |
|:-------------------:|---------------------|--------------------|
| 0.01533086697337593 | 0.08530114608414956 | 48.146708946023466 |

  2. XG boost Model
     - trained xg_model using same training data as linear regression
     - Model results-
  
|         MSE         | MAE                | r2_score(%)       |
|:-------------------:|--------------------|-------------------|
| 0.01382754921888833 | 0.0760791711050422 | 53.23135113915121 |

- Hence by comparision XG boost model performs best


5. Application Buiding
