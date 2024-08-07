# Overview

This repository contains a comprehensive project that analyzes the impact of natural disasters on the S&P 500 stock market between 2010 and 2020. The project involves data merging, preprocessing, normalization, and the development of a predictive model using LSTM networks.

## Datasets
3 datasets are merged together

First is S&P 500 Dataset
	Description: Contains data on 496 companies from the S&P 500 index.
  Columns: 45 columns including disaster type, disaster group, Origin, etc.
	Date Range: 2010 to 2020.

 Second The dataset  of how each stock represented by the symbol did on the market. 7 Columns include
 High, Low, Volume; merged with the date with the first dataset.
 
 Third dataset is made of 16 columns which consist of  the stock market price, weight, sector of the company, indusrty etc. It is then merged woth the 2 datasets via the column symbol.
 
 
### Natural Disasters Dataset
  Description: Contains data on various natural disasters globally.
	Columns: Includes date, type of disaster, and other relevant details.
  Date Range: 1900 to 2023.

## Preprocessing: 
Start Date, Start Month, and Start Year are combined to form a single date column.
Additional Stock Data
•	Description: Contains date, symbol, and stock-specific columns.
•	Merge Key: Symbol.

## Data Merging
1.	Merging S&P 500 and Natural Disasters Dataset:
o	Merged on the Date column.
o	The combined dataset is filtered to include only data from 2010 to 2020.
2.	Merging with Additional Stock Data:
o	Merged on the Symbol column.
o	Ensures all three datasets are combined for analysis.


## Data Analysis
Percentage of Sectors in the Stock Market
sector_counts = added_data['Sector'].value_counts()
 Output
 Technology: 14.3%
 Financial Services: 14.2%
 Industrial: 14%
 Healthcare: 12.9%
 Consumer Cyclical: 10.8%
 Consumer Defensive: 7.9%
Natural Disaster Types
•	Categories: Wildfire, Landslide, Extreme Temperature, Earthquake, Flood, Storm, etc.


## Data Normalization
Natural Disasters Data
•	Categorical data (disaster types) converted to boolean columns.
•	Normalization applied using MinMaxScaler with a range of 0 to 1.
•	Outliers are addressed using percentiles.
Stock Data
•	Features are not normalized except the label (Close column), which is normalized.


## Model Training
Train-Validation-Test Split
•	The dataset is split into training, validation, and test sets.
•	Features and labels are separated accordingly.
.       Label selected is Close, and the features selected were natural disaster types and the stock features excluding Close.


## Model Architecture

Imports: The necessary components from TensorFlow Keras are imported. Sequential is a linear stack of layers, LSTM is a type of recurrent neural network (RNN) layer suited for sequence prediction, Dropout is a regularization layer to prevent overfitting, and Dense is a fully connected layer.

### Sequential Model:
An instance of the Sequential model is created, which allows layers to be added one after another.

•  LSTM Layer 1: The first layer is an LSTM (Long Short-Term Memory) layer with 64 units. return_sequences=True means this layer returns the full sequence of outputs for each input sequence, which is required when stacking LSTM layers.

•  Input Shape: input_shape=(1, X_train.shape[1]) specifies the shape of the input data. Here, the input has a sequence length of 1 and a number of features equal to X_train.shape[1]

### Dropout Layer:
A Dropout layer with a rate of 0.1 is added. This means 10% of the neurons will be randomly set to zero during training to prevent overfitting.

LSTM Layer 2: A second LSTM layer with 64 units is added. Since return_sequences is not specified, it defaults to False, meaning this layer returns only the last output of the input sequence.

Dense Layer: A Dense layer with a single unit is added. This layer serves as the output layer, typically used for regression tasks (predicting a continuous value).
python


## Early Stopping

1. Clone the Repository:
git clone <repository-url>
cd <repository-directory>
2.Install Dependencies:

pip install -r requirements.txt
3.Run Preprocessing:
Execute the script to preprocess and merge datasets.

4.Train the Model:
Run the training script to train the LSTM model.

5.Evaluate the Model:
Use the validation and test sets to evaluate the model's performance.
•	Implemented to prevent overfitting during training.



## Conclusion

This project provides insights into the correlation between natural disasters and stock market performance, specifically focusing on the S&P 500 index. The use of LSTM networks allows for time-series prediction, aiding in understanding and potentially forecasting the impact of natural disasters on stock prices.

