# Stat-Pat-Rec-Final

The aim of this project was to create two different probabilistic models (linear regression vs. ARIMA) and compare their ability to forecast the 3-month treasury bill yield for the following month. 

You can find the final report attached to this project as [INSERT NAME HERE]. 

The code within this repository was used to build the models, test them on the dataset, and then create the figures in the report. Here is a brief explanation of each file: 

- Full-Data.xlsx: Contains the original dataset from the FRED (Federal Reserve Economic Data) website with the historical 3-month treasury bill data along with other econpmic indicators.
- regr_viz_utils.py: Defines a series of functions for plotting figured. Most important to this project though, it defines function load_dataset which reads in data from Ful-Data.xlsx, standardizes it, and returns it divided into training, test, and validation sets.
- FeatureTransformPolynomial.py: Defines class PolynomialFeatureTransform which applies a polynomial transform of the inputted dimension to a given model.
- LinearRegressionMAPEstimator.py: Defines class LinearRegressionMAPEstimator which includes functions to support training, predicting, extracting and setting parameters, and calculating score along with variance.
- Baseline_Method.py: Uses aforementioned classes and functions to train, predict, evaluate scores, and graph performance with linear regression MAP estimation.
- Improvement.py: Uses aforementioned classes to replicate training, predicting, and scoring from Baseline_Method.py but instead using ARIMA model. 

Note that Baseline_Method.py and Improvement.py both contain “ADD-ON”s which can be used by uncommenting certain code and commenting out other parts to yield other interesting graphs, and examine hyperparameter settings for example.
