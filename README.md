
# Comparison of deep learning algorithms for forecasting stock returns and portfolio optimization

This repository contains the implementation of our paper, *Comparison of deep learning algorithms for forecasting stock returns and portfolio optimization*, as part of our final project for the course **MAE 576: Deep Learning in Physical Systems**.


## Authors

- [@Sarah Witzmann](https://github.com/sarahwitzman)
- [@Mathieu-Savvas Dimitriades](https://github.com/mattdimi)
## Documentation
The package aims at comparing various machine learning algorithms in forecasting stock returns on the CAC 40 and using these forecasts for portfolio optimization.

The repository can be broken down into three main parts:

### Data modules
These modules are responsible for data preparation and standardization:

- `MyDataLoader.py`: Class to load the data used
- `MyDataset.py`: Dataset class to load data into a `torch.utils.data.Dataset`

### Portfolio optimization modules
These modules are responsible for building the precision matrix of the returns, performing portfolio construction given a set of expected returns, and backtesting the strategies considered:

- `BackTester.py`: Class to run backtests on portfolios
- `PortfolioConstructor.py`: Portfolio construction class. Support regression and classification models.
- `PrecisionMatrixBuilder.py`: Class to compute the precision matrix for a given set of returns and a given set of parameters.

### Forecasting modules
These modules are responsible for defining the deep learning models considered in this study and using these models to generate forecasts:

- `networks.py`: Class that encompasses all the networks considered in this study except for TFT
- `TFT.py`: Module to fine-tune, train and test the Temporal Fusion Transformer being very specific, we choose to dedicate a whole module for it
- `Model.py`: Class to define a model by its name, type (`classification` | `regression`) and class which is the actual trainable model
- `Forecaster.py`: Class to build forecasts on a given dataset given a set of different models.


## Requirements

To be able to run the code in the package and the notebooks, please install the package requirements using the command from your CLI:

`pip install -r requirements.txt`