
# Comparison of deep learning algorithms for forecasting stock returns and portfolio optimization

This repository contains the implementation of our paper, *Comparison of deep learning algorithms for forecasting stock returns and portfolio optimization*, as part of our final project for the course **MAE 576: Deep Learning in Physical Systems**.

## Authors

- [@Sarah Witzman](https://github.com/sarahwitzman)
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

## Requirements and installation

To be able to run the code in the package and the notebooks, please install the package requirements using the command from your CLI:

`pip install -r requirements.txt`

To run an example of regression and classification models we implemented, you can run the `run.py` file using:

`python run.py`

You should be able to see the advancement bar of the model being trained once the code is ran. The output is the model summary and performance plots.

For further reference, extensive tests have been conducted in the Jupyter notebooks: `CNN_ModelTesting.ipynb`, `LinearModels.ipynb` and `ModelTesting.ipynb`.

## Results
We obtain the following results for each model type considered (only best model included):

|                  | Yearly returns | Yearly volatility | Yearly Sharpe ratio |
|------------------|---------------|------------------|---------------------|
| CAC40            | 13.2%         | 18.9%            | 0.7                 |

| Model    | Yearly returns | Yearly volatility | Yearly Sharpe ratio | Total mean-squared error |
|----------|----------------|-------------------|---------------------|-------------------------|
| LASSO    | -92.2%         | 72.7%             | -1.27               | 0.46                    |
| MLP      | -2.9%          | 3.3%              | -0.87               | 0.28                    |
| **CNN**  | **404.1%**     | **414.5%**        | **0.97**            | **25.94**               |
| LSTM     | 32.8%          | 140.8%            | 0.23                | 9.95                    |
| GRU      | -37.5%         | 125.7%            | -0.29               | 13.33                   |
| TFT      | -32.3%         | 10.0%             | -3.2                | 0.39                    |

| Model          | Yearly returns | Yearly volatility | Yearly Sharpe ratio | Misclassification rate |
|----------------|----------------|-------------------|---------------------|-----------------------|
| Logistic Reg   | -182.3%        | 13.4%             | -13.6               | 50.4%                 |
| MLP            | 1.6%           | 12.1%             | -0.13               | 48.5%                 |
| **CNN**        | **64.4%**      | **66.1%**         | **0.97**            | **47.2%**             |
| LSTM           | -376.3%        | 14.7%             | -25.5               | 52.8%                 |
| GRU            | -399.9%        | 14.4%             | -27.8               | 49.0%                 |
| TFT            | 0.5%           | 9.3%              | 0.0055              | 49.1%                 |

## References
1. Huang, J., Chai, J., Cho, S. (2020). "Deep learning in finance and banking: A literature review and classification." 
2. Jean Dessain (2022). "Machine learning models predicting returns: Why most popular performance metrics are misleading and proposal for an efficient metric." Expert Systems with Applications, Volume 199.
3. Hum Nath Bhandari, Binod Rimal, Nawa Raj Pokhrel, Ramchandra Rimal, Keshab R. Dahal, Rajendra K.C. Khatri (2022). "Predicting stock market index using LSTM." Machine Learning with Applications, Volume 9.
4. Sarker, I.H. (2021). "Deep Learning: A Comprehensive Overview on Techniques, Taxonomy, Applications and Research Directions." SN COMPUT. SCI. 2, 420.
5. Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function." Mathematics of control, signals and systems, 2(4), 303–314.
6. Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016). "Deep Learning." MIT Press, p. 326.
7. Serkan Kiranyaz, Onur Avci, Osama Abdeljaber, Turker Ince, Moncef Gabbouj, Daniel J. Inman (2019). "1D Convolutional Neural Networks and Applications: A Survey."
8. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017). "Attention Is All You Need."
9. George Bird, Maxim E. Polivoda (2021). "Backpropagation Through Time For Networks With Long-Term Dependencies."
10. Bengio, Simard, and Frasconi (1994). "Learning long-term dependencies with gradient descent is difficult." IEEE transactions on neural networks, 5(2), 157-166.
11. Hochreiter and Schmidhuber (1997). "Long short-term memory." Neural computation, 9(8), 1735–1780.
12. Cho, Van Merriënboer, Bahdanau, and Bengio (2014). "On the properties of neural machine translation: encoder-decoder approaches."
13. Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (2016). "Layer Normalization."
14. Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister (2020). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting."
15. Harry Markowitz (1952). "Portfolio Selection." The Journal of Finance, Vol. 7, No. 1, pp. 77-91.
16. Lasse Heje Pedersen, Abhilash Babu, Ari Levine (2021). "Enhanced Portfolio Optimization." Financial Analysts Journal, 77(2), 124-151.
17. Jerome Friedman, Trevor Hastie, Robert Tibshirani (2007). "Sparse inverse covariance estimation with the graphical lasso."
18. Hendrik Bessembinder (2003). "Trade Execution Costs and Market Quality after Decimalization." The Journal of Financial and Quantitative Analysis, Vol. 38, No. 4, pp. 747-777.
19. Bell, F., Smyl, S. (2018). "Forecasting at Uber: An introduction." Accessed on 2023-04-23
20. [Optuna](https://optuna.org/)
21. Wei Bao, Jun Yue, Yulei Rao (2017). "A deep learning framework for financial time series using stacked autoencoders and long-short term memory."
22. Jingyi Shen, M. Omair Shafiq (2020). "Short-term stock market price trend prediction using a comprehensive deep learning system."
23. Ryo Akita, Akira Yoshihara, Takashi Matsubara, Kuniaki Uehara (2016). "Deep learning for stock prediction using numerical and textual information."
24. M. Nabipour, P. Nayyeri, H. Jabani, A. Mosavi, E. Salwana, S. S. (2020). "Deep Learning for Stock Market Prediction." Entropy (Basel), vol. 22, no. 8, p. 840.
25. Mukherjee, S., et al. (2023). "Stock market prediction using deep learning algorithms." CAAI Trans. Intell. Technol. 8(1), 82–94.
26. B L, S., B R, S. (2023). "Combined deep learning classifiers for stock market prediction: integrating stock price and news sentiments." Kybernetes, Vol. 52, No. 3, pp. 748-773.
27. K. Rekha, M. Sabu (2022). "A cooperative deep learning model for stock market prediction using deep autoencoder and sentiment analysis." PeerJ Comput Sci, vol. 8, p. e1158.
28. Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (2016). "Layer Normalization."