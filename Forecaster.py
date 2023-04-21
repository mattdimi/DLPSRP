import utils
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

class Forecaster:

    def __init__(self, dataset, stock_names, time_index, valid_size, test_size, n_splits = 10):
        self.dataset = dataset
        self.stock_names = stock_names
        self.time_index = time_index
        self.valid_size = valid_size
        self.test_size = test_size
        self.n_split = n_splits

    def get_tscv_splits(self):
        return utils.time_series_cross_validation(self.dataset, self.time_index, self.valid_size, self.test_size, self.n_split)
    
    def get_forecasts(self, time_series_cross_validation, models, stock_names):
        """Method to get, from each model in models, the train/valid/test prediction metrics and the predictions themselves for the validation and test sets.

        Args:
            time_series_cross_validation (_type_): _description_
            models (list[Model]): _description_
            stock_names (list[str]): _description_

        Returns:
            _type_: _description_
        """

        results = {}

        predictions_val = defaultdict(dict)
        predictions_test = defaultdict(dict)

        train_index_full = []
        valid_index_full = []
        test_index_full = []

        for model in models:
            results[model.name] = defaultdict(dict)
            if model.type == "classification":
                results[model.name]["Accuracy"] = defaultdict(dict)
            if model.type == "regression":
                results[model.name]["MSE"] = defaultdict(dict)
                results[model.name]["MAE"] = defaultdict(dict)
        
        for batch in time_series_cross_validation:
            
            train, train_index, valid, valid_index, test, test_index = batch

            train_index_full.append(train_index)
            valid_index_full.append(valid_index)
            test_index_full.append(test_index)
            
            n_stocks = train.shape[1]

            for stock in range(n_stocks):

                stock_name = stock_names[stock]
                
                stock_train = train[:, stock, :]
                stock_valid = valid[:, stock, :]
                stock_test = test[:, stock, :]

                y_reg_train = stock_train[:, 0]
                y_classification_train = stock_train[:, 1]
                X_train = stock_train[:, 2:]
                
                y_reg_valid = stock_valid[:, 0]
                y_classification_valid = stock_valid[:, 1]
                X_valid = stock_valid[:, 2:]

                y_reg_test = stock_test[:, 0]
                y_classification_test = stock_test[:, 1]
                X_test = stock_test[:, 2:]

                scaler_X = StandardScaler()
                scaler_X.fit(X_train)
                X_train = scaler_X.transform(X_train)
                X_valid = scaler_X.transform(X_valid)
                X_test = scaler_X.transform(X_test)
                
                scaler_y = StandardScaler()
                scaler_y.fit(y_reg_train.reshape(-1, 1))
                y_reg_train = scaler_y.transform(y_reg_train.reshape(-1, 1)).reshape(-1)
                y_reg_valid = scaler_y.transform(y_reg_valid.reshape(-1, 1)).reshape(-1)
                y_reg_test = scaler_y.transform(y_reg_test.reshape(-1, 1)).reshape(-1)

                for model in models:

                    cls = model.cls

                    if model.type == "classification":
                        
                        fitted_model = cls.fit(X_train, y_classification_train)

                        train_ = fitted_model.predict(X_train)
                        valid_ = fitted_model.predict(X_valid)
                        test_ = fitted_model.predict(X_test)

                        results[model.name]["Accuracy"][stock_name]['train'] = results[model.name]["Accuracy"][stock_name].get('train', []) + [accuracy_score(y_classification_train, train_)]
                        results[model.name]["Accuracy"][stock_name]['valid'] =  results[model.name]["Accuracy"][stock_name].get('valid', []) + [accuracy_score(y_classification_valid, valid_)]
                        results[model.name]["Accuracy"][stock_name]['test'] = results[model.name]["Accuracy"][stock_name].get('test', []) + [accuracy_score(y_classification_test, test_)]

                    if model.type == "regression":

                        fitted_model = cls.fit(X_train, y_reg_train)

                        train_ = fitted_model.predict(X_train)
                        valid_ = fitted_model.predict(X_valid)
                        test_ = fitted_model.predict(X_test)

                        results[model.name]["MSE"][stock_name]['train'] = results[model.name]["MSE"][stock_name].get('train', []) + [mean_squared_error(y_reg_train, train_)]
                        results[model.name]["MSE"][stock_name]['valid'] =  results[model.name]["MSE"][stock_name].get('valid', []) + [mean_squared_error(y_reg_valid, valid_)]
                        results[model.name]["MSE"][stock_name]['test'] = results[model.name]["MSE"][stock_name].get('test', []) + [mean_squared_error(y_reg_test, test_)]

                        results[model.name]["MAE"][stock_name]['train'] = results[model.name]["MAE"][stock_name].get('train', []) + [mean_absolute_error(y_reg_train, train_)]
                        results[model.name]["MAE"][stock_name]['valid'] =  results[model.name]["MAE"][stock_name].get('valid', []) + [mean_absolute_error(y_reg_valid, valid_)]
                        results[model.name]["MAE"][stock_name]['test'] = results[model.name]["MAE"][stock_name].get('test', []) + [mean_absolute_error(y_reg_test, test_)]
                    
                    predictions_val[model.name][stock_name] = predictions_val[model.name].get(stock_name, []) + [valid_]
                    predictions_test[model.name][stock_name] = predictions_test[model.name].get(stock_name, []) + [test_]

        for model in models:
            for metric in results[model.name]:
                for stock in results[model.name][metric]:
                    for split in results[model.name][metric][stock]:
                        results[model.name][metric][stock][split] = np.mean(results[model.name][metric][stock][split])
                results[model.name][metric] = pd.DataFrame.from_dict(results[model.name][metric])
        
        for model in models:
            for stock in predictions_test[model.name]:
                predictions_test[model.name][stock] = pd.Series(np.concatenate(predictions_test[model.name][stock]), index = np.concatenate(test_index_full))
                predictions_val[model.name][stock] = pd.Series(np.concatenate(predictions_val[model.name][stock]), index = np.concatenate(valid_index_full))
            predictions_test[model.name] = pd.concat(predictions_test[model.name], axis = 1)
            predictions_val[model.name] = pd.concat(predictions_val[model.name], axis = 1)
        
        return results, predictions_val, predictions_test, np.concatenate(train_index_full), np.concatenate(valid_index_full), np.concatenate(test_index_full)