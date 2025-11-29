"""wind power forecasting using machine learning models"""

# import libraries at module level
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


class WindForecast():
    """
    Class for wind power forecasting using machine learning models

    ====Parameters====
    df : pandas dataframe for input dataset (timeseries)
    x_train, x_test : model input from dataset split
    y_train, y_test : imodel output from dataset split
    model : object, machine learning model

    ====Methods=======
    load_data(self)
        asks User to choose site from available list of site datasets
        read and correct data
    split(self)
        split time-series dataset 80/20 into train/test
    train_ML_model(self)
        asks User for model from list (SVR, NN)
        uses .fit() from scikit_learn to train the chosen model
    test_ML_model(self)
        uses .predict() from scikit_learn
        gives metrics (MSE, RMSE, MAe)
        & compare with persistence model & test data
    """

    def __init__(self):
        self.df = None
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        ask user which site for forecasting
        import from corresponding file
        print and check data
        correct data format (datetime and Celsius)
        """
        site_number = ask_value("Choose site number", [
                                "1", "2", "3", "4"], "1")

        input_file_name = 'inputs/Location' + site_number + '.csv'
        print(input_file_name)
        # import data: read csv & print to check imported data
        df = pd.read_csv(input_file_name)
        print(df.describe(include='all'))

        # data shall be converted & reprinted to check
        df['Time'] = pd.to_datetime(df['Time'])
        df['temperature_2m'] = (df['temperature_2m']-32)*5/9
        print('====== DATA CONVERTED ========')
        print(df.describe(include='all'))
        print('timestamps into datetime format')
        print('temperature from Fahrenheit to Celsius')
        self.df = df
        return self

    def split(self):
        """splitting the dataset 80-20 train-test"""
        # can be upgraded adding x = df['temperature_2m']
        x = self.df[['windspeed_100m']]
        y = self.df[['Power']]
        split_index = int(0.8 * len(self.df))
        # x dataset has to be a 2D dataframe for ML models
        self.x_train, self.x_test = x.iloc[:split_index], x.iloc[split_index:]
        self.y_train, self.y_test = y.iloc[:split_index], y.iloc[split_index:]

        return self

    def train_ml_model(self):
        """
        User's choice of ML model to train (self.model)
        choice from a list of ML models supporting .fit() method
        training of the model with self.model.fit() method
        """
        # medel choice
        model_name = ask_value("Choose ML model to train",
                               ['SVR', 'NN'], 'NN')
        if model_name == 'SVR':
            self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            # kernel="rbf" → ........
            # C=1.0 → .......
            # epsilon=0.1 → allowed deviation
        if model_name == 'NN':
            self.model = MLPRegressor(
                hidden_layer_sizes=100, activation='relu', max_iter=500)
            # hidden_layer_sizes=(100,) → 1 layer with 100 neurons
            # activation="relu" → ........
            # max_iter=500 → nb of iterations for convergence

        # method returning trained model
        self.model.fit(self.x_train, self.y_train)
        return self

    def test_ml_model(self):
        """
        test the model over test dataset
        metrics: MSE (emphasis on large error), RMSE (MSE in same units), MAE (outliers)
        plot predicted against true data & against persistence model
        persistence only works for test data consecutive to train time series"""

        # apply ML model to test dataset
        y_predicted = self.model.predict(self.x_test)

        # persistence model 1-hour on test time series"
        y_persistence = self.y_test.shift(1)

        # metrics of ML model
        mse = mean_squared_error(self.y_test, y_predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_predicted)
        print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, ")

        # plot ML model predictions vs test data vs persistence
        fig, ax = plt.subplots()
        ax.plot(self.x_test, self.y_test, label="test data")
        ax.plot(self.x_test, y_persistence, label="persistence model")
        ax.plot(self.x_test, y_predicted, label="tested ML model")
        ax.legend()
        ax.set_ylabel("Power [% Prated]")
        ax.set_title('testing of ML model')
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        fig.savefig('testing of ML model.png', format='png',
                    dpi=200, bbox_inches='tight')


# =============SUPPORT FUNCTIONS===================

def ask_value(prompt, choices, default):
    """ask User for a value from a list of choices"""
    text = input(f"{prompt} {choices} (default = {default}): ")
    if text == "":  # user presses ENTER
        return default
    if text in choices:
        return text
    print("value not valid")
    return ask_value(prompt, choices, default)


# =============MAIN SCRIPT==========================

prediction = WindForecast()
prediction.load_data()
prediction.split()
prediction.train_ml_model()
prediction.test_ml_model()
