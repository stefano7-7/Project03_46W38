"""wind power forecasting using machine learning models"""

# import libraries at module level
from .utilities import (ask_value,
                        choose_time_window,
                        choose_from_list,
                        prediction_metrics)
from .keras_wrapper import KerasWrapper
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# 1: no INFO, 2: no INFO+WARNING, 3: solo ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.makedirs("outputs", exist_ok=True)


class WindForecast():
    """
    Class for wind power forecasting using machine learning models

    ====Parameters====
    df : pandas dataframe for input dataset (timeseries)
    x_train, x_test : model inputs from dataset split
    x_time_test : timestamps of model test inputs from dataset split
    x_scaler = scaled input
    y_train, y_test : model outputs from dataset split
    model : object, machine learning model

    ====Methods=======
    load_data(self)
        asks User to choose site from available list of site datasets
        read and correct data
    split(self)
        split time-series dataset 80/20 into train/test
    train_ML_model(self)
        asks User for model from list (SVR, NN, NN_Keras)
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
        self.x_time_test = None
        self.y_shifted_train = None
        self.y_shifted_test = None
        self.y_original_test = None
        self.x_scaler = None

    def load_data(self, chosen_site_number):
        """
        imports input data file (inputs/LocationN.csv)
        of the site chosen by User and passed as input to the function
        prints data for visual check
        corrects data format (datetime and Celsius)
        NOTE: this is specific for the given files in input/ folder
        """
        input_file_name = 'inputs/Location' + chosen_site_number + '.csv'
        print(input_file_name)
        # import data: read csv & print to check imported data
        df = pd.read_csv(input_file_name)
        print(df.describe(include='all'))

        # data shall be converted & reprinted to check
        df['Time'] = pd.to_datetime(df['Time'])
        df['temperature_2m'] = (df['temperature_2m']-32)*5/9
        print('====== DATA CONVERTED ========')
        print(df.describe(include='all'))
        print('Timestamps into datetime format')
        print('Temperature from Fahrenheit to Celsius')
        self.df = df
        return self

    def plot_timeseries(self):
        """
        plot of time series of variables chosen by User
        with a time window chosen by the User
        """
        # time window
        start_time_dataset = self.df['Time'].min()
        end_time_dataset = self.df['Time'].max()

        start_time_to_plot, end_time_to_plot = choose_time_window(
            start_time_dataset, end_time_dataset
        )
        # create a mask with dt.date()
        mask_selected_time_window = self.df['Time'].dt.date.between(
            start_time_to_plot, end_time_to_plot)
        selected_time_window = self.df['Time'].loc[mask_selected_time_window]

        # variable to plot
        list_of_variables = self.df.columns.tolist()
        list_of_variables.remove('Time')
        chosen_variables = choose_from_list(list_of_variables
                                            )
        fig, ax = plt.subplots()
        for variable in chosen_variables:
            ax.plot(selected_time_window,
                    self.df[variable].loc[mask_selected_time_window], label=variable)
            ax.legend()
        ax.set_title('Time histories')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

    def split(self):
        """splitting the dataset 80-20 train-test"""
        # can be upgraded adding df['temperature_2m']

        # working copy of the dataset
        # for shifting, scaling and dropping NaN
        df2 = self.df.copy()

        # time t+1h
        df2['Power_future'] = df2['Power'].shift(-1)
        df2['Time_future'] = df2['Time'].shift(-1)
        df2 = df2.dropna(subset=['Power_future', 'Time_future'])

        # I will train the model on the relation between
        # x-variable windspeed_100m at time t and
        # power in the future at time t+1h
        # now on the same line in the df2
        x = df2[['windspeed_100m']]
        y = df2['Power_future']

        # to plot the predictions
        x_time = df2['Time_future']  # for the plotting

        # splitting 80/20 train/test
        split_index = int(0.8 * len(df2))
        x_train_raw = x.iloc[:split_index]
        x_test_raw = x.iloc[split_index:]

        # scaling of x
        # the same fit of train data is applied to test data
        # x_scaled = (x_raw + mean)/std_dev
        self.x_scaler = StandardScaler()
        self.x_train = self.x_scaler.fit_transform(x_train_raw)
        self.x_test = self.x_scaler.transform(x_test_raw)

        # time axis for plotting
        self.x_time_test = x_time.iloc[split_index:]

        # y - power
        self.y_shifted_train = y.iloc[:split_index]  # P(t+1) train
        self.y_shifted_test = y.iloc[split_index:]  # P(t+1) test
        self.y_original_test = df2['Power'].iloc[split_index:]  # P(t) test

        # check extremes of train & test datasets
        print("TRAIN Power_future min/max:",
              self.y_shifted_train.min(), self.y_shifted_train.max())
        print("TEST  Power_future min/max:",
              self.y_shifted_test.min(), self.y_shifted_test.max())

        return self

    def train_ml_model(self):
        """
        User's choice of ML model to train (self.model) 
        with support function ask_value()
        choice from a list of ML models supporting .fit() method
        training of the model with self.model.fit() method
        """
        # medel choice
        model_name = ask_value("Choose ML model to train",
                               ['SVR', 'NN', 'NN_Keras'], 'NN')
        if model_name == 'SVR':
            # C: higher values fit data more closely
            #    but increase the risk of overfitting
            # epsilon: error smaller than epsilon are ignored
            # kernel RBF maps input into a non-linear feature space
            c = int(ask_value('Choose C', ['1', '10'], '10'))
            epsilon = float(ask_value('Choose epsilon (allowed deviations)',
                                      ['0.01', '0.1'], '0.01'))
            self.model = SVR(kernel='rbf', C=c, epsilon=epsilon)

        if model_name == 'NN':
            # hidden_layer_sizes: nb neurons hidden layer
            #                     more increase comput. time & overfitting risk
            # activation=ReLU: for non-linearity
            # max_iter: max nb number of epochs (training iterations)
            #           higher helps convergence but increase compuy. time
            hidden_layer_sizes = int(ask_value('Choose nb neurons per layer',
                                               ['100', '500'], '100'))
            max_iter = int(
                ask_value('Choose max iterations', ['500', '1000'], '500'))
            self.model = MLPRegressor(
                hidden_layer_sizes=(hidden_layer_sizes, hidden_layer_sizes),
                activation='relu', max_iter=max_iter)

        if model_name == "NN_Keras":
            epochs = int(ask_value('Choose epochs', ['50', '200'], '50'))

            def build_model():
                # first Dense(64, activation='relu')
                #               64 neurons with ReLU to learn
                # non-linear combinations of the input features
                # second Dense(64, activation='relu'):
                #               second hidden layer
                # third Dense(1, activation='sigmoid')
                #               output layer sigmoid
                #               forces predicted power
                #               into the [0, 1] range
                # optimizer='adam' adaptive gradient method
                # loss=MSE penalizes more large prediction errors
                model = Sequential([
                    Input(shape=(self.x_train.shape[1],)),
                    Dense(64, activation='relu'),
                    Dense(64, activation='relu'),
                    # forcing output in (0,1)
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='mse')
                return model
            self.model = KerasWrapper(build_model, epochs=epochs, batch_size=3)

        # method returning trained model
        self.model.fit(self.x_train, self.y_shifted_train)
        return self

    def test_ml_model(self):
        """
        tests the model over test dataset
        calculates metrics using support function prediction_metrics()
        plots predicted against true data & against persistence model
        takes into account first NaN with persistence
        """
        # apply ML model to scaled test dataset
        x_test_scaled = self.x_test

        # 1-hour ahead prediction with trained model of choice
        y_predicted = self.model.predict(x_test_scaled)

        # persistence 1-hour ahead: P_{t+1}^pers = P_t
        y_persistence = self.y_original_test

        unit = '% Prated'
        print("metrics of persistence model (1-hour ahead)")
        prediction_metrics(self.y_shifted_test, y_persistence, unit)

        print("metrics of ML model (1-hour ahead)")
        prediction_metrics(self.y_shifted_test, y_predicted, unit)

        # plot ML vs test vs persistence
        model_name = self.model.__class__.__name__
        fig, ax = plt.subplots()
        ax.plot(self.x_time_test, self.y_shifted_test,
                label="true power (t+1)", marker='x')
        ax.plot(self.x_time_test, y_persistence,
                label="persistence (P_t)")
        ax.plot(self.x_time_test, y_predicted,
                label=f"{model_name} model")

        ax.legend()
        ax.set_ylabel("Power [% Prated]")
        ax.set_title(f'testing of ML model {model_name} (1-hour ahead)')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

        fig.savefig(f'outputs/testing of ML model {model_name}.png',
                    format='png', dpi=200, bbox_inches='tight')
