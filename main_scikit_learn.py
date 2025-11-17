import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import joblib


class WindPowerExperiment:
    def __init__(self, csv_path):
        """basic data (path, datasets, model)"""
        self.csv_path = csv_path   # input data path
        self.df = None             # DataFrame
        self.X = None              # feature(s)
        self.Y = None              # target (Power)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self):
        """read csv and columns of interest"""
        dataset = pd.read_csv(self.csv_path)
        self.df = dataset.loc[:, ('Time', 'winddirection_100m',
                                  'windspeed_100m', 'Power')]
        return self

    def basic_stats(self):
        """stats on input data"""
        print("Max windspeed_100m:", self.df['windspeed_100m'].max())
        print("Mean windspeed_100m:", self.df['windspeed_100m'].mean())
        print("\nCorrelazioni:")
        print(self.df.corr(numeric_only=True))
        fig = px.scatter_matrix(self.df, dimensions=[
                                'windspeed_100m', 'winddirection_100m', 'Power'])
        fig.show()

    def prepare_features(self):
        """X (input) eand y (target) are defined"""
        self.X = self.df[['winddirection_100m', 'windspeed_100m']]
        self.Y = self.df['Power']
        return self

    def split(self, test_size=0.2, random_state=42):
        """split into subsets train/test."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )
        return self

    def train_model(self, model=None):
        """Train model with default choice Linear Regression"""
        if model is None:
            model = LinearRegression()
        self.model = model
        self.model.fit(self.X_train, self.y_train)
        return self

    def test_model(self):
        """test model on test subset and report MSE and R^2"""
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print("MSE:", mse)
        print("R2:", r2)
        return {"mse": mse, "r2": r2}

    def save_model(self, path):
        """save model"""
        joblib.dump(self.model, path)


exp = WindPowerExperiment("Location1.csv")
exp.load_data()
exp.basic_stats()
exp.prepare_features()
exp.split(test_size=0.2, random_state=42)
exp.train_model()
metrics = exp.test_model()
exp.save_model("windpower_model_linear.pkl")
