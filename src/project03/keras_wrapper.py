"""Keras Wrapper"""

import numpy as np


class KerasWrapper:
    """"
    Wrapper to make a Keras model behave like a scikit-learn estimator
    """

    def __init__(self, build_fn, epochs=200, batch_size=32):
        """
        build_fn : function that returns a compiled Keras model
        epochs : int, nb of training epochs
        batch_size : int, batch size during training
        """
        self.build_fn = build_fn
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, x, y):
        """
        Fit the Keras model with numpy arrays x and y
        """
        x = np.array(x)
        y = np.array(y)

        # build model only once
        if self.model is None:
            self.model = self.build_fn()

        self.model.fit(
            x, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        return self

    def predict(self, x):
        """
        Predict using the trained Keras model
        Returns a 1D numpy array (like sklearn)
        """
        x = np.array(x)
        return self.model.predict(x).ravel()
