"""Keras Wrapper"""

import numpy as np


class KerasWrapper:
    """
    :var start_date: Description
    :vartype start_date: first
    :var end_date: Description
    :vartype end_date: last
    """

    def __init__(self, build_fn):
        self.build_fn = build_fn
        self.model = None

    def fit(self, x, y):
        """
        :param self: Description
        :param X: Description
        :param y: Description
        """
        x = np.array(x)
        y = np.array(y)

        self.model = self.build_fn()
        epochs = int(ask_value('Choose epochs', ['50', '200'], '200'))
        self.model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
        return self

    def predict(self, x):
        """
        :param self: Description
        :param x: Description
        """
        x = np.array(x)
        return self.model.predict(x).ravel()
