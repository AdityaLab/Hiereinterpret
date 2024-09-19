import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAGen:
    def __init__(self, Y: pd.Series, exog: pd.DataFrame, order: tuple, trend: str):
        self.Y = Y
        self.exog = exog
        self.order = order
        self.trend = trend

    def fit(self):
        model = ARIMA(self.Y, order=self.order, exog=self.exog, trend=self.trend)
        self.model = model.fit()
