import pandas as pd
import numpy as np


class Statistics:
    def __init__(self, data):
        self.data = data

    def calculate_mean(self, column):
        return self.data[column].mean()

    def calculate_median(self, column):
        return self.data[column].median()

    def calculate_mode(self, column):
        return self.data[column].mode()[0]

    def calculate_standard_deviation(self, column):
        return self.data[column].std()
    
    def calculate_variance(self, column):
        return self.data[column].var()
    
    def calculate_min(self, column):
        return self.data[column].min()
    
    def calculate_max(self, column):
        return self.data[column].max()
    
    def calculate_quantiles(self, column, quantiles):
        return self.data[column].quantile(quantiles)
    
    def calculate_skewness(self, column):
        return self.data[column].skew()
    
    def calculate_kurtosis(self, column):
        return self.data[column].kurtosis()
    
    def calculate_correlation(self, column1, column2):
        return self.data[column1].corr(self.data[column2])
    
    def describe(self):
        return self.data.describe()
    
