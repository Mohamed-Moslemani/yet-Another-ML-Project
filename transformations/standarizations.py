from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

class Standarizations: 
    def __init__(self, data):
        self.data = data
    
    def standardize(self, column):
        scaler = StandardScaler()
        self.data[column] = scaler.fit_transform(self.data[[column]])
        return self.data
    
    def min_max_scale(self, column):
        scaler = MinMaxScaler()
        self.data[column] = scaler.fit_transform(self.data[[column]])
        return self.data
    
    def robust_scale(self, column):
        scaler = RobustScaler()
        self.data[column] = scaler.fit_transform(self.data[[column]])
        return self.data
