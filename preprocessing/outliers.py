

class Outliers: 
    def __init__(self, data):
        self.data = data
    
    def detect_outliers(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
    
    def remove_outliers(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
    
    def impute_outliers(self, column, method='mean'):
        outliers = self.detect_outliers(column)
        if method == 'mean':
            imputed_value = self.data[column].mean()
        elif method == 'median':
            imputed_value = self.data[column].median()
        elif method == 'mode':
            imputed_value = self.data[column].mode()[0]
        else:
            raise ValueError("Method must be 'mean', 'median', or 'mode'")
        
        self.data.loc[outliers.index, column] = imputed_value
        return self.data