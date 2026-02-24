

class Nulls: 
    def __init__(self, data):
        self.data = data
    
    def count_nulls(self):
        return self.data.isnull().sum()
    
    def percentage_nulls(self):
        return (self.data.isnull().sum() / len(self.data)) * 100
    
    def drop_nulls(self):
        return self.data.dropna()
    
    def impute_nulls(self, column, method='mean'):
        if method == 'mean':
            self.data[column].fillna(self.data[column].mean(), inplace=True)
        elif method == 'median':
            self.data[column].fillna(self.data[column].median(), inplace=True)
        elif method == 'mode':
            self.data[column].fillna(self.data[column].mode()[0], inplace=True)
        else:
            raise ValueError("Method must be 'mean', 'median', or 'mode'")
        
    def fill_nulls_with_value(self, column, value):
        self.data[column].fillna(value, inplace=True)

    