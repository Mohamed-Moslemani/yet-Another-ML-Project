import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import BinaryEncoder
class Encoding:
    def __init__(self, data):
        self.data = data
    
    def label_encoding(self, column):
        le = LabelEncoder()
        self.data[column] = le.fit_transform(self.data[column])
        return self.data

    def one_hot_encoding(self, column):
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        encoded = ohe.fit_transform(self.data[[column]])
        encoded_df = pd.DataFrame(
            encoded, columns=ohe.get_feature_names_out([column]), index=self.data.index
        )
        self.data = pd.concat([self.data.drop(column, axis=1), encoded_df], axis=1)
        return self.data
    
    def ordinal_encoding(self, column, categories):
        oe = OrdinalEncoder(categories=[categories])
        self.data[column] = oe.fit_transform(self.data[[column]])
        return self.data
    
    
    def binary_encoding(self, column):
        be = BinaryEncoder(cols=[column])
        self.data = be.fit_transform(self.data)
        return self.data
    
    def frequency_encoding(self, column):
        freq = self.data[column].value_counts() / len(self.data)
        self.data[column] = self.data[column].map(freq)
        return self.data