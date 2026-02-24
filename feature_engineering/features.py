import pandas as pd
import numpy as np


class FeatureEngineering:
    def __init__(self, data):
        self.data = data.copy()

    def create_datetime_features(self):
        datetime_cols = [
            'order_purchase_timestamp', 'order_approved_at',
            'order_delivered_carrier_date', 'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
        for col in datetime_cols:
            self.data[col] = pd.to_datetime(self.data[col])

        self.data['delivery_time'] = (
            self.data['order_delivered_customer_date'] - self.data['order_purchase_timestamp']
        ).dt.total_seconds() / 86400

        self.data['estimated_delivery_time'] = (
            self.data['order_estimated_delivery_date'] - self.data['order_purchase_timestamp']
        ).dt.total_seconds() / 86400

        # Positive = late delivery
        self.data['delivery_vs_estimate'] = (
            self.data['delivery_time'] - self.data['estimated_delivery_time']
        )

        self.data['approval_time'] = (
            self.data['order_approved_at'] - self.data['order_purchase_timestamp']
        ).dt.total_seconds() / 3600

        self.data['carrier_pickup_time'] = (
            self.data['order_delivered_carrier_date'] - self.data['order_approved_at']
        ).dt.total_seconds() / 86400

        self.data['purchase_dayofweek'] = self.data['order_purchase_timestamp'].dt.dayofweek
        self.data['purchase_hour'] = self.data['order_purchase_timestamp'].dt.hour
        self.data['purchase_month'] = self.data['order_purchase_timestamp'].dt.month

        return self.data

    def create_product_features(self):
        self.data['product_volume_cm3'] = (
            self.data['product_length_cm']
            * self.data['product_height_cm']
            * self.data['product_width_cm']
        )
        return self.data

    def create_price_features(self):
        self.data['price_freight_ratio'] = (
            self.data['total_price'] / (self.data['total_freight'] + 1)
        )
        return self.data

    def filter_delivered(self):
        self.data = self.data[self.data['order_status'] == 'delivered'].copy()
        print(f"Filtered to delivered orders: {self.data.shape}")
        return self.data

    def run_all(self):
        self.create_datetime_features()
        self.create_product_features()
        self.create_price_features()
        self.filter_delivered()
        return self.data
