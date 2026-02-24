import pandas as pd


class DataLoader:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.orders = None
        self.items = None
        self.payments = None
        self.reviews = None
        self.products = None
        self.customers = None
        self.sellers = None
        self.categories = None
        self.data = None

    def load_all(self):
        self.orders = pd.read_csv(f'{self.data_path}/olist_orders_dataset.csv')
        self.items = pd.read_csv(f'{self.data_path}/olist_order_items_dataset.csv')
        self.payments = pd.read_csv(f'{self.data_path}/olist_order_payments_dataset.csv')
        self.reviews = pd.read_csv(f'{self.data_path}/olist_order_reviews_dataset.csv')
        self.products = pd.read_csv(f'{self.data_path}/olist_products_dataset.csv')
        self.customers = pd.read_csv(f'{self.data_path}/olist_customers_dataset.csv')
        self.sellers = pd.read_csv(f'{self.data_path}/olist_sellers_dataset.csv')
        self.categories = pd.read_csv(f'{self.data_path}/product_category_name_translation.csv')

        self.products = self.products.merge(self.categories, on='product_category_name', how='left')

        print(f"Loaded {len(self.orders)} orders, {len(self.items)} items, "
              f"{len(self.reviews)} reviews, {len(self.products)} products")
        return self

    def _aggregate_items(self):
        return self.items.groupby('order_id').agg(
            total_items=('order_item_id', 'max'),
            total_price=('price', 'sum'),
            total_freight=('freight_value', 'sum'),
            avg_price=('price', 'mean'),
            n_sellers=('seller_id', 'nunique'),
            n_products=('product_id', 'nunique'),
        ).reset_index()

    def _aggregate_payments(self):
        return self.payments.groupby('order_id').agg(
            total_payment=('payment_value', 'sum'),
            n_installments=('payment_installments', 'max'),
            n_payment_types=('payment_type', 'nunique'),
            main_payment_type=('payment_type', lambda x: x.mode().iloc[0]),
        ).reset_index()

    def _get_product_features(self):
        first_product = self.items.sort_values('price', ascending=False) \
            .drop_duplicates('order_id', keep='first')
        first_product = first_product[['order_id', 'product_id']] \
            .merge(self.products, on='product_id', how='left')
        return first_product[[
            'order_id', 'product_category_name_english',
            'product_weight_g', 'product_length_cm', 'product_height_cm',
            'product_width_cm', 'product_photos_qty',
            'product_name_lenght', 'product_description_lenght'
        ]]

    def merge_all(self):
        items_agg = self._aggregate_items()
        payments_agg = self._aggregate_payments()
        product_features = self._get_product_features()

        df = self.reviews[['order_id', 'review_score']].merge(self.orders, on='order_id', how='inner')
        df = df.merge(self.customers[['customer_id', 'customer_state']], on='customer_id', how='left')
        df = df.merge(items_agg, on='order_id', how='left')
        df = df.merge(payments_agg, on='order_id', how='left')
        df = df.merge(product_features, on='order_id', how='left')

        self.data = df
        print(f"Merged dataset shape: {df.shape}")
        return self.data
