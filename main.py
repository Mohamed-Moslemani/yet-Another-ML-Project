import warnings
import numpy as np
import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler
from Splitting.tr_ts_split import Tr_Ts_Split
from dataLoading.data_loader import DataLoader
from feature_engineering.features import FeatureEngineering
from preprocessing.nulls import Nulls
from Encoding.Encoding import Encoding
from modeling.trainer import Trainer
from evaluation.evaluator import Evaluator

warnings.filterwarnings('ignore')


# MLflow Setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("review_score_prediction")


# 1. Load & Merge
print("=" * 70)
print("STEP 1: Loading and merging data")
print("=" * 70)

loader = DataLoader(data_path='data')
loader.load_all()
df = loader.merge_all()


#   2. Feature Engineering  
print("\n" + "=" * 70)
print("STEP 2: Feature engineering")
print("=" * 70)

fe = FeatureEngineering(df)
df = fe.run_all()

print(f"\nTarget distribution (review_score):")
print(df['review_score'].value_counts().sort_index())


#   3. Preprocessing  
print("\n" + "=" * 70)
print("STEP 3: Preprocessing")
print("=" * 70)

FEATURE_COLS = [
    'delivery_time', 'estimated_delivery_time', 'delivery_vs_estimate',
    'approval_time', 'carrier_pickup_time',
    'total_items', 'total_price', 'total_freight', 'avg_price',
    'n_sellers', 'n_products',
    'total_payment', 'n_installments', 'n_payment_types',
    'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
    'product_photos_qty', 'product_name_lenght', 'product_description_lenght',
    'product_volume_cm3', 'price_freight_ratio',
    'purchase_dayofweek', 'purchase_hour', 'purchase_month',
    'customer_state', 'main_payment_type', 'product_category_name_english'
]

model_df = df[FEATURE_COLS + ['review_score']].copy()

# Handle nulls
nulls_handler = Nulls(model_df)
print("Null percentages before handling:")
nulls_pct = nulls_handler.percentage_nulls()
print(nulls_pct[nulls_pct > 0])

numeric_cols = model_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if model_df[col].isnull().any():
        nulls_handler.impute_nulls(col, method='median')

categorical_cols = ['customer_state', 'main_payment_type', 'product_category_name_english']
for col in categorical_cols:
    if model_df[col].isnull().any():
        nulls_handler.fill_nulls_with_value(col, 'unknown')

print(f"Remaining nulls: {model_df.isnull().sum().sum()}")

# Encode categoricals
encoder = Encoding(model_df)
encoder.frequency_encoding('customer_state')
encoder.frequency_encoding('product_category_name_english')
model_df = encoder.one_hot_encoding('main_payment_type')

print(f"Final feature matrix shape: {model_df.shape}")


#   4. Split & Scale  
print("\n" + "=" * 70)
print("STEP 4: Train/test split and scaling")
print("=" * 70)

X = model_df.drop(columns=['review_score'])
tr_ts_split = Tr_Ts_Split(model_df, target_column='review_score', test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = tr_ts_split.split()

# Shift 1-5 → 0-4 (required by XGBoost)
y_train = y_train - 1
y_test = y_test - 1

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
print(f"\nTarget distribution in train set:")
print(y_train.value_counts(normalize=True).sort_index().round(3))


#   5. Train Models (each logged as an MLflow run)  
print("\n" + "=" * 70)
print("STEP 5: Training models (logging to MLflow)")
print("=" * 70)

trainer = Trainer(X_train_scaled, X_test_scaled, y_train, y_test)
trainer.get_default_models()
trainer.train_and_evaluate(cv_folds=5)

results_df = trainer.get_results_table()
print("\n" + "=" * 70)
print("MODEL COMPARISON (sorted by Test F1)")
print("=" * 70)
print(results_df.to_string())


#   6. Evaluate Best Model (logged as MLflow run with artifacts)  
print("\n" + "=" * 70)
print("STEP 6: Evaluating best model (logging artifacts to MLflow)")
print("=" * 70)

best_name, best_result = trainer.get_best_model()
print(f"\nBest Model: {best_name}\n")

evaluator = Evaluator(
    y_test=y_test,
    y_pred=best_result['predictions'],
    model=best_result['model'],
    feature_names=X_train.columns.tolist()
)

evaluator.log_to_mlflow(best_name, best_result, results_df, X.shape[0], X.shape[1])
evaluator.print_summary(best_name, best_result, X.shape[0], X.shape[1])

print(f"\nMLflow tracking URI: file:./mlruns")
print(f"Run 'mlflow ui' in this directory to view results in the browser.")
