from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd


class Trainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}

    def get_default_models(self):
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7, n_jobs=-1
            ),
            'Gradient Boosting': HistGradientBoostingClassifier(
                max_iter=200, max_depth=5, learning_rate=0.1, random_state=42
            ),
            'XGBoost': XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, eval_metric='mlogloss',
                use_label_encoder=False
            ),
        }
        return self.models

    def set_models(self, models_dict):
        self.models = models_dict

    def _log_model_to_mlflow(self, name, model):
        if isinstance(model, XGBClassifier):
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

    def train_and_evaluate(self, cv_folds=5):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, model in self.models.items():
            with mlflow.start_run(run_name=name):
                print(f"Training {name}...")

                # Log all model hyperparameters
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

                mlflow.log_param("cv_folds", cv_folds)
                mlflow.set_tag("model_type", name)

                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv, scoring='f1_weighted', n_jobs=-1
                )

                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                test_acc = accuracy_score(self.y_test, y_pred)
                test_f1 = f1_score(self.y_test, y_pred, average='weighted')

                # Log metrics
                mlflow.log_metric("cv_f1_mean", cv_scores.mean())
                mlflow.log_metric("cv_f1_std", cv_scores.std())
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("test_f1_weighted", test_f1)

                # Log the trained model artifact
                self._log_model_to_mlflow(name, model)

                self.results[name] = {
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'test_accuracy': test_acc,
                    'test_f1_weighted': test_f1,
                    'model': model,
                    'predictions': y_pred,
                }
                print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f}) "
                      f"| Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

        return self.results

    def get_results_table(self):
        results_df = pd.DataFrame({
            name: {
                'CV F1 (mean)': r['cv_f1_mean'],
                'CV F1 (std)': r['cv_f1_std'],
                'Test Accuracy': r['test_accuracy'],
                'Test F1 (weighted)': r['test_f1_weighted'],
            }
            for name, r in self.results.items()
        }).T.sort_values('Test F1 (weighted)', ascending=False)
        return results_df

    def get_best_model(self):
        results_df = self.get_results_table()
        best_name = results_df.index[0]
        return best_name, self.results[best_name]
