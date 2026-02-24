import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


class Evaluator:
    def __init__(self, y_test, y_pred, model=None, feature_names=None):
        self.y_test = y_test
        self.y_pred = y_pred
        self.model = model
        self.feature_names = feature_names

    def print_classification_report(self):
        labels = sorted(self.y_test.unique())
        target_names = [f'Score {i + 1}' for i in labels]
        report = classification_report(self.y_test, self.y_pred, target_names=target_names)
        print(report)
        return report

    def plot_confusion_matrix(self, save_path=None):
        labels = [str(i + 1) for i in sorted(self.y_test.unique())]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        cm = confusion_matrix(self.y_test, self.y_pred)
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=axes[0], cmap='Blues', values_format='d'
        )
        axes[0].set_title('Confusion Matrix')

        cm_norm = confusion_matrix(self.y_test, self.y_pred, normalize='true')
        ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(
            ax=axes[1], cmap='Blues', values_format='.2f'
        )
        axes[1].set_title('Normalized Confusion Matrix')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()
        return fig

    def plot_feature_importance(self, top_n=20, save_path=None):
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not support feature_importances_")
            return None

        importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
        top_features = importances.nlargest(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features.sort_values().plot(kind='barh', color='steelblue', ax=ax)
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_xlabel('Importance')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()
        return importances

    @staticmethod
    def plot_model_comparison(results_df, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df[['Test F1 (weighted)', 'CV F1 (mean)']].plot(
            kind='barh', color=['steelblue', 'coral'], ax=ax
        )
        ax.set_title('Model Comparison: F1 Scores')
        ax.set_xlabel('F1 Score (weighted)')
        ax.legend(loc='lower right')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()
        return fig

    def log_to_mlflow(self, model_name, result_dict, results_df, n_samples, n_features):
        artifacts_dir = "mlflow_artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("best_model", model_name)
            mlflow.log_metric("best_test_accuracy", result_dict['test_accuracy'])
            mlflow.log_metric("best_test_f1_weighted", result_dict['test_f1_weighted'])
            mlflow.log_metric("best_cv_f1_mean", result_dict['cv_f1_mean'])
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("n_features", n_features)

            # Save and log confusion matrix
            cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
            self.plot_confusion_matrix(save_path=cm_path)
            mlflow.log_artifact(cm_path)

            # Save and log feature importance
            if hasattr(self.model, 'feature_importances_'):
                fi_path = os.path.join(artifacts_dir, "feature_importance.png")
                self.plot_feature_importance(save_path=fi_path)
                mlflow.log_artifact(fi_path)

            # Save and log model comparison
            mc_path = os.path.join(artifacts_dir, "model_comparison.png")
            Evaluator.plot_model_comparison(results_df, save_path=mc_path)
            mlflow.log_artifact(mc_path)

            # Log classification report as text artifact
            report = classification_report(
                self.y_test, self.y_pred,
                target_names=[f'Score {i + 1}' for i in sorted(self.y_test.unique())]
            )
            report_path = os.path.join(artifacts_dir, "classification_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            mlflow.log_artifact(report_path)

            # Log results comparison table
            table_path = os.path.join(artifacts_dir, "model_comparison.csv")
            results_df.to_csv(table_path)
            mlflow.log_artifact(table_path)

        print(f"Evaluation artifacts logged to MLflow.")

    def print_summary(self, model_name, result_dict, n_samples, n_features):
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Dataset: {n_samples} orders, {n_features} features")
        print(f"Target: review_score (1-5, multi-class classification)")
        print(f"Best model: {model_name}")
        print(f"  - Test Accuracy: {result_dict['test_accuracy']:.4f}")
        print(f"  - Test F1 (weighted): {result_dict['test_f1_weighted']:.4f}")
        print(f"  - CV F1 (mean +/- std): {result_dict['cv_f1_mean']:.4f} "
              f"+/- {result_dict['cv_f1_std']:.4f}")

        if hasattr(self.model, 'feature_importances_') and self.feature_names is not None:
            importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
            print(f"\nTop 5 most important features:")
            for feat, imp in importances.nlargest(5).items():
                print(f"  {feat}: {imp:.4f}")
