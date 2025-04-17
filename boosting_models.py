from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from pokemon_utils import load_and_prepare_data, print_model_evaluation, plot_feature_importance
import numpy as np

def run_adaboost_analysis():
    """Run and evaluate AdaBoost model."""
    X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data()
    
    print("Model: AdaBoost")
    print("Pros:")
    print("- Focuses on difficult-to-classify instances")
    print("- Less prone to overfitting")
    print("- Good for binary classification")
    print("- Works well with imbalanced datasets")
    print("\nCons:")
    print("- Sensitive to noisy data")
    print("- Can be slower than single models")
    print("- May require careful tuning of base estimator")
    
    ada_model = AdaBoostClassifier(random_state=42)
    ada_model.fit(X_train_scaled, y_train)
    ada_pred = ada_model.predict(X_test_scaled)
    ada_cv_scores = cross_val_score(ada_model, X_train_scaled, y_train, cv=5)
    
    print_model_evaluation("AdaBoost", ada_cv_scores, 
                         classification_report(y_test, ada_pred),
                         confusion_matrix(y_test, ada_pred))
    
    plot_feature_importance(ada_model.feature_importances_, features, "AdaBoost")

def run_xgboost_analysis():
    """Run and evaluate XGBoost model."""
    X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data()
    
    print("Model: XGBoost")
    print("Pros:")
    print("- Highly efficient implementation of gradient boosting")
    print("- Built-in handling of missing values")
    print("- Advanced regularization")
    print("- Excellent feature importance metrics")
    print("\nCons:")
    print("- More hyperparameters to tune")
    print("- Can overfit if not properly configured")
    print("- More complex than simpler models")
    
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5)
    
    print_model_evaluation("XGBoost", xgb_cv_scores,
                         classification_report(y_test, xgb_pred),
                         confusion_matrix(y_test, xgb_pred))
    
    plot_feature_importance(xgb_model.feature_importances_, features, "XGBoost")

def run_lightgbm_analysis():
    """Run and evaluate LightGBM model."""
    X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data()
    
    print("Model: LightGBM")
    print("Pros:")
    print("- Faster training speed and higher efficiency")
    print("- Lower memory usage")
    print("- Better accuracy than many other boosting algorithms")
    print("- Handles large datasets well")
    print("\nCons:")
    print("- Can be sensitive to overfitting")
    print("- Requires careful parameter tuning")
    print("- May not perform as well on small datasets")
    
    lgb_model = LGBMClassifier(random_state=42)
    lgb_model.fit(X_train_scaled, y_train)
    lgb_pred = lgb_model.predict(X_test_scaled)
    lgb_cv_scores = cross_val_score(lgb_model, X_train_scaled, y_train, cv=5)
    
    print_model_evaluation("LightGBM", lgb_cv_scores,
                         classification_report(y_test, lgb_pred),
                         confusion_matrix(y_test, lgb_pred))
    
    plot_feature_importance(lgb_model.feature_importances_, features, "LightGBM")

def compare_boosting_models():
    """Compare all boosting models."""
    X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data()
    
    models = {
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42)
    }
    
    results = {}
    feature_importance_dict = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        results[name] = cv_scores.mean()
        feature_importance_dict[name] = model.feature_importances_
    
    print("\nModel Comparison:")
    print("=" * 50)
    for name, score in results.items():
        print(f"{name}: {score:.3f}")
    
    # Plot comparison of feature importances across models
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(features))
    width = 0.25
    
    for i, (name, importance) in enumerate(feature_importance_dict.items()):
        plt.bar(x + i*width, importance, width, label=name)
    
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance Comparison Across Boosting Models')
    plt.xticks(x + width, features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('boosting_models_comparison.png')
    plt.close()

if __name__ == "__main__":
    print("Running all boosting models analysis...")
    run_adaboost_analysis()
    run_xgboost_analysis()
    run_lightgbm_analysis()
    print("\nRunning comparison of all boosting models...")
    compare_boosting_models() 