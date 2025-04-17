from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from pokemon_utils import load_and_prepare_data, print_model_evaluation, plot_feature_importance
import numpy as np

def run_xgboost_analysis():
    """Run and evaluate XGBoost model for Mega Evolution prediction."""
    # Load and prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data()
    
    print("Model: XGBoost")
    print("Pros:")
    print("- Highly efficient implementation of gradient boosting")
    print("- Built-in handling of missing values")
    print("- Advanced regularization to prevent overfitting")
    print("- Excellent feature importance metrics")
    print("- Parallel processing support")
    print("\nCons:")
    print("- More hyperparameters to tune compared to simpler models")
    print("- Can overfit if not properly configured")
    print("- More complex than traditional models")
    print("- Requires careful parameter tuning for optimal performance")
    
    # Create and train model with some tuned parameters
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Train the model
    xgb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    xgb_pred = xgb_model.predict(X_test_scaled)
    
    # Calculate cross-validation scores
    xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5)
    
    # Get evaluation metrics
    classification_rep = classification_report(y_test, xgb_pred)
    confusion_mat = confusion_matrix(y_test, xgb_pred)
    
    # Print results
    print_model_evaluation("XGBoost", xgb_cv_scores, classification_rep, confusion_mat)
    
    # Plot feature importance
    plot_feature_importance(xgb_model.feature_importances_, features, "XGBoost")
    
    # Print top 5 most important features
    feature_importance = list(zip(features, xgb_model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Most Important Features for XGBoost:")
    for feature, importance in feature_importance[:5]:
        print(f"{feature}: {importance:.4f}")
    
    return xgb_model, xgb_cv_scores.mean()

if __name__ == "__main__":
    model, cv_score = run_xgboost_analysis()
    print(f"\nFinal Cross-Validation Score: {cv_score:.3f}")