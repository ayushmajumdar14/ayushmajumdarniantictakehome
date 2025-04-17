from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from pokemon_utils import load_and_prepare_data, print_model_evaluation, plot_feature_importance
import numpy as np

def run_random_forest_analysis():
    """Run and evaluate Random Forest model for Mega Evolution prediction."""
    # Load and prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data()
    
    print("Model: Random Forest")
    print("Pros:")
    print("- Handles non-linear relationships well")
    print("- Provides feature importance")
    print("- Less prone to overfitting than individual decision trees")
    print("- Handles both numerical and categorical data well")
    print("\nCons:")
    print("- Less interpretable than logistic regression")
    print("- Can be computationally intensive")
    print("- May require more tuning")
    
    # Train and evaluate model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Calculate cross-validation scores
    rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    
    # Get evaluation metrics
    classification_rep = classification_report(y_test, rf_pred)
    confusion_mat = confusion_matrix(y_test, rf_pred)
    
    # Print results
    print_model_evaluation("Random Forest", rf_cv_scores, classification_rep, confusion_mat)
    
    # Plot feature importance
    plot_feature_importance(rf_model.feature_importances_, features, "Random Forest")
    
    # Print top 5 most important features
    feature_importance = list(zip(features, rf_model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Most Important Features:")
    for feature, importance in feature_importance[:5]:
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    run_random_forest_analysis() 