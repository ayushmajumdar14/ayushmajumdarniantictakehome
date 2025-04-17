from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from pokemon_utils import load_and_prepare_data, print_model_evaluation, plot_feature_importance
import numpy as np

def run_adaboost_analysis():
    """Run AdaBoost analysis on Pokemon data."""
    # Load and prepare data
    X_train, X_test, y_train, y_test, features = load_and_prepare_data()
    
    print("Model: AdaBoost")
    print("Pros:")
    print("- Focuses on difficult-to-classify instances")
    print("- Less prone to overfitting")
    print("- Good for binary classification")
    print("- Works well with imbalanced datasets")
    print("- Can combine multiple weak learners into a strong one")
    print("\nCons:")
    print("- Sensitive to noisy data and outliers")
    print("- Can be slower than single models")
    print("- May require careful tuning of base estimator")
    print("- Performance depends on the strength of base learners")
    
    # Initialize and train AdaBoost model
    model = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Get cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print_model_evaluation(
        "AdaBoost",
        cv_scores,
        classification_report(y_test, y_pred),
        confusion_matrix(y_test, y_pred)
    )
    
    # Plot feature importance
    plot_feature_importance(model.feature_importances_, features, "AdaBoost")
    
    # Print top 5 most important features
    feature_importance = list(zip(features, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Most Important Features for AdaBoost:")
    for feature, importance in feature_importance[:5]:
        print(f"{feature}: {importance:.4f}")
    
    return model, cv_scores.mean()

if __name__ == "__main__":
    model, cv_score = run_adaboost_analysis()
    print(f"\nFinal Cross-Validation Score: {cv_score:.3f}") 