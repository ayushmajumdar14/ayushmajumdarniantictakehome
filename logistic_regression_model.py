from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from pokemon_utils import load_and_prepare_data, print_model_evaluation
import numpy as np

def run_logistic_regression_analysis():
    """Run and evaluate Logistic Regression model for Mega Evolution prediction."""
    # Load and prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data()
    
    print("Model: Logistic Regression")
    print("Pros:")
    print("- Simple and interpretable")
    print("- Provides probability scores")
    print("- Fast to train and predict")
    print("- Less prone to overfitting")
    print("\nCons:")
    print("- Assumes linear relationship")
    print("- May underperform with complex relationships")
    print("- Sensitive to outliers")
    
    # rain and evaluate model
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    # calculate cross-validation scores
    lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)
    
    # get evaluation metrics
    classification_rep = classification_report(y_test, lr_pred)
    confusion_mat = confusion_matrix(y_test, lr_pred)
    
    # print results
    print_model_evaluation("Logistic Regression", lr_cv_scores, classification_rep, confusion_mat)
    
    # print feature coefficients
    print("\nFeature Coefficients:")
    for feature, coef in zip(features, lr_model.coef_[0]):
        print(f"{feature}: {coef:.4f}")

if __name__ == "__main__":
    run_logistic_regression_analysis() 