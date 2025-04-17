from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from pokemon_utils import load_and_prepare_data, print_model_evaluation
import numpy as np

def run_svm_analysis():
    """Run and evaluate SVM model for Mega Evolution prediction."""
    # Load and prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data()
    
    print("Model: Support Vector Machine")
    print("Pros:")
    print("- Effective in high-dimensional spaces")
    print("- Versatile through different kernel functions")
    print("- Good for binary classification")
    print("- Works well with clear margin of separation")
    print("\nCons:")
    print("- Can be computationally intensive")
    print("- Sensitive to feature scaling")
    print("- Less interpretable")
    print("- Requires careful parameter tuning")
    
    # Train and evaluate model
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    
    # Calculate cross-validation scores
    svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
    
    # Get evaluation metrics
    classification_rep = classification_report(y_test, svm_pred)
    confusion_mat = confusion_matrix(y_test, svm_pred)
    
    # Print results
    print_model_evaluation("Support Vector Machine", svm_cv_scores, classification_rep, confusion_mat)
    
    # Print support vectors information
    print("\nModel Details:")
    print(f"Number of Support Vectors: {svm_model.n_support_}")
    print(f"Number of Support Vectors per Class: {dict(zip(['No Mega', 'Has Mega'], svm_model.n_support_))}")

if __name__ == "__main__":
    run_svm_analysis() 