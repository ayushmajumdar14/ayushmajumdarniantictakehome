from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from pokemon_utils import load_and_prepare_data, print_model_evaluation
import numpy as np

def run_knn_analysis():
    """Run and evaluate K-Nearest Neighbors model for Mega Evolution prediction."""
    # Load and prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, features = load_and_prepare_data()
    
    print("Model: K-Nearest Neighbors")
    print("Pros:")
    print("- Simple and intuitive algorithm")
    print("- No assumptions about data distribution")
    print("- Can model non-linear decision boundaries")
    print("- No training phase (lazy learning)")
    print("- Easy to understand and interpret")
    print("\nCons:")
    print("- Computationally expensive for large datasets")
    print("- Sensitive to irrelevant features")
    print("- Requires feature scaling")
    print("- Memory intensive (stores all training data)")
    
    # Create and train model
    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,  # Euclidean distance
        metric='minkowski'
    )
    
    # Train the model
    knn_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    knn_pred = knn_model.predict(X_test_scaled)
    
    # Calculate cross-validation scores
    knn_cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5)
    
    # Get evaluation metrics
    classification_rep = classification_report(y_test, knn_pred)
    confusion_mat = confusion_matrix(y_test, knn_pred)
    
    # Print results
    print_model_evaluation("K-Nearest Neighbors", knn_cv_scores, classification_rep, confusion_mat)
    
    # Since KNN doesn't have built-in feature importance, we can calculate feature importance
    # based on the correlation between features and predictions
    feature_importance = []
    for i in range(X_train_scaled.shape[1]):
        correlation = np.corrcoef(X_train_scaled[:, i], y_train)[0, 1]
        feature_importance.append((features[i], abs(correlation)))
    
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Most Correlated Features for KNN:")
    for feature, importance in feature_importance[:5]:
        print(f"{feature}: {importance:.4f}")
    
    return knn_model, knn_cv_scores.mean()

if __name__ == "__main__":
    model, cv_score = run_knn_analysis()
    print(f"\nFinal Cross-Validation Score: {cv_score:.3f}") 