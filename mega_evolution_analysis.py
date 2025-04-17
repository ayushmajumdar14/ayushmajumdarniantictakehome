import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
data_dir = 'pokemon-env'
df = pd.read_csv(f'{data_dir}/pokemon_data_science.csv')

# Prepare features and target
# Select relevant features that might influence Mega Evolution
features = ['Total', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 
           'Generation', 'isLegendary', 'hasGender', 'Pr_Male', 'Height_m', 'Weight_kg', 'Catch_Rate']

# Create X (features) and y (target)
X = df[features]
y = df['hasMegaEvolution']

# Handle missing values
X = X.fillna(X.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Logistic Regression
print("Model 1: Logistic Regression")
print("Pros:")
print("- Simple and interpretable")
print("- Provides probability scores")
print("- Fast to train and predict")
print("- Less prone to overfitting")
print("Cons:")
print("- Assumes linear relationship")
print("- May underperform with complex relationships")
print("- Sensitive to outliers")
print("\nTesting Logistic Regression...")

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)

print(f"Cross-validation accuracy: {lr_cv_scores.mean():.3f} (+/- {lr_cv_scores.std() * 2:.3f})")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# 2. Random Forest
print("\nModel 2: Random Forest")
print("Pros:")
print("- Handles non-linear relationships well")
print("- Provides feature importance")
print("- Less prone to overfitting than individual decision trees")
print("- Handles both numerical and categorical data well")
print("Cons:")
print("- Less interpretable than logistic regression")
print("- Can be computationally intensive")
print("- May require more tuning")
print("\nTesting Random Forest...")

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)

print(f"Cross-validation accuracy: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std() * 2:.3f})")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# feature importance for Random Forest
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Mega Evolution Prediction')
plt.tight_layout()
plt.savefig('mega_evolution_feature_importance.png')
plt.close()

# 3. Support Vector Machine (SVM)
print("\nModel 3: Support Vector Machine")
print("Pros:")
print("- Effective in high-dimensional spaces")
print("- Versatile through different kernel functions")
print("- Good for binary classification")
print("- Works well with clear margin of separation")
print("Cons:")
print("- Can be computationally intensive")
print("- Sensitive to feature scaling")
print("- Less interpretable")
print("- Requires careful parameter tuning")
print("\nTesting SVM...")

svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)

print(f"Cross-validation accuracy: {svm_cv_scores.mean():.3f} (+/- {svm_cv_scores.std() * 2:.3f})")
print("\nClassification Report:")
print(classification_report(y_test, svm_pred))

# Compare model performances
models = ['Logistic Regression', 'Random Forest', 'SVM']
cv_scores = [lr_cv_scores.mean(), rf_cv_scores.mean(), svm_cv_scores.mean()]
cv_stds = [lr_cv_scores.std(), rf_cv_scores.std(), svm_cv_scores.std()]

plt.figure(figsize=(10, 6))
plt.bar(models, cv_scores, yerr=cv_stds)
plt.title('Model Performance Comparison')
plt.ylabel('Cross-validation Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

print("\nModel Comparison Summary:")
for model, score, std in zip(models, cv_scores, cv_stds):
    print(f"{model}: {score:.3f} (+/- {std * 2:.3f})")

# Additional analysis of prediction errors
print("\nConfusion Matrix Analysis:")
for model_name, y_pred in [("Logistic Regression", lr_pred), 
                          ("Random Forest", rf_pred), 
                          ("SVM", svm_pred)]:
    print(f"\n{model_name}:")
    print(confusion_matrix(y_test, y_pred))

# Class distribution analysis
print("\nClass Distribution:")
print(df['hasMegaEvolution'].value_counts(normalize=True)) 