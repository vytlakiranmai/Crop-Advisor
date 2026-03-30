import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load and prepare the dataset
print("Loading dataset...")
df = pd.read_csv(os.path.join('backend', 'Crop_recommendation.csv'))

# Prepare features (X) and target (y)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values
y = df['label'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'models/scaler.joblib')

# Initialize models
models = {
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'svm': SVC(kernel='rbf', probability=True, random_state=42),
    'lr': LogisticRegression(random_state=42, max_iter=1000),
    'knn': KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name} model...")
    model.fit(X_train_scaled, y_train)
    
    # Save the model
    joblib.dump(model, f'models/crop_model_{name}.joblib')
    print(f"Saved {name} model")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique class labels
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    
    # Print confusion matrix with labels
    print(f"\nConfusion Matrix - {name.upper()}:")
    print("True labels (rows) vs Predicted labels (columns)")
    print("Labels:", unique_labels.tolist())
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels)
    plt.title(f'Confusion Matrix - {name.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'models/confusion_matrix_{name}.png')
    plt.close()
    
    # Print classification report
    print(f"\nClassification Report - {name.upper()}:")
    print(classification_report(y_test, y_pred))

print("\nAll models have been trained and saved!")

# Print feature importance for Random Forest
rf_model = models['rf']
feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
feature_importance = dict(zip(feature_names, rf_model.feature_importances_))

print("\nRandom Forest Feature Importance:")
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.3f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=True)

plt.barh(importance_df['feature'], importance_df['importance'])
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('models/feature_importance.png')
plt.close()

# Save feature names and their order for the backend
joblib.dump(feature_names, 'models/feature_names.joblib') 