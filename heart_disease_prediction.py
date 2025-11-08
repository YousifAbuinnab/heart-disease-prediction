"""
Heart Disease Prediction using Machine Learning
Algorithms: Logistic Regression, SVM, Neural Network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('archive/heart_disease_uci.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset info:")
print(df.info())
print(f"\nMissing values:")
print(df.isnull().sum())

# Data Preprocessing
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Convert target variable to binary (0 = no disease, 1 = disease)
df['target'] = (df['num'] > 0).astype(int)

# Drop unnecessary columns
df_processed = df.drop(['id', 'dataset', 'num'], axis=1)

# Handle missing values
print(f"\nMissing values before handling: {df_processed.isnull().sum().sum()}")
df_processed = df_processed.dropna()
print(f"Missing values after handling: {df_processed.isnull().sum().sum()}")

# Encode categorical variables
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
label_encoders = {}

for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

# Separate features and target
X = df_processed.drop('target', axis=1)
y = df_processed['target']

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"\nTarget percentage:\n{y.value_counts(normalize=True) * 100}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Exploratory Data Analysis
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Create visualizations directory
import os
os.makedirs('visualizations', exist_ok=True)

# 1. Target distribution
plt.figure(figsize=(8, 6))
y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Target Variable Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Heart Disease (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualizations/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(14, 10))
correlation_matrix = df_processed.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Age distribution by target
plt.figure(figsize=(10, 6))
df_processed.boxplot(column='age', by='target', ax=plt.gca())
plt.title('Age Distribution by Heart Disease Status', fontsize=16, fontweight='bold')
plt.suptitle('')
plt.xlabel('Heart Disease (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Key features comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Key Features Comparison by Heart Disease Status', fontsize=16, fontweight='bold')

# Cholesterol
df_processed.boxplot(column='chol', by='target', ax=axes[0, 0])
axes[0, 0].set_title('Cholesterol')
axes[0, 0].set_xlabel('Heart Disease')
axes[0, 0].set_ylabel('Cholesterol (mg/dl)')

# Resting Blood Pressure
df_processed.boxplot(column='trestbps', by='target', ax=axes[0, 1])
axes[0, 1].set_title('Resting Blood Pressure')
axes[0, 1].set_xlabel('Heart Disease')
axes[0, 1].set_ylabel('BP (mm Hg)')

# Max Heart Rate
df_processed.boxplot(column='thalch', by='target', ax=axes[1, 0])
axes[1, 0].set_title('Max Heart Rate')
axes[1, 0].set_xlabel('Heart Disease')
axes[1, 0].set_ylabel('Heart Rate (bpm)')

# ST Depression
df_processed.boxplot(column='oldpeak', by='target', ax=axes[1, 1])
axes[1, 1].set_title('ST Depression')
axes[1, 1].set_xlabel('Heart Disease')
axes[1, 1].set_ylabel('ST Depression')

plt.tight_layout()
plt.savefig('visualizations/features_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualizations saved in 'visualizations' directory")

# Model Training and Evaluation
print("\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

models = {}
results = {}

# 1. Logistic Regression
print("\n1. Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_pred_proba_lr)

models['Logistic Regression'] = lr_model
results['Logistic Regression'] = {
    'accuracy': lr_accuracy,
    'auc': lr_auc,
    'predictions': y_pred_lr,
    'probabilities': y_pred_proba_lr
}

print(f"   Accuracy: {lr_accuracy:.4f}")
print(f"   AUC-ROC: {lr_auc:.4f}")

# 2. Support Vector Machine
print("\n2. Training Support Vector Machine...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_auc = roc_auc_score(y_test, y_pred_proba_svm)

models['SVM'] = svm_model
results['SVM'] = {
    'accuracy': svm_accuracy,
    'auc': svm_auc,
    'predictions': y_pred_svm,
    'probabilities': y_pred_proba_svm
}

print(f"   Accuracy: {svm_accuracy:.4f}")
print(f"   AUC-ROC: {svm_auc:.4f}")

# 3. Neural Network
print("\n3. Training Neural Network...")
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True)
nn_model.fit(X_train_scaled, y_train)
y_pred_nn = nn_model.predict(X_test_scaled)
y_pred_proba_nn = nn_model.predict_proba(X_test_scaled)[:, 1]

nn_accuracy = accuracy_score(y_test, y_pred_nn)
nn_auc = roc_auc_score(y_test, y_pred_proba_nn)

models['Neural Network'] = nn_model
results['Neural Network'] = {
    'accuracy': nn_accuracy,
    'auc': nn_auc,
    'predictions': y_pred_nn,
    'probabilities': y_pred_proba_nn
}

print(f"   Accuracy: {nn_accuracy:.4f}")
print(f"   AUC-ROC: {nn_auc:.4f}")

# Model Comparison
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'AUC-ROC': [results[m]['auc'] for m in results.keys()]
})

print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))

# Visualizations for Model Performance
# 1. Model Comparison Bar Chart
plt.figure(figsize=(12, 6))
x = np.arange(len(comparison_df))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, comparison_df['Accuracy'], width, label='Accuracy', color='skyblue')
bars2 = ax.bar(x + width/2, comparison_df['AUC-ROC'], width, label='AUC-ROC', color='salmon')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'])
ax.legend()
ax.set_ylim([0, 1])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. ROC Curves
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {result['auc']:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

for idx, (model_name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}', fontsize=12)
    axes[idx].set_ylabel('True Label', fontsize=11)
    axes[idx].set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# Detailed Classification Reports
print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORTS")
print("="*50)

for model_name, result in results.items():
    print(f"\n{model_name}:")
    print(classification_report(y_test, result['predictions'], 
                                target_names=['No Disease', 'Disease']))

# Save results to CSV
comparison_df.to_csv('model_results.csv', index=False)
print("\nModel results saved to 'model_results.csv'")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print(f"\nBest Model: {comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']}")
print(f"Best Accuracy: {comparison_df['Accuracy'].max():.4f}")
print(f"Best AUC-ROC: {comparison_df['AUC-ROC'].max():.4f}")
print("\nAll visualizations saved in 'visualizations' directory")

