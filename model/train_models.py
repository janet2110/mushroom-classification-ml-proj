"""
ML Assignment 2 - Mushroom Classification
Train all 6 models and save them for Streamlit app

Student Details:
BITS ID: 2025AA05835
Name: JANET DEVARAJ
Email: 2025aa05835@wilp.bits-pilani.ac.in
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

import joblib
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a classification model and return all required metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # For AUC, we need probability predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = roc_auc_score(y_test, y_pred)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC Score: {mcc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    }

def main():
    print("="*80)
    print("ML Assignment 2 - Mushroom Classification")
    print("Student: JANET DEVARAJ (2025AA05835)")
    print("="*80)
    
    # 1. Load Dataset
    print("\n[1/8] Loading dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
               'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
               'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
               'stalk-surface-below-ring', 'stalk-color-above-ring',
               'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
               'ring-type', 'spore-print-color', 'population', 'habitat']
    
    df = pd.read_csv(url, names=columns)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Features: {len(columns) - 1}")
    print(f"Classes: {len(df['class'].unique())}")
    print(f"Samples: {len(df)}")
    
    # 2. Prepare Data
    print("\n[2/8] Preparing data...")
    
    # Handle missing values
    df = df.replace('?', np.nan)
    print(f"Missing values: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"Shape after removing missing values: {df.shape}")
    
    # Encode all categorical features
    label_encoders = {}
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Save label encoders
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("Label encoders saved!")
    
    # Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Class distribution in training: {y_train.value_counts().to_dict()}")
    print(f"Class distribution in testing: {y_test.value_counts().to_dict()}")
    
    # Save test data
    test_df = X_test.copy()
    test_df['class'] = y_test.values
    test_df.to_csv('test_data.csv', index=False)
    print("Test data saved!")
    
    # 3. Train Models
    results = []
    
    # Model 1: Logistic Regression
    print("\n[3/8] Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=10000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results.append(lr_metrics)
    joblib.dump(lr_model, 'logistic_regression_model.pkl')
    print("✓ Model saved as 'logistic_regression_model.pkl'")
    
    # Model 2: Decision Tree
    print("\n[4/8] Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_metrics = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    results.append(dt_metrics)
    joblib.dump(dt_model, 'decision_tree_model.pkl')
    print("✓ Model saved as 'decision_tree_model.pkl'")
    
    # Model 3: K-Nearest Neighbors
    print("\n[5/8] Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_metrics = evaluate_model(knn_model, X_test, y_test, "K-Nearest Neighbors")
    results.append(knn_metrics)
    joblib.dump(knn_model, 'knn_model.pkl')
    print("✓ Model saved as 'knn_model.pkl'")
    
    # Model 4: Naive Bayes
    print("\n[6/8] Training Naive Bayes (Gaussian)...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_metrics = evaluate_model(nb_model, X_test, y_test, "Naive Bayes (Gaussian)")
    results.append(nb_metrics)
    joblib.dump(nb_model, 'naive_bayes_model.pkl')
    print("✓ Model saved as 'naive_bayes_model.pkl'")
    
    # Model 5: Random Forest
    print("\n[7/8] Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results.append(rf_metrics)
    joblib.dump(rf_model, 'random_forest_model.pkl')
    print("✓ Model saved as 'random_forest_model.pkl'")
    
    # Model 6: XGBoost
    print("\n[8/8] Training XGBoost...")
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    results.append(xgb_metrics)
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    print("✓ Model saved as 'xgboost_model.pkl'")
    
    # 4. Create Comparison Table
    print("\n" + "="*80)
    print("MODEL COMPARISON - ALL EVALUATION METRICS")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\n✓ Results saved to 'model_comparison_results.csv'")
    
    # 5. Visualize Results
    print("\nGenerating comparison visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Mushroom Classification - Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = ['#2e7d32', '#d32f2f', '#1976d2', '#f57c00', '#7b1fa2', '#00796b']
    
    for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
        ax.bar(results_df['Model'], results_df[metric], color=color, alpha=0.7)
        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.1])
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(results_df[metric]):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to 'model_comparison.png'")
    
    # 6. Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTotal Models Trained: 6")
    print(f"Best Model (Accuracy): {results_df.loc[results_df['Accuracy'].idxmax(), 'Model']}")
    print(f"Best Accuracy: {results_df['Accuracy'].max():.4f}")
    print(f"Best Model (F1): {results_df.loc[results_df['F1'].idxmax(), 'Model']}")
    print(f"Best F1 Score: {results_df['F1'].max():.4f}")
    print(f"Best Model (MCC): {results_df.loc[results_df['MCC'].idxmax(), 'Model']}")
    print(f"Best MCC Score: {results_df['MCC'].max():.4f}")
    
    print("\nFiles Generated:")
    print("  ✓ label_encoders.pkl")
    print("  ✓ test_data.csv")
    print("  ✓ logistic_regression_model.pkl")
    print("  ✓ decision_tree_model.pkl")
    print("  ✓ knn_model.pkl")
    print("  ✓ naive_bayes_model.pkl")
    print("  ✓ random_forest_model.pkl")
    print("  ✓ xgboost_model.pkl")
    print("  ✓ model_comparison_results.csv")
    print("  ✓ model_comparison.png")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Run the Streamlit app:")
    print("   streamlit run app.py")
    print("\n2. Upload 'test_data.csv' in the app to see predictions")
    print("\n3. Compare all 6 models using the 'Model Comparison' tab")
    print("="*80)
    
    print("\nStudent: JANET DEVARAJ")
    print("BITS ID: 2025AA05835")
    print("Email: 2025aa05835@wilp.bits-pilani.ac.in")
    print("="*80)

if __name__ == "__main__":
    main()
