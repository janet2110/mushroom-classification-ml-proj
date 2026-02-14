"""
Mushroom Classification - Interactive Web Application
Machine Learning Assignment 2

Student Details:
BITS ID: 2025AA05835
Name: JANET DEVARAJ
Email: 2025aa05835@wilp.bits-pilani.ac.in
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff6f00;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🍄 Mushroom Classification System</p>', unsafe_allow_html=True)
st.markdown("### Machine Learning Assignment 2")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🍄 Project Information")
    
    st.markdown("### 👨‍🎓 Student Details")
    st.info("""
    **BITS ID:** 2025AA05835  
    **Name:** JANET DEVARAJ  
    **Email:** 2025aa05835@wilp.bits-pilani.ac.in
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Dataset Information")
    st.info("""
    **Dataset**: UCI Mushroom Classification
    
    **Features**: 22 categorical features
    
    **Classes**: 
    - Edible (0)
    - Poisonous (1)
    
    **Samples**: 8,124 instances
    """)
    
    st.markdown("---")
    
    st.markdown("### 🍄 Fun Mushroom Facts")
    st.success("""
    - Over 10,000 mushroom species exist!
    - Some mushrooms glow in the dark 🌟
    - The largest living organism is a fungus
    - Mushrooms are closer to animals than plants
    - A single Portobello mushroom has more potassium than a banana!
    """)

# Model names and file paths
MODELS = {
    "Logistic Regression": "model/logistic_regression_model.pkl",
    "Decision Tree": "model/decision_tree_model.pkl",
    "K-Nearest Neighbors": "model/knn_model.pkl",
    "Naive Bayes (Gaussian)": "model/naive_bayes_model.pkl",
    "Random Forest": "model/random_forest_model.pkl",
    "XGBoost": "model/xgboost_model.pkl"
}

# Function to load model
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except:
        return None

# Function to load label encoders
@st.cache_resource
def load_encoders():
    try:
        return joblib.load('label_encoders.pkl')
    except:
        return None

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
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
    
    return {
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc
    }, y_pred, confusion_matrix(y_test, y_pred)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📈 Model Comparison", "ℹ️ About"])

with tab1:
    st.markdown('<p class="sub-header">Upload Test Data</p>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your test dataset in CSV format")
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        # Display data preview
        with st.expander("🔍 View Data Preview"):
            st.dataframe(df.head(10))
        
        # Check if target column exists
        if 'class' in df.columns:
            X_test = df.drop('class', axis=1)
            y_test = df['class']
            has_target = True
        else:
            X_test = df
            y_test = None
            has_target = False
            st.warning("⚠️ No 'class' column found. Predictions will be made without evaluation.")
        
        # Model selection
        st.markdown('<p class="sub-header">Select Model</p>', unsafe_allow_html=True)
        selected_model_name = st.selectbox(
            "Choose a classification model:",
            list(MODELS.keys()),
            help="Select the model you want to use for prediction"
        )
        
        if st.button("🚀 Run Prediction", type="primary"):
            # Load model
            model = load_model(MODELS[selected_model_name])
            
            if model is None:
                st.error("❌ Model file not found. Please ensure the model is trained and saved.")
            else:
                with st.spinner('Making predictions...'):
                    
                    if has_target:
                        # Evaluate model
                        metrics, y_pred, cm = evaluate_model(model, X_test, y_test)
                        
                        # Display metrics
                        st.markdown('<p class="sub-header">📊 Evaluation Metrics</p>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                            st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                        
                        with col2:
                            st.metric("Precision", f"{metrics['Precision']:.4f}")
                            st.metric("Recall", f"{metrics['Recall']:.4f}")
                        
                        with col3:
                            st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                            st.metric("MCC Score", f"{metrics['MCC']:.4f}")
                        
                        # Confusion Matrix
                        st.markdown('<div class="metric-card"><h3>🎯 Confusion Matrix</h3></div>', unsafe_allow_html=True)
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            fig, ax = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                                        xticklabels=['Edible', 'Poisonous'],
                                        yticklabels=['Edible', 'Poisonous'])
                            plt.title(f'Confusion Matrix - {selected_model_name}')
                            plt.ylabel('Actual')
                            plt.xlabel('Predicted')
                            st.pyplot(fig)

                        st.markdown('<div class="metric-card"><h3>📋 Classification Report</h3></div>', unsafe_allow_html=True)
                        report = classification_report(y_test, y_pred,
                                                        target_names=['Edible', 'Poisonous'],
                                                        output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.highlight_max(axis=0))
                        
                        # Prediction distribution
                        st.markdown('<p class="sub-header">📊 Prediction Distribution</p>', unsafe_allow_html=True)
                        
                        pred_df = pd.DataFrame({
                            'Actual': y_test.values,
                            'Predicted': y_pred
                        })
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            pred_df['Actual'].value_counts().plot(kind='bar', ax=ax, color=['#4caf50', '#f44336'])
                            plt.title('Actual Class Distribution')
                            plt.xlabel('Class (0: Edible, 1: Poisonous)')
                            plt.ylabel('Count')
                            plt.xticks(rotation=0)
                            st.pyplot(fig)
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            pd.Series(y_pred).value_counts().plot(kind='bar', ax=ax, color=['#4caf50', '#f44336'])
                            plt.title('Predicted Class Distribution')
                            plt.xlabel('Class (0: Edible, 1: Poisonous)')
                            plt.ylabel('Count')
                            plt.xticks(rotation=0)
                            st.pyplot(fig)
                    
                    else:
                        # Just make predictions
                        y_pred = model.predict(X_test)
                        
                        st.markdown('<p class="sub-header">🔮 Predictions</p>', unsafe_allow_html=True)
                        
                        pred_df = pd.DataFrame({
                            'Sample Index': range(len(y_pred)),
                            'Prediction': y_pred,
                            'Class': ['Edible' if p == 0 else 'Poisonous' for p in y_pred]
                        })
                        
                        st.dataframe(pred_df)
                        
                        # Summary
                        col1, col2 = st.columns(2)
                        with col1:
                            edible_count = (y_pred == 0).sum()
                            st.metric("🟢 Edible Mushrooms", edible_count)
                        with col2:
                            poisonous_count = (y_pred == 1).sum()
                            st.metric("🔴 Poisonous Mushrooms", poisonous_count)
                        
                        # Download predictions
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="mushroom_predictions.csv",
                            mime="text/csv"
                        )

with tab2:
    st.markdown('<p class="sub-header">Model Performance Comparison</p>', unsafe_allow_html=True)
    
    # Load comparison results if available
    try:
        results_df = pd.read_csv('model/model_comparison_results.csv')
        
        # Display comparison table
        st.markdown("#### 📋 Comprehensive Metrics Comparison")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']))
        
        # Visualizations
        st.markdown("#### 📊 Visual Comparison")
        
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
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
        st.pyplot(fig)
        
        # Best models
        st.markdown("#### 🏆 Best Performing Models")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_acc_value = results_df['Accuracy'].max()
            best_acc_models = results_df[results_df['Accuracy'] == best_acc_value]['Model'].tolist()
            models_text = "  \n".join([f"• {model}" for model in best_acc_models])
            st.success(f"**Best Accuracy: {best_acc_value:.4f}**\n\n{models_text}")

        with col2:
            best_f1_value = results_df['F1'].max()
            best_f1_models = results_df[results_df['F1'] == best_f1_value]['Model'].tolist()
            models_text = "  \n".join([f"• {model}" for model in best_f1_models])
            st.success(f"**Best F1 Score: {best_f1_value:.4f}**\n\n{models_text}")

        with col3:
            best_mcc_value = results_df['MCC'].max()
            best_mcc_models = results_df[results_df['MCC'] == best_mcc_value]['Model'].tolist()
            models_text = "  \n".join([f"• {model}" for model in best_mcc_models])
            st.success(f"**Best MCC Score: {best_mcc_value:.4f}**\n\n{models_text}")

    except FileNotFoundError:
        st.warning("⚠️ Model comparison results not found. Please train the models first using the training script.")

with tab3:
    st.markdown('<p class="sub-header">About This Application</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    This application demonstrates a complete machine learning pipeline for mushroom classification 
    using the UCI Mushroom Classification dataset to predict whether a mushroom is edible or poisonous.
    
    ### 📊 Dataset Details
    - **Source**: UCI Machine Learning Repository
    - **Dataset Link:** https://archive.ics.uci.edu/ml/datasets/mushroom

    - **Features**: 22 categorical features describing mushroom characteristics
    - **Classes**: Binary classification (Edible vs Poisonous)
    - **Total Samples**: 8,124 instances
    - **Feature Categories**: Cap, Gill, Stalk, Veil, Ring, Spore, Population, and Habitat characteristics
    
    ### 🍄 Feature Information
    The dataset includes features such as:
    - Cap shape, surface, and color
    - Bruises presence
    - Odor characteristics
    - Gill attachment, spacing, size, and color
    - Stalk features (shape, root, surface, color)
    - Veil type and color
    - Ring number and type
    - Spore print color
    - Population and habitat
    
    ### 🤖 Models Implemented
    1. **Logistic Regression**: Linear model for binary classification
    2. **Decision Tree**: Tree-based model with interpretable rules
    3. **K-Nearest Neighbors**: Instance-based learning algorithm
    4. **Naive Bayes (Gaussian)**: Probabilistic classifier based on Bayes' theorem
    5. **Random Forest**: Ensemble of decision trees
    6. **XGBoost**: Gradient boosting framework
    
    ### 📈 Evaluation Metrics
    - **Accuracy**: Overall correctness of predictions
    - **AUC Score**: Area Under the ROC Curve
    - **Precision**: Proportion of positive predictions that are correct
    - **Recall**: Proportion of actual positives correctly identified
    - **F1 Score**: Harmonic mean of precision and recall
    - **MCC**: Matthews Correlation Coefficient
    
    ### 🛠️ Technologies Used
    - Python 3.12
    - Scikit-learn
    - XGBoost
    - Streamlit
    - Pandas, NumPy
    - Matplotlib, Seaborn
    
    ### 👨‍💻 Developer
    **JANET DEVARAJ**  
    BITS ID: 2025AA05835  
    Email: 2025aa05835@wilp.bits-pilani.ac.in
    
    **Course**: M.Tech (AIML/DSE) - Machine Learning Assignment 2
    
    ### 📝 Instructions
    1. Upload your test dataset (CSV format with encoded features)
    2. Select a model from the dropdown
    3. Click "Run Prediction" to see results
    4. View evaluation metrics, confusion matrix, and classification report
    
    ---
    
    *This is an educational project for demonstrating ML model deployment.*
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🍄 Built with ❤️ using Streamlit | ML Assignment 2 - JANET DEVARAJ (2025AA05835) | 2026</p>
    </div>
    """,
    unsafe_allow_html=True
)
