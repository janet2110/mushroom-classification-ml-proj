# 🍄 Mushroom Classification - Machine Learning Assignment 2

## 👨‍🎓 Student Details
- **BITS ID:** 2025AA05835  
- **Name:** JANET DEVARAJ  
- **Email:** 2025aa05835@wilp.bits-pilani.ac.in  

## 🧩 Problem Statement
> **Goal:** Build a binary classification system to predict whether a mushroom is **edible** or **poisonous** based on its physical characteristics.

## 📊 Dataset Description
- **Source:** UCI Machine Learning Repository - Mushroom Dataset  
- **Dataset Link:** https://archive.ics.uci.edu/ml/datasets/mushroom

- **Overview:**
	- This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family
	- Each mushroom is classified as definitely edible or definitely poisonous

- **Total Instances:** 8,124  
- **Number of Features:** 22 (categorical)  
- **Target Variable:** Class (Edible or Poisonous)  
- **Missing Values:** Some features contain missing values denoted by '?'  
- **Feature Categories**: Cap, Gill, Stalk, Veil, Ring, Spore, Population, and Habitat characteristics

## ⚙️ Models Used
- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors  
- Naive Bayes (Gaussian)  
- Random Forest  
- XGBoost  

## 📊 Comparison Table — Evaluation Metrics

| Model                  | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
|------------------------|----------|-------|-----------|--------|----------|-------|
| Logistic Regression    | 0.9628   | 0.9887| 0.9629    | 0.9628 | 0.9627   | 0.921 |
| Decision Tree          | 1.0000   | 1.0000| 1.0000    | 1.0000 | 1.0000   | 1.000 |
| K-Nearest Neighbors    | 0.9991   | 1.0000| 0.9991    | 0.9991 | 0.9991   | 0.9981|
| Naive Bayes (Gaussian) | 0.7130   | 0.9570| 0.7739    | 0.7130 | 0.6612   | 0.394 |
| Random Forest          | 1.0000   | 1.0000| 1.0000    | 1.0000 | 1.0000   | 1.000 |
| XGBoost                | 1.0000   | 1.0000| 1.0000    | 1.0000 | 1.0000   | 1.000 |

## 📌 Model Performance Observations
| ML Model Name          | Observation about Model Performance |
|------------------------|-------------------------------------|
| Logistic Regression    | Achieved **96.3% accuracy** with strong AUC (0.9887). As a linear classifier, it performs remarkably well, suggesting strong linear separability between edible and poisonous mushrooms. It is fast to train and interpretable via feature coefficients. Slightly underperforms compared to ensemble methods, indicating some non-linear relationships in the data. |
| Decision Tree          | Achieved **perfect 100% accuracy**. Captured all complex non-linear decision boundaries in the mushroom features. Highly interpretable, showing which features (e.g., odor, spore-print-color) drive classification. Perfect accuracy may suggest overfitting, but cross-validation supports genuine performance. |
| K-Nearest Neighbors    | Achieved **99.9% accuracy** with k=5 neighbors. Excellent performance indicates that similar mushrooms in feature space belong to the same class, validating biological consistency. KNN adapts well to local patterns but is computationally expensive during prediction. |
| Naive Bayes (Gaussian) | Achieved only **71.3% accuracy**, the lowest among all models. The conditional independence assumption is violated here, leading to poor performance. Mushroom features are highly correlated, which Naive Bayes cannot capture. Still, it is extremely fast and memory-efficient, making it suitable for lightweight real-time tasks despite lower accuracy. |
| Random Forest          | Achieved **perfect 100% accuracy** through ensemble learning. Combines multiple decision trees to reduce overfitting while maintaining high accuracy. Provides feature importance rankings and captures complex non-linear interactions effectively. More reliable than a single decision tree. |
| XGBoost                | Achieved **perfect 100% accuracy** using gradient boosting. Sequentially builds trees to correct previous errors, with built-in regularization to prevent overfitting. Highly efficient, robust, and considered state-of-the-art for tabular classification tasks. |

## 📂 Project Structure
	mushroom-classification-ml-proj/
	│
	├── app.py                           # Streamlit web application
	├── requirements.txt                 # Python dependencies
	├── README.md                        # Project documentation
	│
	└── model/
	├── train_models.py              # Model training script
	├── train_models.ipynb           # Jupyter notebook for training
	├── decision_tree_model.pkl      # Saved Decision Tree model
	├── knn_model.pkl                # Saved KNN model
	├── logistic_regression_model.pkl # Saved Logistic Regression model
	├── naive_bayes_model.pkl        # Saved Naive Bayes model
	├── random_forest_model.pkl      # Saved Random Forest model
	├── xgboost_model.pkl            # Saved XGBoost model
	├── label_encoders.pkl           # Saved label encoders
	├── test_data.csv                # Test dataset features
	├── model_comparison.png         # Visualization of model comparison
	└── model_comparison_results.csv # Tabular results of model comparison


## 🛠️ Technologies Used
- **Python 3.12**  
- **Streamlit**  
- **Scikit-learn**  
- **XGBoost**  
- **Pandas**  
- **NumPy**  
- **Plotly**  
- **Matplotlib/Seaborn**  

---

<br><br><br><br><br><br><br><br><br><br>
---

# 📚 Extended Documentation (For Future Use)

## 🧩 Data Preprocessing
- Label encoding for categorical features  
- Removal of missing values  
- Stratified 80/20 train-test split  


### 🔑 Key Insights

1. **Perfect Classification:** Four models — Decision Tree, KNN, Random Forest, and XGBoost — achieved near-perfect or perfect classification, showing that the mushroom dataset has clear distinguishing features between edible and poisonous classes.

2. **Feature Separability:** The consistently high performance across most models indicates strong feature discriminability. Certain attributes (like **odor** and **spore-print-color**) are highly predictive.

3. **Model Complexity Trade-off:** Logistic Regression achieved **96.3% accuracy**, performing very well while remaining fast and interpretable. In contrast, Naive Bayes dropped to **71.3% accuracy**, highlighting its limitations when features are correlated.

4. **Ensemble Superiority:** Ensemble methods (Random Forest and XGBoost) delivered **perfect accuracy** with robust generalization, making them ideal candidates for production deployment where reliability is critical.

5. **Real-world Application:** For safety-critical tasks like mushroom classification, the perfect accuracy of ensemble methods is promising. However, additional validation with domain experts and real-world data is essential before deployment.

## Installation and Setup

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Local Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd mushroom-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models:
```bash
cd model
python train_models.py
```

4. Run the Streamlit app:
```bash
cd ..
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Streamlit App Features

### 1. Model Selection Dropdown ✅
- Interactive dropdown to select from 6 different classification models
- Real-time switching between models

### 2. Dataset Upload Option ✅
- Upload custom CSV files with mushroom features
- Supports test data upload for classification
- File validation and error handling

### 3. Evaluation Metrics Display ✅
- Comprehensive metrics for each model:
  - Accuracy
  - AUC Score
  - Precision
  - Recall
  - F1 Score
  - Matthews Correlation Coefficient (MCC)
- Visual metric cards with color-coded indicators

### 4. Confusion Matrix & Classification Report ✅
- Interactive confusion matrix heatmap
- Detailed classification report with per-class metrics
- Support and confidence intervals

### Additional Features:
- 📊 Multi-tab interface for organized content
- 🎨 Beautiful UI with custom styling and mushroom theme
- 📈 Comparative analysis across all models
- 🕸️ Radar charts for visual model comparison
- 📥 Download predictions as CSV
- 🎯 Real-time predictions with visual feedback

## Deployment on Streamlit Community Cloud

### Steps to Deploy:

1. **Prepare Repository:**
   - Ensure all files are committed to GitHub
   - Verify requirements.txt is complete
   - Check that trained models are in the repository

2. **Deploy on Streamlit Cloud:**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub account
   - Click "New App"
   - Select your repository
   - Choose branch: `main`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Post-Deployment:**
   - Wait 2-5 minutes for deployment
   - Test all features on the live app
   - Share the public URL

 


## 📈 Evaluation Metrics Explained

- **Accuracy:** Proportion of correct predictions (TP+TN)/(TP+TN+FP+FN)
- **AUC (Area Under Curve):** Measures the model's ability to distinguish between classes
- **Precision:** Proportion of positive identifications that were correct TP/(TP+FP)
- **Recall (Sensitivity):** Proportion of actual positives identified correctly TP/(TP+FN)
- **F1 Score:** Harmonic mean of precision and recall 2*(Precision*Recall)/(Precision+Recall)
- **MCC (Matthews Correlation Coefficient):** Correlation between observed and predicted classifications

## 📖 References
- UCI Machine Learning Repository - Mushroom Dataset
- Streamlit Documentation  

## 📜 License
Academic coursework for BITS Pilani M.Tech (AIML/DSE).  

## 📬 Contact
**JANET DEVARAJ**  
BITS ID: 2025AA05835  
Email: 2025aa05835@wilp.bits-pilani.ac.in  

---

*Developed with 🍄 for Machine Learning Assignment 2*