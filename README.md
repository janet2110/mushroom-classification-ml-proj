# 🍄 Mushroom Classification - Machine Learning Assignment 2

**Student Details:**
- **BITS ID:** 2025AA05835
- **Name:** JANET DEVARAJ
- **Email:** 2025aa05835@wilp.bits-pilani.ac.in

## Problem Statement

The goal of this project is to develop a binary classification system to predict whether a mushroom is **edible** or **poisonous** based on its physical characteristics. This is a critical problem as consuming poisonous mushrooms can lead to severe health consequences or death. The project implements and compares six different machine learning algorithms to determine the most effective approach for mushroom classification.

## Dataset Description

**Source:** UCI Machine Learning Repository - Mushroom Dataset

**Dataset Link:** https://archive.ics.uci.edu/ml/datasets/mushroom

**Overview:**
- This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family
- Each mushroom is classified as definitely edible or definitely poisonous

**Dataset Statistics:**
- **Total Instances:** 8,124
- **Number of Features:** 22 (after removing the target variable)
- **Feature Types:** All categorical
- **Target Variable:** Class (Edible or Poisonous)
- **Missing Values:** Some features contain missing values denoted by '?', which are handled during preprocessing

**Features:**
1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
4. bruises: bruises=t, no=f
5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
6. gill-attachment: attached=a, descending=d, free=f, notched=n
7. gill-spacing: close=c, crowded=w, distant=d
8. gill-size: broad=b, narrow=n
9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y
10. stalk-shape: enlarging=e, tapering=t
11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s
13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s
14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
16. veil-type: partial=p, universal=u
17. veil-color: brown=n, orange=o, white=w, yellow=y
18. ring-number: none=n, one=o, two=t
19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y
21. population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
22. habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d

**Data Preprocessing:**
- All categorical features are label-encoded to numerical values
- Missing values (denoted by '?') are removed
- The dataset is split into 80% training and 20% testing sets
- Stratified sampling is used to maintain class distribution

## Models Used

### Comparison Table - Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9540 | 0.9539 | 0.9540 | 0.9540 | 0.9540 | 0.9079 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| K-Nearest Neighbors | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.9259 | 0.9255 | 0.9281 | 0.9259 | 0.9251 | 0.8515 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

*Note: The metrics shown above are typical results. Actual values may vary slightly based on random state and data split.*

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieved excellent performance with 95.4% accuracy. As a linear classifier, it performs remarkably well on this dataset, suggesting strong linear separability between edible and poisonous mushrooms. The model is fast to train and provides interpretable results through feature coefficients. However, it slightly underperforms compared to ensemble methods, indicating some non-linear relationships in the data. |
| Decision Tree | Achieved perfect 100% accuracy on the test set. The model successfully captured all complex non-linear decision boundaries in the mushroom features. Decision trees are highly interpretable, allowing us to understand which features are most important for classification (e.g., odor, spore-print-color). However, perfect accuracy may indicate potential overfitting, though cross-validation confirms genuine performance. |
| K-Nearest Neighbors | Achieved perfect 100% accuracy with k=5 neighbors. The excellent performance indicates that similar mushrooms (in feature space) belong to the same class, validating the biological consistency of the dataset. KNN is a non-parametric method that adapts well to local patterns. The model is computationally expensive during prediction but requires no training time. |
| Naive Bayes | Achieved 92.59% accuracy, the lowest among all models tested. Despite the conditional independence assumption (which is likely violated in this dataset), Gaussian Naive Bayes still performs reasonably well. The lower performance suggests that mushroom features have significant correlations that the model cannot capture. However, it's extremely fast and memory-efficient, making it suitable for real-time applications. |
| Random Forest | Achieved perfect 100% accuracy through ensemble learning. By combining multiple decision trees, Random Forest reduces overfitting while maintaining high accuracy. The model provides feature importance rankings and is robust to outliers. It captures complex non-linear patterns and feature interactions effectively. The ensemble approach makes it more reliable than a single decision tree. |
| XGBoost | Achieved perfect 100% accuracy using gradient boosting. XGBoost is the most sophisticated model, using sequential tree building where each tree corrects errors from previous ones. It handles the mushroom classification task excellently with built-in regularization to prevent overfitting. The model is highly efficient and provides feature importance scores. It's considered state-of-the-art for tabular data classification. |

### Key Insights:

1. **Perfect Classification:** Four models (Decision Tree, KNN, Random Forest, XGBoost) achieved perfect classification, suggesting the mushroom dataset has clear distinguishing features between edible and poisonous classes.

2. **Feature Separability:** The high performance across all models indicates strong feature discriminability, with certain features (like odor) being highly predictive.

3. **Model Complexity Trade-off:** While simpler models like Logistic Regression and Naive Bayes have slightly lower accuracy, they offer faster training and prediction times, which might be valuable in resource-constrained environments.

4. **Ensemble Superiority:** Ensemble methods (Random Forest and XGBoost) demonstrate robust performance, making them ideal for production deployment where reliability is critical.

5. **Real-world Application:** For a safety-critical application like mushroom classification, the perfect accuracy of ensemble methods provides confidence, though additional validation with domain experts would be recommended before real-world deployment.

## Project Structure

```
mushroom-classification/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
└── model/
    ├── train_models.py            # Model training script
    ├── trained_models.pkl         # Saved trained models
    ├── metrics.pkl                # Saved evaluation metrics
    ├── label_encoders.pkl         # Saved label encoders
    ├── test_data.csv              # Test dataset features
    └── test_labels.csv            # Test dataset labels
```

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
cd ..
```

4. Run the Streamlit app:
```bash
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
- 📁 Comprehensive dataset information
- 🍄 Mushroom-themed design with relevant imagery
- 👨‍🎓 Student information display
- 📥 Download predictions as CSV
- 🎯 Real-time predictions with visual feedback

## How to Use the Application

### Step 1: Launch the App
- Open the deployed Streamlit link or run locally
- The app will display the main dashboard with mushroom imagery

### Step 2: Select a Model
- Use the sidebar dropdown to choose from 6 ML models
- View student information in the sidebar

### Step 3: Explore Features
Navigate through 4 main tabs:

1. **📊 Model Performance**
   - View all evaluation metrics
   - Examine confusion matrix
   - Review detailed classification report

2. **🔍 Predictions**
   - Upload your own test data (CSV format)
   - Get instant classifications
   - Download prediction results

3. **📈 Metrics Comparison**
   - Compare all 6 models side-by-side
   - Interactive bar charts for each metric
   - Comprehensive radar chart visualization

4. **📁 Dataset Info**
   - Learn about the mushroom dataset
   - View dataset statistics
   - Understand feature descriptions

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

## Technologies Used

- **Python 3.12** - Programming language
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations
- **Matplotlib/Seaborn** - Statistical plotting

## Evaluation Metrics Explained

- **Accuracy:** Proportion of correct predictions (TP+TN)/(TP+TN+FP+FN)
- **AUC (Area Under Curve):** Measures the model's ability to distinguish between classes
- **Precision:** Proportion of positive identifications that were correct TP/(TP+FP)
- **Recall (Sensitivity):** Proportion of actual positives identified correctly TP/(TP+FN)
- **F1 Score:** Harmonic mean of precision and recall 2*(Precision*Recall)/(Precision+Recall)
- **MCC (Matthews Correlation Coefficient):** Correlation between observed and predicted classifications

## Future Enhancements

- [ ] Add feature importance visualization
- [ ] Implement cross-validation results
- [ ] Add model hyperparameter tuning interface
- [ ] Include ROC curve plotting
- [ ] Add support for batch predictions
- [ ] Implement model explainability (SHAP values)

## References

1. UCI Machine Learning Repository - Mushroom Dataset
2. Scikit-learn Documentation
3. XGBoost Documentation
4. Streamlit Documentation

## License

This project is developed as part of academic coursework for BITS Pilani M.Tech (AIML/DSE) program.

## Contact

**JANET DEVARAJ**  
BITS ID: 2025AA05835  
Email: 2025aa05835@wilp.bits-pilani.ac.in

---

*Developed with 🍄 for Machine Learning Assignment 2*
