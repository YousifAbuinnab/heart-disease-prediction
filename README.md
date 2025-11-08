# Heart Disease Prediction using Machine Learning

## 📋 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection and prediction of heart disease can significantly improve patient outcomes and reduce mortality rates. This project aims to develop machine learning models that can predict the presence of heart disease in patients based on various health metrics and clinical features.

The goal is to build and compare multiple machine learning algorithms to identify the most effective model for heart disease prediction, which can assist healthcare professionals in early diagnosis and treatment planning.

## 📊 Dataset

**Dataset Name:** Heart Disease UCI  
**Source:** [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
**Alternative Link:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

### Dataset Description

The dataset contains 303 patient records with 14 attributes. The target variable indicates the presence of heart disease (0 = no disease, 1-4 = presence of disease). For this binary classification task, we convert the target to binary (0 = no disease, 1 = disease).

### Features

- **age**: Age in years
- **sex**: Gender (Male/Female)
- **cp**: Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (True/False)
- **restecg**: Resting electrocardiographic results (normal, lv hypertrophy, etc.)
- **thalch**: Maximum heart rate achieved
- **exang**: Exercise induced angina (True/False)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (upsloping, flat, downsloping)
- **ca**: Number of major vessels colored by flourosopy
- **thal**: Thalassemia (normal, fixed defect, reversable defect)

## 🔧 Algorithms Used

This project implements and compares three machine learning algorithms:

1. **Logistic Regression**: A linear classification algorithm that models the probability of a binary outcome using a logistic function.

2. **Support Vector Machine (SVM)**: A powerful classification algorithm that finds the optimal hyperplane to separate classes using the RBF kernel.

3. **Neural Network**: A multi-layer perceptron (MLP) classifier with hidden layers (100, 50 neurons) that can learn complex non-linear patterns.

## 📈 Model Performance

The models are evaluated using the following metrics:
- **Accuracy**: Overall correctness of predictions
- **AUC-ROC**: Area Under the Receiver Operating Characteristic Curve
- **Classification Report**: Precision, Recall, and F1-Score for each class

### Results Summary

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 0.8333 | 0.9230 |
| SVM | 0.8167 | 0.9107 |
| Neural Network | 0.8333 | 0.9397 |

**Best Model**: Neural Network (Highest AUC-ROC: 0.9397)

*Note: Results are based on a test set of 60 samples (20% of the dataset). The dataset contains 299 samples after handling missing values.*

## 📊 Visualizations and Insights

The project generates several visualizations to understand the data and model performance:

### 1. **Target Distribution**
- Shows the distribution of heart disease cases vs. no disease cases
- Helps identify class imbalance issues

### 2. **Correlation Heatmap**
- Displays correlations between all features
- Identifies highly correlated features that might affect model performance
- Key insights:
  - Age, cholesterol, and blood pressure show moderate correlations
  - Exercise-induced angina and ST depression are important predictors

### 3. **Age Distribution by Disease Status**
- Compares age distributions between patients with and without heart disease
- **Insight**: Older patients tend to have higher risk of heart disease

### 4. **Key Features Comparison**
- Box plots comparing important features (cholesterol, blood pressure, heart rate, ST depression) across disease status
- **Insights**:
  - Higher cholesterol levels are associated with heart disease
  - Lower maximum heart rate during exercise may indicate heart problems
  - Higher ST depression values correlate with disease presence

### 5. **Model Comparison**
- Bar chart comparing accuracy and AUC-ROC scores across all models
- Helps identify the best-performing model

### 6. **ROC Curves**
- Receiver Operating Characteristic curves for all models
- Shows the trade-off between true positive rate and false positive rate
- Higher AUC indicates better model performance

### 7. **Confusion Matrices**
- Visual representation of model predictions vs. actual values
- Shows true positives, true negatives, false positives, and false negatives
- Helps understand model errors and performance

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ML-SUPERVISED-project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. Ensure the dataset is in the `archive` folder:
   - `archive/heart_disease_uci.csv`

2. Run the main script:
```bash
python heart_disease_prediction.py
```

3. The script will:
   - Load and preprocess the data
   - Train all three models
   - Generate visualizations in the `visualizations` folder
   - Save model results to `model_results.csv`
   - Display performance metrics in the console

### Output Files

- `visualizations/target_distribution.png`: Distribution of target variable
- `visualizations/correlation_heatmap.png`: Feature correlation matrix
- `visualizations/age_distribution.png`: Age distribution by disease status
- `visualizations/features_comparison.png`: Key features comparison
- `visualizations/model_comparison.png`: Model performance comparison
- `visualizations/roc_curves.png`: ROC curves for all models
- `visualizations/confusion_matrices.png`: Confusion matrices for all models
- `model_results.csv`: Detailed model performance metrics

## 📁 Project Structure

```
ML-SUPERVISED-project/
│
├── archive/
│   └── heart_disease_uci.csv          # Dataset
│
├── visualizations/                     # Generated visualizations
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── age_distribution.png
│   ├── features_comparison.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── confusion_matrices.png
│
├── heart_disease_prediction.py         # Main script
├── requirements.txt                    # Python dependencies
├── model_results.csv                   # Model performance results
└── README.md                           # Project documentation
```

## 🔍 Key Insights

1. **Feature Importance**: 
   - Exercise-induced angina, ST depression, and maximum heart rate are strong predictors
   - Age and cholesterol levels also contribute significantly

2. **Model Performance**:
   - All three models achieve good performance (>80% accuracy)
   - Neural Network achieves the highest AUC-ROC (93.97%), indicating best discrimination ability
   - Logistic Regression and Neural Network tie for highest accuracy (83.33%)
   - All models show strong predictive capability with AUC-ROC scores above 91%

3. **Clinical Relevance**:
   - The models can assist in early detection of heart disease
   - High AUC-ROC scores indicate good discrimination between healthy and diseased patients
   - Models can be used as a screening tool to prioritize patients for further testing

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

## 📝 Future Improvements

- Feature engineering and selection
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Cross-validation for more robust evaluation
- Handling class imbalance using SMOTE or other techniques
- Deployment as a web application using Flask/FastAPI
- Integration with electronic health records (EHR) systems

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Created as part of a Machine Learning Supervised Learning project.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Kaggle community for dataset availability and discussions
- Scikit-learn developers for excellent ML tools

---

**Note**: This project is for educational purposes. The models should not be used as the sole basis for medical diagnosis. Always consult healthcare professionals for medical decisions.

