# Airbnb Price-Quality Classification Project

## 📊 Dataset

### Download Links
- **San Francisco listings:** [Download here](https://insideairbnb.com/san-francisco)
- **San Diego listings:** [Download here](https://insideairbnb.com/san-diego)

### Place the files in the correct location:
```
data/raw/san_francisco.csv
data/raw/san_diego.csv
```

### Dataset Statistics:
- **San Francisco:** 7,780 listings × 79 features
- **San Diego:** 13,162 listings × 79 features
- **Combined:** 20,942 listings

---

## 📁 Project Directory Structure

```
ML-Project/
│
├── README.md                          # Project overview and instructions
├── .gitignore                         # Specifies files to be ignored by Git
├── requirements.txt                   # List of Python dependencies
│
├── data/                              # DATA STORE (Ignored by Git, keep local)
│   ├── raw/                          # Original, immutable data dump
│   │   ├── san_francisco.csv
│   │   └── san_diego.csv
│   │
│   ├── processed/                    # Cleaned, canonical data sets for modeling
│   │   ├── listings_cleaned.csv
│   │   ├── listings_with_algebraic_features.csv
│   │   ├── listings_with_categorical_encoding.csv
│   │   ├── listings_final_selected_features.csv
│   │   ├── X_train_scaled.csv
│   │   ├── X_test_scaled.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   │
│   └── external/                     # Data from third party sources
│
├── notebooks/                        # JUPYTER NOTEBOOKS
│   ├── week1/                       # Week 1: Data Preparation & Feature Engineering
│   │   ├── 01_data_exploration_omer.ipynb          # T1.1-T1.6: Data prep
│   │   ├── 02_nlp_sentiment_fatih.ipynb            # T1.7-T1.12: NLP
│   │   └── 03_eda_target_emircan.ipynb             # T1.13-T1.18: EDA
│   │
│   ├── week2/                       # Week 2: Model Development & Comparison
│   │   ├── 04_supervised_models_omer.ipynb         # Supervised learning
│   │   ├── 05_unsupervised_models_fatih.ipynb      # Unsupervised learning
│   │   └── 06_ensemble_evaluation_emircan.ipynb    # Ensemble & evaluation
│   │
│   └── week3/                       # Week 3: Advanced Techniques & Final Analysis
│       ├── 07_deep_learning_nlp_omer.ipynb         # Deep learning & BERT
│       ├── 08_optimization_tuning_fatih.ipynb      # Hyperparameter tuning
│       └── 09_interpretability_report_emircan.ipynb # SHAP, LIME, final report
│
├── src/                             # SOURCE CODE (Production ready code)
│   ├── __init__.py                  # Makes src a Python module
│   ├── data_loader.py               # Scripts to download or generate data
│   ├── preprocessing.py             # Scripts to clean data and generate features
│   ├── feature_engineering.py       # Algebraic feature creation
│   ├── nlp_processing.py            # NLP and sentiment analysis functions
│   ├── visualization.py             # Scripts to create common plots
│   └── models.py                    # Scripts to train models and make predictions
│
├── models/                          # SERIALIZED MODELS
│   ├── standard_scaler.pkl          # Fitted StandardScaler
│   ├── xgboost_classifier.pkl       # Trained XGBoost model
│   ├── random_forest.pkl            # Trained Random Forest model
│   └── kmeans_clustering.pkl        # Trained K-Means model
│
├── outputs/                         # OUTPUT FILES
│   ├── reports/                     # Generated reports
│   │   ├── Task_1.3_Algebraic_Features_Report.docx
│   │   ├── eda_report.pdf
│   │   ├── model_performance_report.pdf
│   │   └── final_report.pdf
│   │
│   ├── figures/                     # Generated graphics and figures
│   │   ├── correlation_heatmap.png
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   ├── shap_analysis.png
│   │   ├── feature_importance_comparison.png
│   │   ├── scaling_comparison.png
│   │   └── train_test_split_distribution.png
│   │
│   └── *.csv                        # Analysis results (correlations, VIF, etc.)
│
└── tests/                           # Unit tests for src/ code
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <repository-url>
cd ML-Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets
Place the San Francisco and San Diego CSV files in `data/raw/`

### 4. Run notebooks
Navigate to `notebooks/week1/` and start with `01_data_exploration_omer.ipynb`

---

## 👥 Team Members

- **Omer:** Data preparation, supervised learning, deep learning
- **Fatih:** NLP & sentiment analysis, unsupervised learning, optimization
- **Emircan:** EDA, ensemble methods, interpretability & reporting

---

## 📅 Project Timeline

### Week 1: Data Preparation & Feature Engineering
- ✅ Task 1.1-1.6: Data exploration, cleaning, feature engineering (Omer)
- Task 1.7-1.12: NLP and sentiment analysis (Fatih)
- Task 1.13-1.18: Exploratory data analysis (Emircan)

### Week 2: Model Development & Comparison
- Task 2.1-2.6: Supervised learning models (Omer)
- Task 2.7-2.12: Unsupervised learning models (Fatih)
- Task 2.13-2.18: Ensemble methods and evaluation (Emircan)

### Week 3: Advanced Techniques & Final Analysis
- Task 3.1-3.6: Deep learning and BERT (Omer)
- Task 3.7-3.12: Hyperparameter tuning and optimization (Fatih)
- Task 3.13-3.18: Model interpretability and final report (Emircan)

---

## 🎯 Project Goal

Classify Airbnb listings into value categories (Poor Value, Fair Value, Excellent Value) based on the relationship between price and quality metrics.

### Target Variable
- **value_category:** 3-class classification
  - Poor Value (0): High price, low quality
  - Fair Value (1): Balanced price-quality ratio
  - Excellent Value (2): Low price, high quality

### Key Features (28 selected)
Selected through correlation analysis, VIF testing, and feature importance scoring from 94 engineered features.

---



---

## 📈 Model Performance (To be updated in Week 2)

| Model | Accuracy | F1-Score | Notes |
|-------|----------|----------|-------|
| XGBoost | TBD | TBD | Supervised |
| Random Forest | TBD | TBD | Supervised |
| K-Means | TBD | TBD | Unsupervised |

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, xgboost
- **Deep Learning:** TensorFlow/PyTorch (Week 3)
- **NLP:** NLTK, spaCy, transformers (BERT)
- **Model Interpretation:** SHAP, LIME

---

## 📝 License

This project is for educational purposes as part of a Machine Learning course.

---


---

