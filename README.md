# Airbnb Price-Quality Classification Project

## 📋 Project Overview

This project aims to analyze the Inside Airbnb dataset using Machine Learning techniques to classify listings into value categories based on price-quality relationships.

- **Target Cities:** San Francisco & San Diego, California
- **Main Goal:** Value Classification (FP Score) - Identifying "Budget Gems" vs "Luxury Ripoffs"
- **Approach:** Supervised & Unsupervised Learning with Advanced NLP
- **Course:** Machine Learning

### Research Objectives

- Develop comprehensive feature engineering pipeline (algebraic + NLP features)
- Implement and compare 5+ supervised and 4+ unsupervised ML algorithms
- Apply advanced NLP techniques (sentiment analysis, BERT, Word2Vec)
- Conduct rigorous model evaluation with multiple metrics
- Provide model interpretability through SHAP and LIME
- Generate actionable business insights for Airbnb hosts and guests

---

## 📊 Dataset Setup (Important!)

⚠️ **We are using large datasets from Inside Airbnb. GitHub does not store the data files. You must download them manually.**

### Instructions for Team Members:

**1. Download the datasets:**
- **San Francisco listings:** [Download here](https://insideairbnb.com/san-francisco)
- **San Diego listings:** [Download here](https://insideairbnb.com/san-diego)

**2. Place the files in the correct location:**
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
│   ├── raw/                          # Original, immutable data dump. Never modify these files.
│   │   ├── san_francisco.csv
│   │   └── san_diego.csv
│   │
│   ├── processed/                    # Cleaned, canonical data sets for modeling
│   │   ├── listings_cleaned.csv
│   │   ├── listings_with_nlp.csv
│   │   ├── listings_with_algebraic_features.csv
│   │   ├── listings_with_categorical_encoding.csv
│   │   ├── listings_final_selected_features.csv
│   │   ├── X_train_scaled.csv
│   │   ├── X_test_scaled.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   │
│   └── external/                     # Data from third party sources (e.g., GeoJSON maps)
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
│   # Naming convention: [Step]_[Topic]_[MemberName].ipynb
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
│   ├── kmeans_clustering.pkl        # Trained K-Means model
│   └── model_metadata.json          # Model performance metrics
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
└── references/                      # REFERENCES
    ├── paper_notes.md               # Notes on related papers and literature
    └── project_roadmap.md           # Project timeline and task distribution
```

**Key Points:**
- `data/raw` vs `data/processed`: We never edit the raw data. If we make a mistake in processing, we can always go back to the original raw files.
- `src/` vs `notebooks/`: Notebooks are for experimentation. Once a piece of code works well, we move it to `src/` to keep the project clean and reproducible.
- `.gitignore`: Prevents us from uploading large data files (>100MB) and model files that would crash the repository.

---

## 🚀 Getting Started

### Initial Setup:

**1. Clone the repository:**
```bash
git clone https://github.com/MuhammedFatihAsan/ML-Project.git
cd ML-Project
```

**2. Create the directory structure:**
```bash
# Windows
mkdir data\raw data\processed data\external notebooks\week1 notebooks\week2 notebooks\week3 src models outputs\reports outputs\figures references

# Mac/Linux
mkdir -p data/raw data/processed data/external notebooks/week1 notebooks/week2 notebooks/week3 src models outputs/reports outputs/figures references
```

**3. Download datasets and place them in `data/raw/`**

**4. Install requirements:**
```bash
pip install -r requirements.txt
```

**5. Create your branch:**
```bash
git checkout -b week1-omer  # or week1-fatih, week1-emircan
```

### Running the Project:

**1. Start Jupyter Lab:**
```bash
jupyter lab
```

**2. Open your notebook in `notebooks/week1/` and start working!**

**3. Commit your work:**
```bash
git add .
git commit -m "Complete T1.X: Task description"
git push origin week1-omer
```

---

## 🎯 Project Goal

Classify Airbnb listings into value categories (Poor Value, Fair Value, Excellent Value) based on the relationship between price and quality metrics.

### Target Variable
- **value_category:** 3-class classification
  - **Poor Value (0):** High price, low quality
  - **Fair Value (1):** Balanced price-quality ratio
  - **Excellent Value (2):** Low price, high quality

### Key Features (28 selected)
Selected through correlation analysis, VIF testing, and feature importance scoring from 94 engineered features.

---

## 📅 Project Timeline (3 Weeks)

### Week 1: Data Preparation & Feature Engineering
**Focus:** Establish data foundation, create features, perform EDA

| Member | Role | Tasks |
|--------|------|-------|
| Ömer | Data Preparation & Feature Engineering Lead | T1.1-T1.6: Data exploration, cleaning, algebraic features, encoding, feature selection, train-test split |
| Fatih | NLP & Sentiment Analysis Lead | T1.7-T1.12: Text extraction, preprocessing, sentiment analysis, TF-IDF, feature merging |
| Emircan | Target Variable Definition & EDA Lead | T1.13-T1.18: FP Score definition, univariate/bivariate analysis, geographic analysis, EDA report |

**Week 1 Deliverables:**
- ✅ `listings_cleaned.csv` - Cleaned dataset with engineered features
- ✅ `listings_with_nlp.csv` - Dataset with NLP sentiment features
- ✅ `X_train_scaled.csv`, `X_test_scaled.csv`, `y_train.csv`, `y_test.csv` - Train-test splits
- ✅ `EDA_Report.pdf` - Comprehensive exploratory data analysis
- ✅ `Feature_Engineering_Documentation.md` - All created features explained
- ✅ `Correlation_Heatmap.png` - Feature correlation visualization

### Week 2: Model Development & Comparison
**Focus:** Implement ML algorithms, compare performance

| Member | Role | Tasks |
|--------|------|-------|
| Ömer | Supervised Learning Models Lead | T2.1-T2.6: Logistic Regression, Random Forest, XGBoost, SVM, Neural Networks, model comparison |
| Fatih | Unsupervised Learning Models Lead | T2.7-T2.12: K-Means, Hierarchical, DBSCAN, GMM, PCA+Clustering, comparison |
| Emircan | Ensemble Methods & Evaluation Lead | T2.13-T2.18: Voting Classifier, Stacking, cross-validation, confusion matrices, ROC curves, performance report |

**Week 2 Deliverables:**
- ✅ Trained models saved as `.pkl` files (9+ models)
- ✅ `Model_Performance_Comparison.csv` - All metrics in tabular format
- ✅ `Confusion_Matrices.png` - Visual comparison of all models
- ✅ `ROC_Curves.png` - ROC-AUC comparison visualization
- ✅ `Cluster_Visualization_tSNE.png` - 2D cluster visualization
- ✅ `Feature_Importance_Analysis.png` - Top features from tree models
- ✅ `Model_Performance_Report.pdf` - Comprehensive evaluation report

### Week 3: Advanced Techniques & Final Analysis
**Focus:** Deep learning, optimization, interpretability, reporting

| Member | Role | Tasks |
|--------|------|-------|
| Ömer | Deep Learning & Advanced NLP Lead | T3.1-T3.6: BERT sentiment, Word2Vec, LDA, Deep Neural Networks, LSTM, feature integration |
| Fatih | Model Optimization & Tuning Lead | T3.7-T3.12: Bayesian optimization, RandomSearch, SMOTE, feature selection, learning curves, calibration |
| Emircan | Model Interpretability & Analysis Lead | T3.13-T3.18: SHAP analysis, LIME, partial dependence plots, error analysis, business insights, final report |

**Week 3 Deliverables:**
- ✅ `BERT_sentiment_features.csv` - Advanced NLP features
- ✅ Optimized models with best hyperparameters
- ✅ `SHAP_Analysis.html` - Interactive SHAP visualizations
- ✅ `Learning_Curves.png` - Model learning behavior analysis
- ✅ `Error_Analysis_Report.pdf` - Detailed misclassification analysis
- ✅ `Business_Insights_Presentation.pdf` - Actionable recommendations
- ✅ `Final_Research_Report.pdf` - Complete project documentation




---

## 📈 Model Performance (To be updated in Week 2)

| Model | Accuracy | F1-Score | Notes |
|-------|----------|----------|-------|
| XGBoost | TBD | TBD | Supervised |
| Random Forest | TBD | TBD | Supervised |
| SVM | TBD | TBD | Supervised |
| Neural Network | TBD | TBD | Supervised |
| K-Means | TBD | TBD | Unsupervised |
| Hierarchical | TBD | TBD | Unsupervised |
| DBSCAN | TBD | TBD | Unsupervised |
| GMM | TBD | TBD | Unsupervised |

---

## 🛠️ Technologies & Tools

### Programming
- **Python 3.8+**

### Core Libraries
- **Data Processing:** pandas, numpy, scipy
- **Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn, xgboost, lightgbm
- **Deep Learning:** TensorFlow, Keras, PyTorch
- **NLP:** NLTK, spaCy, transformers (Hugging Face), gensim
- **Interpretability:** SHAP, LIME
- **Optimization:** Optuna, Hyperopt
- **Imbalanced Learning:** imbalanced-learn (SMOTE, ADASYN)

### Development Environment
- Jupyter Notebook / Jupyter Lab
- VS Code (with Python & Jupyter extensions)
- Git for version control
- GitHub for collaboration

---

## 👥 Team Members & Roles

| Name | Role | Responsibilities |
|------|------|------------------|
| **Ömer** | Member 1 - Data & Deep Learning Lead | Data cleaning, feature engineering, supervised models, deep learning |
| **Muhammed Fatih Asan** | Member 2 - NLP & Optimization Lead | NLP processing, sentiment analysis, unsupervised models, hyperparameter tuning |
| **Emircan** | Member 3 - EDA & Evaluation Lead | Target definition, EDA, ensemble methods, interpretability, final report |

---

## 🔄 Git Workflow

### Branch Strategy:
- `main` - Production-ready code
- `week1-omer`, `week1-fatih`, `week1-emircan` - Individual work branches
- `week2-omer`, `week2-fatih`, `week2-emircan` - Week 2 branches
- `week3-omer`, `week3-fatih`, `week3-emircan` - Week 3 branches

### Workflow:
```bash
# 1. Create your branch
git checkout -b week1-omer

# 2. Do your work

# 3. Commit regularly
git add .
git commit -m "Descriptive message"

# 4. Push to GitHub
git push origin week1-omer

# 5. Create Pull Request when week is complete
# 6. Team reviews and merges to main
```

---

## ✅ Success Metrics

- ✅ Minimum 5 supervised and 4 unsupervised algorithms implemented
- ✅ NLP sentiment analysis successfully integrated
- ✅ At least 10 engineered features created
- ✅ Best model achieves >70% accuracy on test set
- ✅ Comprehensive model interpretability analysis (SHAP + LIME)
- ✅ Clear performance comparison showing which approach works best
- ✅ Actionable business insights generated
- ✅ Complete documentation with reproducible code
- ✅ Final report demonstrates research-level rigor

---

## 🔗 Project Links

- **GitHub Repository:** [https://github.com/MuhammedFatihAsan/ML-Project](https://github.com/MuhammedFatihAsan/ML-Project)
- **Project Board:** [https://github.com/users/MuhammedFatihAsan/projects/6/views/1](https://github.com/users/MuhammedFatihAsan/projects/6/views/1)
- **Inside Airbnb Data Source:** [http://insideairbnb.com/get-the-data/](http://insideairbnb.com/get-the-data/)

---

## 📚 References

- Inside Airbnb: [http://insideairbnb.com/](http://insideairbnb.com/)
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- SHAP Documentation: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)
- Hugging Face Transformers: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)

---



