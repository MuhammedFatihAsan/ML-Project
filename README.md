# Airbnb Price-Quality Classification Project

## Project Overview

This project analyzes Inside Airbnb datasets using Machine Learning techniques to classify listings into value categories based on price-quality relationships.

- **Target Cities:** San Francisco & San Diego, California
- **Main Goal:** Value Classification - Identifying "Budget Gems" vs "Luxury Ripoffs"
- **Approach:** Supervised & Unsupervised Learning with Advanced NLP
- **Dataset:** 20,942 combined listings (SF: 7,780 + SD: 13,162)

### Research Objectives

- Develop comprehensive feature engineering pipeline (algebraic + NLP features)
- Implement and compare 5+ supervised and 4+ unsupervised ML algorithms
- Apply advanced NLP techniques (sentiment analysis, Word2Vec, TF-IDF)
- Conduct rigorous model evaluation with cross-validation
- Provide actionable insights for Airbnb hosts and guests

---

## Dataset Setup

**Large datasets are not stored in GitHub. Download manually.**

### Instructions:

1. **Download datasets:**
   - [San Francisco listings](https://insideairbnb.com/san-francisco)
   - [San Diego listings](https://insideairbnb.com/san-diego)

2. **Place files in:**
   ```
   data/raw/san_francisco.csv
   data/raw/san_diego.csv
   ```

---

## Project Directory Structure

```
ML-PROJECT/
│
├── README.md
├── .gitignore
├── .gitattributes
├── requirements.txt
│
├── data/
│   ├── raw/                              # Original data (not in Git)
│   │   ├── san_francisco.csv
│   │   └── san_diego.csv
│   │
│   ├── processed/                        # Intermediate processed data
│   │   ├── listings_cleaned_with_target.csv
│   │   ├── listings_landlord_features_only.csv
│   │   ├── listings_with_algebraic_features.csv
│   │   ├── listings_with_categorical_encoding.csv
│   │   ├── listings_with_geo_features.csv
│   │   ├── listings_nlp_features.csv
│   │   ├── nlp_scores.csv
│   │   ├── word2vec_embeddings.csv
│   │   ├── X_train_landlord.csv
│   │   ├── X_test_landlord.csv
│   │   ├── X_train_landlord_scaled.csv
│   │   ├── X_test_landlord_scaled.csv
│   │   ├── y_train_landlord.csv
│   │   ├── y_test_landlord.csv
│   │   ├── *_predictions.csv             # Model predictions
│   │   ├── *_results.csv                 # Model results
│   │   └── *_feature_importance.csv      # Feature importance files
│   │
│   └── finalized/                        # Final production data
│       ├── final_data_with_nlp_score.csv
│       └── numeric_final_data.csv
│
├── notebooks/
│   ├── Finalized/                        # Production-ready notebooks
│   │   ├── data_finalPrep.ipynb
│   │   └── final_model_with_NLP.ipynb
│   │
│   ├── Milestone-1/                            # Data Preparation & Feature Engineering
│   │   ├── ofb_T1.1_data_prep.ipynb
│   │   ├── ofb_T1.2_data_cleaning.ipynb
│   │   ├── ofb_T1.3_algebraic_features.ipynb
│   │   ├── ofb_T1.4_categorical_encoding.ipynb
│   │   ├── ofb_T1.5_Feature_selection.ipynb
│   │   ├── mfa_T1.7_T1.8_nlp_pipeline.ipynb
│   │   ├── mfa_T1.9_T1.10_sentiment_features.ipynb
│   │   ├── mfa_T1.11_T1.12_nlp_tfidf_NlpTablesMerge.ipynb
│   │   ├── mfa_T1.TEST_NLP_Quality_Assurance.ipynb
│   │   ├── eck_T1.13_define_target_variable.ipynb
│   │   ├── eck_T1.14_eda_univariate.ipynb
│   │   ├── eck_T1.15_eda_bivariate.ipynb
│   │   ├── eck_T1.16_geographic_analysis.ipynb
│   │   ├── eck_T1.17_class_distribution_analysis.ipynb
│   │   └── eck_T1.18_EDA_Report.ipynb
│   │
│   ├── Milestone-2/                      # Model Development
│   │   ├── ofb_T2.1_logistic_regression.ipynb
│   │   ├── ofb_T2.2_random_forest.ipynb
│   │   ├── ofb_T2.3_xgboost.ipynb
│   │   ├── ofb_T2.4_svm.ipynb
│   │   ├── ofb_T2.5_MLP_Classifier.ipynb
│   │   ├── ofb_T2.6_Final_Model_Selection.ipynb
│   │   ├── mfa_Unsupervised-Learning-Models.ipynb
│   │   └── Ensemble & Evaluation.ipynb
│   │
│   └── Milestone-3/                      # Advanced Techniques
│       ├── mfa_Advanced_NLP-AND-Feature_Engineering.ipynb
│       ├── numeric_data.ipynb
│       └── Detailed_Report.ipynb
│
├── models/                               # Trained models
│   ├── logistic_regression_landlord.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── xgboost_bayesian_optimized.pkl
│   ├── svm_linear_model.pkl
│   ├── svm_rbf_model.pkl
│   ├── best_mlp_model.pkl
│   ├── standard_scaler.pkl
│   └── standard_scaler_landlord.pkl
│
├── outputs/
│   ├── figures/                          # Visualizations
│   │   ├── algebraic_features_*.png
│   │   ├── bivariate_*.png
│   │   ├── eda_*.png
│   │   ├── correlation_heatmap_landlord.png
│   │   ├── model_comparison_landlord_features.png
│   │   ├── mlp_learning_curve.png
│   │   ├── geo_price_heatmap.png
│   │   └── ...
│   │
│   └── reports/                          # Generated reports
│       ├── feature_reconstruction_report.txt
│       ├── landlord_feature_names.txt
│       ├── T1.2_preprocessing_summary.txt
│       ├── T1.3_algebraic_features_summary.txt
│       ├── T1.4_categorical_encoding_summary.txt
│       ├── model_comparison_summary_landlord.csv
│       ├── EDA_Executive_Summary.md
│       └── *_encoding_map.csv
│
├── src/                                  # Source code modules
├── references/                           # Documentation & notes
└── venv/                                 # Virtual environment (not in Git)
```

---

## Target Variable

**value_category:** 3-class classification based on FP Score (rating/price ratio)
- **0 - Poor Value:** High price, low quality
- **1 - Fair Value:** Balanced price-quality ratio  
- **2 - Excellent Value:** Low price, high quality

### Important: Data Leakage Prevention

Review-based features are used **only for creating labels**, not as model inputs. The model uses only **landlord-controlled features** (27 features) that are available before any reviews exist.

---

## Model Performance

| Model | Accuracy | F1-Score (Weighted) | Type |
|-------|----------|---------------------|------|
| XGBoost | ~95% | ~0.96 | Supervised |
| Random Forest | ~95% | ~0.95 | Supervised |
| MLP Classifier | ~95% | ~0.95 | Supervised |
| SVM (RBF) | ~92% | ~0.93 | Supervised |
| Logistic Regression | ~95% | ~0.95 | Supervised |
| K-Means | - | - | Unsupervised |
| DBSCAN | - | - | Unsupervised |
| GMM | - | - | Unsupervised |

---

## Key Features 

Selected features available to hosts before receiving reviews:

- **Price-related:** price, price_per_bedroom, price_per_bathroom
- **Property:** accommodates, bedrooms, beds, space_efficiency
- **Location:** latitude, longitude, neighbourhood_frequency
- **Availability:** availability_30/60/90/365, minimum_nights, maximum_nights
- **Host:** host_is_superhost, host_identity_verified, host_response_rate
- **Booking:** instant_bookable
- **Encoded:** room_type_*, property_type_label, property_type_frequency
- **NLP:** w2v_score (from listing description)

---

## Team Members

| Name | Role | Tasks |
|------|------|-------|
| **Ömer (ofb)** | Data & Supervised Learning Lead | T1.1-T1.5, T2.1-T2.6 |
| **Muhammed Fatih Asan (mfa)** | NLP & Unsupervised Learning Lead | T1.7-T1.12, Unsupervised Models, Advanced NLP |
| **Emircan (eck)** | EDA & Evaluation Lead | T1.13-T1.18, Ensemble & Evaluation |

---

## Getting Started

```bash
# 1. Clone repository
git clone https://github.com/MuhammedFatihAsan/ML-Project.git
cd ML-Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install notebook filter (prevents merge conflicts)
nbstripout --install

# 5. Download datasets to data/raw/

# 6. Run notebooks in order or use finalized versions
jupyter lab
```

---

## Technologies

- **Python 3.8+**
- **Data:** pandas, numpy, scipy
- **ML:** scikit-learn, xgboost
- **NLP:** NLTK, gensim (Word2Vec), TF-IDF
- **Visualization:** matplotlib, seaborn, plotly
- **Deep Learning:** TensorFlow/Keras (MLP)

---

## Links

- **Repository:** [GitHub](https://github.com/MuhammedFatihAsan/ML-Project)
- **Data Source:** [Inside Airbnb](http://insideairbnb.com/get-the-data/)