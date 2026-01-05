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
├── README.md
├── requirements.txt                      # Necessary libraries
├── data/
│   ├── raw/                             # Original Airbnb data
│   │   ├── san_francisco.csv
│   │   └── san_diego.csv
│   ├── processed/                        # Intermediate processed data
│   └── finalized/                        # Final production data
├── notebooks/
│   ├── Milestone-1/                      # Data Preparation & Feature Engineering
│   ├── Milestone-2/                      # Model Development
│   └── Milestone-3/                      # Advanced Techniques
│   ├── Finalized/                        
│   │   ├── data_finalPrep.ipynb          # Milestone-1 Combined Tasks
│   │   └── version1_final_model_with_NLP.ipynb      # Final model V1
│   │   └── version2_final_model.ipynb               # Final model V2
├── models/                               # Trained models
├── outputs/
│   ├── figures/                          # Visualizations
│   └── reports/

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