# Inside Airbnb Analysis Project

## 1. Project Overview
This project aims to analyze the **Inside Airbnb** dataset using Machine Learning techniques.
- **Target City:** [TBD - To Be Decided]
- **Main Goal:** [TBD - e.g., Price Prediction, Commercial Host Detection, Sentiment Analysis]
- **Course:** Machine Learning

## 2. Dataset Setup (Important!)
We are using large datasets from [Inside Airbnb](http://insideairbnb.com/get-the-data/).
**GitHub does not store the data files.** You must download them manually.

**Instructions for Team Members:**


## 3. Directory Structure
### 📂 Project Directory Structure

```text
PROJE_ADI/
│
├── README.md              # Project overview and instructions.
├── .gitignore             # Specifies files to be ignored by Git (e.g., large data).
├── requirements.txt       # List of python dependencies (pip install -r requirements.txt).
│
├── data/                  # DATA STORE (Ignored by Git, keep local)
│   ├── raw/               # Original, immutable data dump. Never modify these files.
│   ├── processed/         # Cleaned, canonical data sets for modeling.
│   └── external/          # Data from third party sources (e.g., GeoJSON maps).
│
├── notebooks/             # JUPYTER NOTEBOOKS
│   ├── 01_eda_names.ipynb          # Exploratory Data Analysis.
│   ├── 02_cleaning_names.ipynb     # Data cleaning and preprocessing.
│   └── 03_model_names.ipynb        # Model training and evaluation.
│   # Naming convention: [Step]_[Topic]_[MemberName].ipynb
│
├── src/                   # SOURCE CODE (Production ready code)
│   ├── __init__.py        # Makes src a Python module.
│   ├── data_loader.py     # Scripts to download or generate data.
│   ├── preprocessing.py   # Scripts to clean data and generate features.
│   ├── visualization.py   # Scripts to create common plots.
│   └── models.py          # Scripts to train models and make predictions.
│
├── models/                # SERIALIZED MODELS
│   └── random_forest_v1.pkl  # Trained and serialized models, model predictions, or summaries.
│
├── reports/               # REPORTS
│   ├── figures/           # Generated graphics and figures to be used in reporting.
│   └── final_report.pdf   # The final academic report.
│
└── references/            # REFERENCES
    └── paper_notes.md     # Notes on related papers and literature.
```

- **data/raw vs data/processed:** We never edit the raw data. If we make a mistake in processing, we can always go back to the original raw files.
- **src/ vs notebooks/:** Notebooks are for experimentation. Once a piece of code (like a cleaning function) works well, we move it to src/ to keep the project clean and reproducible.
- **.gitignore:** Prevents us from uploading large data files (>100MB) that would crash the repository.

## 4. How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```

## 5. Team Members
- Muhammed Fatih Asan
- Ömer
- Emircan
