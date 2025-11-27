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
1. Check the group chat/docs for the specific City and Date version we agreed on.
2. Download `listings.csv.gz` and `reviews.csv.gz`.
3. Place them into the `data/raw/` folder.
4. **DO NOT UNZIP** (scripts will handle compressed files).
5. **DO NOT PUSH** any CSV files to GitHub.

## 3. Directory Structure
- `data/`: Local data files (ignored by Git).
- `notebooks/`: Experimental work. Naming convention: `01_description_name.ipynb`
- `src/`: Reusable clean code.
  - `data_loader.py`: Use this to load data instead of writing `pd.read_csv` every time.
  - `preprocessing.py`: Shared cleaning logic.
- `models/`: Trained model binaries (ignored by Git).
- `reports/`: Final report and images.

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
- Talha Ubeydullah Gamga
- Aziz Önder
- Buğra Bildiren
- Muhammed Fatih Asan
