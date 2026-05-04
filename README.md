# Automotive Coupon Recommendation System

This is a Streamlit web application that predicts whether a driver will accept an in-vehicle coupon based on various demographic and contextual features.

## Features

- Predict coupon acceptance using three different machine learning models:
  - Decision Tree
  - Random Forest
  - Naive Bayes

- Interactive web interface for inputting driver and coupon features

## Dataset

The application uses the "In-Vehicle Coupon Recommendation" dataset, which contains information about drivers and coupon offers.

## Models

The models are trained on the dataset and saved as pickle files for fast inference.

## Deployment

This app is ready for deployment on Streamlit Cloud.

### Main File Path
`app.py`

### Requirements
See `requirements.txt` for the required Python packages.

## Running Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training script (if needed):
   ```
   python train.py
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Files

- `app.py`: Main Streamlit application
- `train.py`: Script to train and save models
- `requirements.txt`: Python dependencies
- `*.pkl`: Saved model files
- `in-vehicle-coupon-recommendation.csv`: Dataset
- Jupyter notebooks: Original analysis notebooks