# Cryptocurrency Price Prediction using Random Forest

This project predicts cryptocurrency prices using a **Random Forest Regressor** model. It allows users to input the cryptocurrency symbol, specify a date range, and visualize the predicted vs. actual prices.

## Features

- Predict cryptocurrency prices using historical data.
- User-friendly interface built with **Streamlit**.
- Model trained using **Random Forest Regressor** from **scikit-learn**.
- Visualize the predicted vs. actual prices with **Matplotlib**.
- Handle user inputs for cryptocurrency symbols and date ranges.

## Technologies Used

- **Python**  
  - `yfinance` for fetching historical cryptocurrency data.
  - `scikit-learn` for machine learning (Random Forest Regressor).
  - `pandas` and `numpy` for data processing.
  - `matplotlib` for data visualization.
  - `streamlit` for building the interactive web app.

## Installation

### Prerequisites

Make sure you have **Python 3.x** installed. You can install all the required libraries using `pip`.

### Install the required dependencies:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn streamlit
```
### Run the Streamlit app
```bash
streamlit run crypto_predictor.py
```
