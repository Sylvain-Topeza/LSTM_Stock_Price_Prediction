# LSTM Stock Price Prediction

## Overview
This project implements a Long Short-Term Memory (LSTM) model to predict AAPL stock prices based on historical closing prices. The model is trained using Python and TensorFlow, with a focus on preprocessing financial data, building a robust LSTM architecture, and visualizing the prediction results.

Key Features:
- Data collection via `yfinance`.
- LSTM model for time series forecasting.
- Performance evaluation using Mean Squared Error (MSE).
- Visualization of predicted vs actual prices.

## Results
- **Mean Squared Error on Test Data**: `0.0007619597599841654`
- Below is a sample graph showing actual vs predicted prices:

![Predicted vs Actual Prices](graphs/Actual_vs_predicted_prices.png)

## Installation

### Requirements
- Python 3.12.0
- Libraries:
  - NumPy 2.0.2
  - Matplotlib 3.10.0
  - Pandas 2.2.3
  - SciPy 1.14.1
  - Scikit-learn 1.6.0
  - TensorFlow 2.18.0
  - Yfinance 0.2.51
