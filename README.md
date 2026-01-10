# AI-Based Launch Vehicle Performance Prediction

## Overview
This project applies Machine Learning techniques to predict launch vehicle performance
parameters such as payload capacity, fuel efficiency, and fault detection.

## Problem Statement
Accurate prediction of launch vehicle performance is critical for mission planning
and safety. This system uses data-driven models to estimate key performance metrics.

## Dataset
- Synthetic dataset with 501 samples and 10 launch parameters
- Parameters include thrust, fuel mass, dry mass, burn time, and engine type
- Dataset created using publicly available aerospace references

## Methodology
- Data preprocessing and feature engineering
- Regression models for payload and fuel efficiency prediction
- Classification models for safety and fault detection
- Model evaluation using Accuracy, R² Score, and RMSE

## Models Used
- Random Forest
- Gradient Boosting

## Results
- Achieved approximately 90% accuracy in fuel efficiency prediction and fault detection
- Performance validated using actual vs predicted visualizations

## Technologies Used
- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib
- Streamlit

## Application
- Interactive Streamlit web application
- Upload dataset and visualize predictions in real time
- Displays actual vs predicted graphs for multiple parameters

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
