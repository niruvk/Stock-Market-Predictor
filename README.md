# StockWise
The Smart Stock Price Forecasting Tool

# Project Description
"StockWise" is an innovative application that blends financial technology with machine learning to offer predictive insights into stock prices. Aimed at both seasoned investors and newcomers to the stock market, "StockWise" provides a user-friendly interface to forecast stock prices over the next 30 days using historical data. By leveraging advanced LSTM (Long Short-Term Memory) neural networks, "StockWise" delivers precise predictions, enabling users to make informed decisions.

# Key Features
- Real-Time Stock Data: Fetches the latest stock prices using the yfinance library.
- Advanced Prediction Model: Utilizes LSTM neural networks for accurate forecasting.
- Personalized Forecasts: Users can input any stock ticker available on Yahoo Finance to generate predictions.
- Interactive Graphs: Displays predicted stock prices in an easy-to-understand format.
- User-Friendly Interface: Designed with Tkinter, ensuring a smooth user experience.
- Historical Data Analysis: Leverages up to two years of historical data for model training and analysis.

# Technical Specifications:
- Python Libraries: Utilizes yfinance for data collection, keras and TensorFlow for the LSTM model, and matplotlib for plotting predictions.
- Data Handling: Employs pandas for data manipulation and numpy for numerical operations.
- GUI Framework: Built with Tkinter for a seamless desktop application experience.
- Machine Learning: The core prediction model is built using the Keras library with a TensorFlow backend.

# Installation
- Prerequisites: Ensure Python 3.x is installed on your system.
- Library Installation: Install the required Python libraries by running: pip install numpy pandas matplotlib yfinance keras tensorflow
- Download the Application: Clone the repository or download the application files to your local machine.
- Running the Application: Navigate to the application directory and run: python main.py

# Usage
- Start the Application: Launch the application by executing the main.py script.
- Enter Stock Ticker: Input the ticker symbol for the stock you wish to predict (e.g., "AAPL" for Apple Inc.).
- Select Date Range: Optionally, adjust the start and end dates for historical data analysis.
- Predict: Click on "Predict" to generate the 30-day stock price forecast.
- View Results: The predicted stock prices will be displayed in an interactive graph.

# Credits
- Niranjan Vijaya Krishnan (nv2608@princeton.edu)
- Amal Ronak (amal.ronak@psu.edu)
- Assisted by ChatGPT
