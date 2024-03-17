import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import datetime
from threading import Thread
import queue
from datetime import timedelta

# Define your LSTM model building function
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to fetch and process stock data
def fetch_and_process_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    return scaled_data, scaler

# Function to create dataset for training
def create_dataset(data, time_step=60):
    X_train, Y_train = [], []
    for i in range(time_step, len(data)):
        X_train.append(data[i-time_step:i, 0])
        Y_train.append(data[i, 0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    return X_train, Y_train

# GUI application
class StockPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Price Predictor")
        self.geometry("800x600")

        # Queue for inter-thread communication
        self.queue = queue.Queue()

        # Inputs frame
        self.frame_inputs = ttk.Frame(self)
        self.frame_inputs.pack(fill=tk.X)

        self.ticker_symbol = tk.StringVar()
        self.start_date = tk.StringVar(value="2022-01-01")
        self.end_date = tk.StringVar(value=str(datetime.datetime.now().date()))

        ttk.Label(self.frame_inputs, text="Ticker Symbol:").pack(side=tk.LEFT)
        ttk.Entry(self.frame_inputs, textvariable=self.ticker_symbol).pack(side=tk.LEFT)

        ttk.Label(self.frame_inputs, text="Start Date:").pack(side=tk.LEFT)
        ttk.Entry(self.frame_inputs, textvariable=self.start_date).pack(side=tk.LEFT)

        ttk.Label(self.frame_inputs, text="End Date:").pack(side=tk.LEFT)
        ttk.Entry(self.frame_inputs, textvariable=self.end_date).pack(side=tk.LEFT)

        ttk.Button(self.frame_inputs, text="Predict", command=self.start_prediction).pack(side=tk.LEFT)
        
        # Status label for waiting message
        self.status_label = ttk.Label(self.frame_inputs, text="")
        self.status_label.pack(side=tk.LEFT)

        # Plot frame
        self.frame_plot = ttk.Frame(self)
        self.frame_plot.pack(fill=tk.BOTH, expand=True)

        # Start the queue processing
        self.process_queue()

    def predict_prices(self):
        try:
            ticker = self.ticker_symbol.get()
            start_date = self.start_date.get()
            end_date = self.end_date.get()
            scaled_data, scaler = fetch_and_process_data(ticker, start_date, end_date)
            X_train, Y_train = create_dataset(scaled_data)
            model = build_model((X_train.shape[1], 1))
            # Training with minimal epochs for demonstration
            model.fit(X_train, Y_train, batch_size=1, epochs=1, verbose=0)

            # Predict using the last 60 days for next 30 days
            test_data = scaled_data[-60:]
            predictions = []
            for _ in range(30):
                x_test = test_data[-60:].reshape(1, 60, 1)
                pred = model.predict(x_test, verbose=0)
                test_data = np.append(test_data, pred[0]).reshape(-1, 1)
                predictions.append(pred[0])

            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            self.queue.put(predictions)
        except Exception as e:
            self.queue.put(e)

    def start_prediction(self):
        """Start the prediction in a separate thread and display waiting message."""
        self.status_label.config(text="Computing predictions, please wait...")
        threading_task = Thread(target=self.predict_prices, daemon=True)
        threading_task.start()

    def process_queue(self):
        try:
            while not self.queue.empty():
                message = self.queue.get_nowait()
                if isinstance(message, Exception):
                    messagebox.showerror("Error", str(message))
                else:
                    self.plot_predictions(message)
                self.status_label.config(text="")
        except queue.Empty:
            pass
        finally:
            # Check the queue again after a short delay
            self.after(100, self.process_queue)

    def plot_predictions(self, predictions):
        # Clear previous plot
        for widget in self.frame_plot.winfo_children():
            widget.destroy()
    
        # Generate future dates for x-axis
        last_date = datetime.datetime.strptime(self.end_date.get(), '%Y-%m-%d')
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

        # Plotting the predictions
        fig, ax = plt.subplots()
        ticker = self.ticker_symbol.get().upper()  # Ensure ticker symbol is uppercase for consistency
        ax.plot(future_dates, predictions, label='Predicted Stock Prices')
        ax.legend()
        ax.set_title(f"{ticker} 30-Day Stock Price Forecast")  # Include stock name in title
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        # Format the dates on the x-axis
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    app = StockPredictorApp()
    app.mainloop()
