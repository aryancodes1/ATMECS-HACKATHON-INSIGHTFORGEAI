import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import seaborn as sns

sns.set_theme()

# Recreate the model architecture
def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(units=256, return_sequences=True),
        keras.layers.LSTM(units=256, return_sequences=False),
        keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load the trained weights
@st.cache_resource
def load_model():
    model = create_model((5, 1))  # Assuming input shape is (5, 1)
    model.load_weights('models/lstm_model_weights.h5')
    return model

# Prepare data function (your original function)
def prepare_data(uploaded_file, product_number) :
    data = pd.read_csv(uploaded_file, index_col=1)
    data = data[1500:]
    data2 = data
    data = data[[f'Q-P{product_number}']]
    data['Cumulative_Sum'] = data[f'Q-P{product_number}'].cumsum()
    data = data[['Cumulative_Sum']]

    # Setting 80 percent data for training
    training_data_len = math.ceil(len(data) * .8)
    training_data_len

    #Splitting the dataset
    train_data = data[:training_data_len].iloc[:,:1]
    test_data = data[training_data_len:].iloc[:,:1]
    print(train_data.shape, test_data.shape)

    # Selecting Open Price values
    dataset_train = train_data.Cumulative_Sum.values
    # Reshaping 1D to 2D array
    dataset_train = np.reshape(dataset_train, (-1,1))

    scaler_train = MinMaxScaler(feature_range=(0,1))
    scaler_test = MinMaxScaler(feature_range=(0,1))
    # scaling dataset
    scaled_train = scaler_train.fit_transform(dataset_train)

    # Selecting Open Price values
    dataset_test = test_data.Cumulative_Sum.values
    # Reshaping 1D to 2D array
    dataset_test = np.reshape(dataset_test, (-1,1))
    # Normalizing values between 0 and 1
    scaled_test = scaler_test.fit_transform(dataset_test)

    return scaled_train, scaled_test, scaler_test, test_data, data2

def split_data(scaled_train, scaled_test, prev=5):
    X_train = []
    y_train = []
    for i in range(prev, len(scaled_train)):
        X_train.append(scaled_train[i-prev:i, 0])
        y_train.append(scaled_train[i, 0])
    
    X_test = []
    y_test = []
    for i in range(prev, len(scaled_test)):
        X_test.append(scaled_test[i-prev:i, 0])
        y_test.append(scaled_test[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    #Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    y_train = np.reshape(y_train, (y_train.shape[0],1))

    # The data is converted to numpy array
    X_test, y_test = np.array(X_test), np.array(y_test)

    #Reshaping
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))

    return X_train, y_train, X_test, y_test

def predict(X_test, regressor, scaler_test, test_data, prev=5):
    y_LSTM = regressor.predict(X_test)
    y_LSTM_O = scaler_test.inverse_transform(y_LSTM)

    y_LSTM_O_reshaped = y_LSTM_O.reshape(-1,)
    predictions = pd.DataFrame({'actual': test_data.Cumulative_Sum[prev:], 'predicted': y_LSTM_O_reshaped})
    predictions['actual_sales'] = predictions['actual'].shift(-1) - predictions['actual']
    predictions['predicted_sales'] = predictions['predicted'].shift(-1) - predictions['predicted']

    return predictions, y_LSTM_O_reshaped

def plot_graph(data2, y_LSTM_O_reshaped, predictions, product_number):
    if data2 is not None:
        try:
            if f'Q-P{product_number}' not in data2.columns:
                st.error("Column 'Q-P1' not found in the uploaded CSV.")
                return

            monthly_train = []
            each_month_train = []

            for i in range(1, len(data2)):
                if i % 30 == 0:
                    monthly_train.append(np.mean(np.array(each_month_train)))
                    each_month_train = []
                else:
                    each_month_train.append(data2['Q-P1'].iloc[i])

            monthly_train_arr = np.round(np.array(monthly_train)).astype(int)

            monthly_test = []
            monthly_preds = []
            each_month_test = []
            each_month_preds = []

            for i in range(1, len(y_LSTM_O_reshaped)):
                if i % 30 == 0:
                    monthly_test.append(np.mean(np.array(each_month_test)))
                    monthly_preds.append(np.mean(np.array(each_month_preds)))
                    each_month_test = []
                    each_month_preds = []
                else:
                    each_month_test.append(predictions['actual_sales'].iloc[i])
                    each_month_preds.append(predictions['predicted_sales'].iloc[i])
                    
            monthly_test_arr = np.array(monthly_test)
            monthly_preds_arr = np.array(monthly_preds)

            # Prepare the data for plotting
            train_plot = monthly_train_arr
            preds_plot = monthly_preds_arr
            train_indices = np.arange(len(monthly_train_arr) + 1)
            test_indices = np.arange(len(monthly_train_arr), len(monthly_train_arr) + len(monthly_test_arr))
            train_plot = np.append(monthly_train_arr, preds_plot[0])

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(15, 6))

            # Initialize empty lines for actual sales and predicted sales
            ax.plot(train_indices, train_plot, label="Past Sales", color="g")
            ax.plot(test_indices, preds_plot, label="Predicted Future Sales", color="brown")

            # Add a vertical line to mark the train-test split
            ax.axvline(x=len(monthly_train_arr), color='r', linestyle='--')

            # Set title and labels
            ax.set_title("Sales Prediction (Monthly Average)")
            ax.set_xlabel('Months')
            ax.set_ylabel('Quantity sold')
            ax.legend()

            # Display the plot in Streamlit
            st.markdown("<h3 style='text-align: center;'>Our Model's Predictions</h3>", unsafe_allow_html=True)
            st.pyplot(fig)

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(15, 6))

            # Initialize empty lines for actual sales and predicted sales
            ax.plot(train_indices, train_plot, label="Past Sales", color="g")
            ax.plot(test_indices, monthly_test_arr, label="Actual Sales", color='blue')

            # Add a vertical line to mark the train-test split
            ax.axvline(x=len(monthly_train_arr), color='r', linestyle='--')

            # Set title and labels
            ax.set_title("Sales Prediction (Monthly Average)")
            ax.set_xlabel('Months')
            ax.set_ylabel('Quantity sold')
            ax.legend()

            # Display the plot in Streamlit
            st.markdown("<h3 style='text-align: center;'>Actual Data</h3>", unsafe_allow_html=True)
            st.pyplot(fig)

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a valid CSV file.")


# Streamlit app
def main():
    st.title('Sales Prediction App')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        product_selection = st.selectbox(
        "Select a Product:",
        options=["Product 1", "Product 2", "Product 3", "Product 4"]
        )

        # Extract the product number from the selection
        product_number = product_selection.split(" ")[-1]

        # Prepare data
        scaled_train, scaled_test, scaler_test, test_data, data2 = prepare_data(uploaded_file, product_number)

        # Load model
        regressor = load_model()

        X_train, y_train, X_test, y_test = split_data(scaled_train, scaled_test)

        predictions, y_LSTM_O_reshaped = predict(X_test, regressor, scaler_test, test_data)

        plot_graph(data2, y_LSTM_O_reshaped, predictions, product_number)

if __name__ == '__main__':
    main()