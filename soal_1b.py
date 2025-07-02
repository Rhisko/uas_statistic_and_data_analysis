import pandas as pd
from sklearn.linear_model import LinearRegression

# Function to read data from a CSV file
def read_data(file_path):
    return pd.read_csv(file_path)

# Function to prepare features (X) and target (y)
def prepare_variables(dataframe):
    X = dataframe[['Waktu_Belajar_Jam', 'Tidur_Jam']]
    y = dataframe['Nilai_Ujian_Matematika']
    return X, y

# Function to train a linear regression model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to display model coefficients and intercept
def display_model_info(model):
    print('Intercept (β₀):', model.intercept_)
    print('Coefficient Waktu_Belajar_Jam (β₁):', model.coef_[0])
    print('Coefficient Tidur_Jam (β₂):', model.coef_[1])

# Function to add predictions to the DataFrame
def add_predictions(dataframe, model, X):
    dataframe['Prediksi'] = model.predict(X)
    return dataframe

# Function to make a new prediction
def make_prediction(model, waktu_belajar, tidur):
    input_data = pd.DataFrame([[waktu_belajar, tidur]], columns=['Waktu_Belajar_Jam', 'Tidur_Jam'])
    prediction = model.predict(input_data)
    print(f'Predicted exam score if studying {waktu_belajar} hours and sleeping {tidur} hours: {prediction[0]:.2f}')

# Function to calculate and display R^2 score
def display_r2_score(model, X, y):
    print('R^2 Score:', model.score(X, y))

# Main function to execute the workflow
def main():
    # Step 1: Read data
    df = read_data('data_nilai_matematika.csv')

    # Step 2: Prepare variables
    X, y = prepare_variables(df)

    # Step 3: Train the model
    model = train_model(X, y)

    # Step 4: Display model information
    display_model_info(model)

    # Step 5: Add predictions to the DataFrame
    df = add_predictions(df, model, X)
    print(df.head())
    # Step 6: Make a new prediction
    waktu_belajar = 2.5
    tidur = 7.0
    make_prediction(model, waktu_belajar, tidur)
    # Step 7: Display R^2 score
    display_r2_score(model, X, y)

# Run the main function
if __name__ == '__main__':
    main()
