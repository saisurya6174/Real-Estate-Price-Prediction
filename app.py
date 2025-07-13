from flask import Flask, render_template, request
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the enlarged dataset
df = pd.read_csv("real_estate_data.csv")

# One-Hot Encode property_type
encoder = OneHotEncoder(sparse_output=False)
encoded_type = encoder.fit_transform(df[['property_type']])
type_cols = encoder.get_feature_names_out(['property_type'])

df_encoded = pd.concat([
    df.drop(['property_type'], axis=1).reset_index(drop=True),
    pd.DataFrame(encoded_type, columns=type_cols)
], axis=1)

# Features & Target
X = df_encoded.drop(['price', 'city'], axis=1)
y = df_encoded['price']

# Train Model
model = XGBRegressor()
model.fit(X, y)

# Prediction Function
def predict_price(property_type, living_space, rooms, year_built, has_balcony):
    encoded_input = encoder.transform([[property_type]])
    input_data = {
        'living_space': [living_space],
        'rooms': [rooms],
        'year_built': [year_built],
        'has_balcony': [has_balcony]
    }
    input_df = pd.DataFrame(input_data)
    encoded_df = pd.concat([
        input_df,
        pd.DataFrame(encoded_input, columns=type_cols)
    ], axis=1)

    # Fill any missing columns
    for col in X.columns:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

    encoded_df = encoded_df[X.columns]
    prediction = model.predict(encoded_df)[0]
    return round(prediction, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city = request.form['city']
        property_type = request.form['property_type'].lower()
        living_space = float(request.form['living_space'])
        rooms = float(request.form['rooms'])
        year_built = int(request.form['year_built'])
        has_balcony = int(request.form['has_balcony'])

        price = predict_price(property_type, living_space, rooms, year_built, has_balcony)
        return render_template('result.html', price=price)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
