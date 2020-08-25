# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# Load the XGboost ressor model
filename = 'big-mart-sale-model_xgb.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        Item_Weight = float(request.form['Item_Weight'])

        Item_Fat_Content=request.form['Item_Fat_Content']

        if (Item_Fat_Content== 'Low Fat'):
            Item_Fat_Content = 3
        elif (Item_Fat_Content== 'Regular'):
            Item_Fat_Content = 2
        else:
            Item_Fat_Content = 1


        Item_Visibility = float(request.form['Item_Visibility'])

        Item_MRP = float(request.form['Item_MRP'])

        Outlet_Establishment_Year = int(request.form['Outlet_Establishment_Year'])

        Outlet_Size  = request.form['Outlet_Size']
        if (Outlet_Size == 'Medium'):
            Outlet_Size = 3
        elif (Outlet_Size == 'Small'):
            Outlet_Size = 2
        else:
            Outlet_Size = 1


        Outlet_Location_Type = int(request.form['Outlet_Location_Type'])

        Outlet_Type = request.form['Outlet_Type']
        if (Outlet_Type == 'Supermarket Type1'):
            Outlet_Type = 1
        elif (Outlet_Type == 'Grocery Store'):
            Outlet_Type = 2
        elif (Outlet_Type == 'Supermarket Type3'):
            Outlet_Type = 3
        else:
            Outlet_Type = 4

        Item_Type_Combined = int(request.form['Item_Type_Combined'])

        data = [Item_Weight,Item_Fat_Content,
            Item_Visibility,Item_MRP,Outlet_Establishment_Year,
            Outlet_Size,Outlet_Location_Type,Outlet_Type,Item_Type_Combined]



        features_value = [np.array(data)]

        features_name = ['Item_Weight','Item_Fat_Content',
        'Item_Visibility','Item_MRP',
        'Outlet_Establishment_Year',
        'Outlet_Size','Outlet_Location_Type',
        'Outlet_Type','Item_Type_Combined']

        df = pd.DataFrame(features_value, columns=features_name)

        myprd = model.predict(df)
        output=round(myprd[0],2)
        return render_template('result.html',prediction = output)


if __name__ == '__main__':
	app.run(debug=True)