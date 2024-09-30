# use the Open Meteo API data and predict temperature using a Decision Tree in Python
#Install Required Libraries:pip install pandas and requests and scikit-learn

#Import Libraries
import pandas as pd
import requests
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np

#Fetch Data from the API
url = "https://api.open-meteo.com/v1/forecast?latitude=36.2795&longitude=50.0046&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"

#Send request to the API
response = requests.get(url)
data = response.json()

#Create a DataFrame from the Retrieved Data
#Extract temperature, humidity, and wind speed data
hourly_data = data["hourly"]
data_frame = pd.DataFrame(
    {
        "Temperature": hourly_data["temperature_2m"],
        "Humidity": hourly_data["relative_humidity_2m"],
        "Wind Speed": hourly_data["wind_speed_10m"],
    }
)

#Split Data into Features and Target
x = data_frame[["Humidity", "Wind Speed"]]
y = data_frame["Temperature"]

#Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(x.values, y.values)
tree = DecisionTreeRegressor().fit(X_train, y_train)


#Build the Decision Tree Model:
model = DecisionTreeRegressor()
clf = model.fit(X_train, y_train)

#Predict Temperature
new_data = [[68, 6]]
answer = clf.predict(new_data)
print(answer)
