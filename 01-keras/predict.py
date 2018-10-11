#%%
from keras.models import load_model

import common

# Load the previous state of the model
model = load_model(common.model_file_name)

#Some inputs to predict
# 'x'
values = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
]

# Use the model to predict the price for a house
y_predicted = model.predict(values)

for i in range(0, len(values)):
    print(values[i], y_predicted[i])
