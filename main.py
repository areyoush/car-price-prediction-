import seaborn as sns
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


#Importing the datasets.
train_df = pd.read_csv('C:\\Users\\Aayush\\Desktop\\kagglecomp\\data2\\train.csv')
test_df = pd.read_csv('C:\\Users\\Aayush\\Desktop\\kagglecomp\\data2\\test.csv')

#X_TRAIN
x_train = train_df.drop(columns=['id', 'price', 'model'], axis=1)

#Y_TRAIN
y_train = train_df['price']

#X_TEST
x_test = test_df.drop(columns=['id', 'model'], axis=1)

#combining the training and testing datasets for collective preprocessing.
combined_df =pd.concat([x_train, x_test], axis=0)

#splitting the 'engine' feature into three more features:- horsepower, cylinders, engine_displacement.
combined_df['horsepower'] = combined_df['engine'].str.extract(r'(\d+\.?\d*)HP', expand=False)
combined_df['cylinders'] = combined_df['engine'].str.extract(r'(\d)\s*(?:Cylinder|Cylinders)', expand=False)
combined_df['engine_displacement'] = combined_df['engine'].str.extract(r'(\d+\.\d+|\d+)L\b', expand=False)
combined_df[['horsepower', 'cylinders', 'engine_displacement']] = combined_df[['horsepower', 'cylinders', 'engine_displacement']].astype(float)

combined_df = combined_df.drop(columns=['engine'])

#columns to encode:
#   clean_title -> Label Encoding
#   accident -> Label Encoding
#   int_col -> One Hot Encoding
#   ext_col -> One Hot Encoding
#   fuel_type -> One Hot Encoding
#   brand -> One Hot Encoding
#   transmission -> One Hot Encoding

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse_output=False, dtype=int)

#encoding the 'clean_title' column.
combined_df['clean_title'] = label_encoder.fit_transform(combined_df['clean_title'])

#encoding the 'accident' column.
combined_df['accident'] = label_encoder.fit_transform(combined_df['accident'].map({'None reported': 0, 'At least 1 accident or damage reported': 1}))


'''
#encoding the 'int_col' column.
int_col_reshaped = combined_df[['int_col']]
onehot_encoded_int = onehot_encoder.fit_transform(int_col_reshaped)
onehot_encoded_int_df = pd.DataFrame(onehot_encoded_int, columns=onehot_encoder.get_feature_names_out(['int_col']))
combined_df = pd.concat([combined_df.drop('int_col', axis=1), onehot_encoded_int_df], axis=1)

#encoding the 'ext_col' columnn.
ext_col_reshaped = combined_df[['ext_col']]
onehot_encoded_ext = onehot_encoder.fit_transform(ext_col_reshaped)
onehot_encoded_ext_df = pd.DataFrame(onehot_encoded_ext, columns=onehot_encoder.get_feature_names_out(['ext_col']))
combined_df = pd.concat([combined_df.drop('ext_col', axis=1), onehot_encoded_ext_df], axis=1)

#encoding the 'fuel_type' column.
fuel_type_reshaped = combined_df[['fuel_type']]
onehot_encoded_fuel = onehot_encoder.fit_transform(fuel_type_reshaped)
onehot_encoded_fuel_df = pd.DataFrame(onehot_encoded_fuel, columns=onehot_encoder.get_feature_names_out(['fuel_type']))
combined_df = pd.concat([combined_df.drop('fuel_type', axis=1), onehot_encoded_fuel_df], axis=1)

#encoding the 'brand' column.
brand_reshaped = combined_df[['brand']]
onehot_encoded_brand = onehot_encoder.fit_transform(brand_reshaped)
onehot_encoded_brand_df = pd.DataFrame(onehot_encoded_brand, columns=onehot_encoder.get_feature_names_out(['brand']))
combined_df = pd.concat([combined_df.drop('brand', axis=1), onehot_encoded_brand_df], axis=1)

#encoding the 'transmission' column.
transmission_reshaped = combined_df[['transmission']]
onehot_encoded_transmission = onehot_encoder.fit_transform(transmission_reshaped)
onehot_encoded_transmission_df = pd.DataFrame(onehot_encoded_transmission, columns=onehot_encoder.get_feature_names_out(['transmission']))
combined_df = pd.concat([combined_df.drop('transmission', axis=1), onehot_encoded_transmission_df], axis=1)
'''

# Helper function for one-hot encoding and updating the dataframe
def one_hot_encode_and_update(df, column_name, encoder):
    reshaped_col = df[[column_name]]
    onehot_encoded = encoder.fit_transform(reshaped_col)
    onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=encoder.get_feature_names_out([column_name]))
    onehot_encoded_df.index = df.index  # Align indices
    df = pd.concat([df.drop(column_name, axis=1), onehot_encoded_df], axis=1)
    return df

# Encoding the 'int_col' column
combined_df = one_hot_encode_and_update(combined_df, 'int_col', onehot_encoder)

# Encoding the 'ext_col' column
combined_df = one_hot_encode_and_update(combined_df, 'ext_col', onehot_encoder)

# Encoding the 'fuel_type' column
combined_df = one_hot_encode_and_update(combined_df, 'fuel_type', onehot_encoder)

# Encoding the 'brand' column
combined_df = one_hot_encode_and_update(combined_df, 'brand', onehot_encoder)

# Encoding the 'transmission' column
combined_df = one_hot_encode_and_update(combined_df, 'transmission', onehot_encoder)

# Splitting combined_df back into x_train and x_test
x_train = combined_df.iloc[:len(train_df)]
x_test = combined_df.iloc[len(train_df):].reset_index(drop=True)


rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
rf_regressor.fit(x_train, y_train)
y_pred = rf_regressor.predict(x_test)

y_pred_rounded = [round(price, 3) for price in y_pred]


submission = pd.read_csv('C:\\Users\\Aayush\\Desktop\\kagglecomp\\submission.csv')
submission['price'] = y_pred_rounded
submission.to_csv('submission.csv', index=False)

print(submission.shape)




#print(x_train.shape)
#print(x_test.shape)
#print(combined_df.shape)
