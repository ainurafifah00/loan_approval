import streamlit as st
import pandas as pd
import numpy as np 
import pickle
from sklearn.ensemble import RandomForestClassifier

# Original data
# https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data
# Cleaned data for this model purposes
# https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv

st.write("""
	# Loan Prediction App

	This app predicts whether loan is approved or not.

	""")

st.sidebar.header('User Input Features')
# st.sidebar.markdown("""
# 	[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# 	""")

# #Upload file (as per format) or user input parameter for prediction
# uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=['csv'])
# if uploaded_file is not None:
# 	input_df = pd.read_csv(uploaded_file)
# else:
def user_input_features():
	gender = st.sidebar.selectbox('Gender', ('male', 'female'), index=None)
	income = st.sidebar.slider('Income', 8000, 7200766, 67048)
	home_ownership = st.sidebar.selectbox('Home Ownership', ('RENT', 'OWN', 'MORTGAGE'), index=None)
	loan_amount = st.sidebar.slider('Loan Amount', 500, 35000, 8000)
	loan_int_rate = st.sidebar.slider('Loan Interest Rate', 5, 20, 11)
	previous_loan = st.sidebar.selectbox('Previous Loan', ('Yes', 'No'), index=None)
	data = {
			'gender': gender,
			'income': income,
			'home_ownership': home_ownership,
			'loan_amount': loan_amount,
			'loan_interest_rate': loan_int_rate,
			'previous_loan': previous_loan
	}
	features = pd.DataFrame(data, index=[0])
	return features

input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
# The sole purpose of combining the entire dataset is for encoding purposes. So it gets standardized & it is easier, less mistake

loan_data_raw = pd.read_csv('data/loan_data_cleaned.csv')
# st.write(loan_data_raw)

loan_data = loan_data_raw.drop(columns=['loan_status'])
df = pd.concat([input_df, loan_data], axis=0) 
#axis 0 means it is concated horizontally. axis 1 is concated vertically
# st.write(df)



# Encoding of ordinal features
encode = ['gender', 'home_ownership', 'previous_loan']
for col in encode:
	#get_dummies returns a dataframe with encoded values. df[col] is the data, prefix=col is the column name
	dummy = pd.get_dummies(df[col], prefix=col)
	df = pd.concat([df,dummy],axis=1)
	del df[col]
df = df[:1] #Selects only the first row (user input data) -> THIS. Bcs you don't need the whole dataset.
# st.write(df)

#Display user input features
st.subheader('User Input Feature')

# if uploaded_file is not None:
# 	st.write(df)
# else:
st.write('Currently using example input parameters (shown below)')
st.write(df)


# Pickle file for machine learning classifier training
# https://www.datacamp.com/tutorial/pickle-python-tutorial
#Load a saved classification model. This part, we need a pickle file (executed & generated in another python file)
load_clf = pickle.load(open('loan_forest_clf.pkl', 'rb'))

#Apply model for our specific input for predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
loan_approval = np.array(['Declined','Approved'])
st.write(loan_approval[prediction])


st.subheader('Prediction Probability')
st.write(prediction_proba)

st.markdown("""[Go back to portfolio](https://ainurafifah00.github.io/)""")



