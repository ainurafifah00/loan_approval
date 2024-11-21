import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle



loan_data = pd.read_csv('data/loan_data_cleaned.csv')

#Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = loan_data.copy()
target = 'loan_status'
encode = ['gender', 'home_ownership', 'previous_loan']

for col in encode:
	#we use get_dummies for encoding. Then concat to the df & delete the non-encoded column
	dummy = pd.get_dummies(df[col], prefix=col)
	df = pd.concat([df,dummy], axis=1)
	del df[col]

target_mapper = {'Declined':0, 'Approved':1}
def target_encode(val):
	return target_mapper[val]
	val is df['loan_status'] 

# df.apply() allows user to pass a function to every single value of Pandas series
df['loan_status'] = df['loan_status'].apply(target_encode)

#Separate X and y
X = df.drop('loan_status', axis=1)
y = df['loan_status']
# print(y)


#NOW We build a random forest model
clf = RandomForestClassifier()
clf.fit(X,y)


#Saving the model
pickle.dump(clf, open('loan_forest_clf.pkl', 'wb'))
