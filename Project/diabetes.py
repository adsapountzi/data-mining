#Authors : Koukougiannis Dimitris 2537
#		   Sapountzi Athanasia Despooina 2624

#This program contains methods that split and train the preproccesed data by using the best classifier (RandomForestClassifier model)
#creates pickle file and saves the classified trained model as that pickle file 
#it also contains a method that find that file and another method that checks the user input and it predicts the outcome by loading and using the pickle.

import pickle
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import os
import preprocess
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 


def train():

	x = preprocess.x
	y = preprocess.y
	#Split data in train and test (test is 20% of data)
	X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)
	# X_train = preprocess.x_train
	# X_test = preprocess.x_test
	# Y_train = preprocess.y_train
	# Y_test = preprocess.y_test

	# run the RandomForestClassifier model
	model = RandomForestClassifier(random_state=0)
	randomForest = model.fit(X_train, Y_train)
	
	#Save Model As Pickle File
	with open('randomForest.pkl','wb') as m:
		pickle.dump(randomForest,m) #function to store the object data to the file
		test(X_test,Y_test)
	
	
#Test accuracy of the model
def test(X_test,Y_test ):
	with open('randomForest.pkl','rb') as mod:
		p=pickle.load(mod) #load the model
		
		pred =p.predict(X_test) #predict 
		# print (accuracy_score(Y_test,pred))
		# indices = [i for i in range(len(Y_test)) if Y_test[i] != pred[i]]
		# wrong_predictions = preprocess.dataset.iloc[indices,:]
		# print(wrong_predictions)

def find_data_file(filename):
	if getattr(sys, "frozen", False):
		# The application is frozen.
		datadir = os.path.dirname(sys.executable)
	else:
		# The application is not frozen.
		datadir = os.path.dirname(__file__)

	return os.path.join(datadir, filename)

def check_input(data) ->int :
	df=pd.DataFrame(data=data,index=[0])
	#df=df.fillna(' ') #fill missing values
	with open(find_data_file('randomForest.pkl'),'rb') as model:
		p=pickle.load(model) #load model 
	op=p.predict(df) #predict outcome 
	return op[0]
 
		
