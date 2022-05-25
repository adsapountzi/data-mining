#Authors : Koukougiannis Dimitris 2537
#		   Sapountzi Athanasia Despooina 2624



#This program uses EDA analysis, in order to visualize data and take care of the outliers.
#After EDA, the program implements data preprocessing, splits data and performs different classification models to determine Accuracy, Precision, Recall and F1 score and confusion matrix . 
#For visualation of the data and analysis methods we use plt.show() after each plot.  
#In this file we put plt.show() and print() in comments because we the preprocess.py is imported at the other .py files, so we do not want to display them.  

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
from sklearn.model_selection import train_test_split

# read dataset
dataset = pd.read_csv('diabetes.csv')

#diplay first 5 rows of the dataset
#print(dataset.head())

#check for NULL values 
# print(dataset.info())  


# Check if there are any NULL values.
# print(dataset.head())
# print(dataset.isnull().sum())
# print(dataset.describe())

### **Heat Map Correlation** 

# plt.figure(figsize=(10,10))
sns.heatmap(dataset.corr(), cmap="YlGnBu", annot= True,)
# plt.show()


# ## **Pie Chart** 

sns.set(style="whitegrid")
labels = ['Healthy', 'Diabetic']
sizes = dataset['Outcome'].value_counts(sort = True)

colors = ["lightblue","red"]
explode = (0.05,0) 
 
# plt.figure(figsize=(7,7))
# plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,)

# plt.title('Number of diabetes in the dataset')
# plt.show()

# df = dataset
##set_theme didnt work we use set_
sns.set(style="darkgrid")
preg_bef = dataset.Pregnancies
sns.boxplot(x= dataset.Pregnancies)
# plt.xlim([-1,15])
# plt.title("Box Plot before Median Imputation")
# plt.show()


#we take care of the outliers by using iqr and setting the upper and lower limit.
#If data point is not included in (Lower_tail, Upper_tail), then it is an outlier and we replace its value with the median of the current column. 
#we do this for each column expept for the last column that is the outcome.

q1 = dataset.Pregnancies.quantile(0.25)
q3 = dataset.Pregnancies.quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(dataset.Pregnancies)
for i in dataset.Pregnancies:
    if i > Upper_tail or i < Lower_tail:
            dataset.Pregnancies = dataset.Pregnancies.replace(i, med)
sns.boxplot(x= dataset.Pregnancies)
# plt.xlim([-1,15])
# plt.title("Box Plot after Median Imputation")
# plt.show()   


# ### Glucose 

sns.set(style="darkgrid")
glucose_bef = dataset.Glucose
sns.boxplot(x= dataset.Glucose)
# plt.xlim([-2,210])
# plt.title("Box Plot before Median Imputation")
# plt.show()

q1 = dataset.Glucose.quantile(0.25)
q3 = dataset.Glucose.quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(dataset.Glucose)
for i in dataset.Glucose:
    if i > Upper_tail or i < Lower_tail:
            dataset.Glucose = dataset.Glucose.replace(i, med)
sns.boxplot(x= dataset.Glucose)
# plt.xlim([-2,210])
# plt.title("Box Plot after Median Imputation")
# plt.show()   


# ### Blood Pressure 

sns.set(style="darkgrid")
blood_bef = dataset.BloodPressure
sns.boxplot(x= dataset.BloodPressure)
# plt.xlim([-1,140])
# plt.title("Box Plot before Median Imputation")
# plt.show()

q1 = dataset.BloodPressure.quantile(0.25)
q3 = dataset.BloodPressure.quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(dataset.BloodPressure)
for i in dataset.BloodPressure:
    if i > Upper_tail or i < Lower_tail:
            dataset.BloodPressure = dataset.BloodPressure.replace(i, med)
sns.boxplot(x= dataset.BloodPressure)
# plt.xlim([-1,140])
# plt.title("Box Plot after Median Imputation")
# plt.show()   


# ### Skin Thickness 

sns.set(style="darkgrid")
skin_bef = dataset.SkinThickness
sns.boxplot(x= dataset.SkinThickness)
# plt.xlim([-3,105])
# plt.title("Box Plot before Median Imputation")
# plt.show()

q1 = dataset.SkinThickness.quantile(0.25)
q3 = dataset.SkinThickness.quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(dataset.SkinThickness)
for i in dataset.SkinThickness:
    if i > Upper_tail or i < Lower_tail:
            dataset.SkinThickness = dataset.SkinThickness.replace(i, med)
sns.boxplot(x= dataset.SkinThickness)
# plt.xlim([-3,105])
# plt.title("Box Plot after Median Imputation")
# plt.show()   


# ### Insulin 

sns.set(style="darkgrid")
Insulin_bef = dataset.Insulin
sns.boxplot(x= dataset.Insulin)
# plt.xlim([-10,905])
# plt.title("Box Plot before Median Imputation")
# plt.show()

q1 = dataset.Insulin.quantile(0.25)
q3 = dataset.Insulin.quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(dataset.Insulin)
for i in dataset.Insulin:
    if i > Upper_tail or i < Lower_tail:
            dataset.Insulin = dataset.Insulin.replace(i, med)
sns.boxplot(x= dataset.Insulin)
# plt.xlim([-10,905])
# plt.title("Box Plot after Median Imputation")
# plt.show()   


# ### BMI 

sns.set(style="darkgrid")
BMI_bef = dataset.BMI
sns.boxplot(x= dataset.BMI)
# plt.xlim([-3,75])
# plt.title("Box Plot before Median Imputation")
# plt.show()

q1 = dataset.BMI.quantile(0.25)
q3 = dataset.BMI.quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(dataset.BMI)
for i in dataset.BMI:
    if i > Upper_tail or i < Lower_tail:
            dataset.BMI = dataset.BMI.replace(i, med)
sns.boxplot(x= dataset.BMI)
# plt.xlim([-3,75])
# plt.title("Box Plot after Median Imputation")
# plt.show()   


# ### Diabetes Pedigree Function 

sns.set(style="darkgrid")
pedigree_bef = dataset.DiabetesPedigreeFunction
sns.boxplot(x= dataset.DiabetesPedigreeFunction)
# plt.title("Box Plot before Median Imputation")
# plt.show()

q1 = dataset.DiabetesPedigreeFunction.quantile(0.25)
q3 = dataset.DiabetesPedigreeFunction.quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(dataset.DiabetesPedigreeFunction)
for i in dataset.DiabetesPedigreeFunction:
    if i > Upper_tail or i < Lower_tail:
            dataset.DiabetesPedigreeFunction = dataset.DiabetesPedigreeFunction.replace(i, med)
sns.boxplot(x= dataset.DiabetesPedigreeFunction)
# plt.xlim([0,1.5])
# plt.title("Box Plot after Median Imputation")
# plt.show()   


# ### Age 

sns.set(style="darkgrid")
Age_bef = dataset.Age
sns.boxplot(x= dataset.Age)
# plt.title("Box Plot before Median Imputation")
# plt.show()

q1 = dataset.Age.quantile(0.25)
q3 = dataset.Age.quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(dataset.Age)
#Age_bef = dataset.Age
for i in dataset.Age:
    if i > Upper_tail or i < Lower_tail:
            dataset.Age = dataset.Age.replace(i, med)
sns.boxplot(x= dataset.Age)
# plt.title("Box Plot after Median Imputation")
# plt.show()   


# df1 = df[df.columns[0:7]]
# df2 = dataset[dataset.columns[0:7]]

# plt.show()

# ### **Pair Plot**


sns.pairplot(data=dataset,hue='Outcome',diag_kind='kde')
# plt.show()

#     Data Preprocessing 

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# x
# y
    
#  --->   Splitting Data into Train and Test Set    <---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

# print("Number of transactions x_train dataset: ", x_train.shape)
# print("Number of  transactions y_train dataset: ", y_train.shape)
# print("Number of transactions x_test dataset: ", x_test.shape)
# print("Number of transactions y_test dataset: ", y_test.shape)


# We are using different classification models to determine Accuracy, Precision, Recall and F1 score.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score

models = []
# models.append(['Logistic Regression', LogisticRegression(random_state=0)])
models.append(['GaussianNB', GaussianNB()])
models.append(['Decision Tree', DecisionTreeClassifier(random_state=0)]) 
models.append(['Random Forest', RandomForestClassifier(random_state=0)]) 

lst_1= []

for m in range(len(models)):
    lst_2= []
    model = models[m][1]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)  #Confusion Matrix
    precision = precision_score(y_test, y_pred)  #Precision Score
    recall = recall_score(y_test, y_pred)  #Recall Score
    f1 = f1_score(y_test, y_pred)  #F1 Score
    # print(models[m][0],':')
    # print('')
    # print(cm)
   
    
    # print('Accuracy Score is: ',accuracy_score(y_test, y_pred))
    # print('Precision score is'.format(precision))
    # print('Recall: {:.2f}'.format(recall))
    # print('F1: {:.2f}'.format(f1))
    # print('True negative is: ',  cm[0][0])
    # print('False negative is: ',  cm[1][0])
    # print('True positive is: ',  cm[1][1])
    # print('False positive is: ',  cm[0][1])
    # print('-----------------------------------')
    # print('')
    lst_2.append(models[m][0])
    lst_2.append((accuracy_score(y_test, y_pred))*100) 
    lst_2.append(precision)
    lst_2.append(recall)
    lst_2.append(f1)
    lst_2.append(cm[0][0])
    lst_2.append(cm[1][0])
    lst_2.append(cm[1][1])
    lst_2.append(cm[0][1])
    lst_1.append(lst_2)


df2 = pd.DataFrame(lst_1, columns= ['Model', 'Accuracy %', 'Precision', 'Recall', 'F1','True negative', 'False negative', 'True positive', 'False positive'])
df2.sort_values(by= ['Accuracy %'], inplace= True, ascending= False)

# print(df2.to_string())

# ## **GaussianNB** 

#Fitting GaussianNB Model
#Fitting GaussianNB Model
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
y_prob = classifier.predict_proba(x_test)[:,1]
cm = confusion_matrix(y_test, y_pred)

# print(classification_report(y_test, y_pred))
# print('Accuracy Score: ',accuracy_score(y_test, y_pred))

# Visualizing Confusion Matrix
# plt.figure(figsize = (6, 6))
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}, 
            yticklabels = ['Healthy', 'Diabetic'], xticklabels = ['Predicted Healthy', 'Predicted Diabetic'])
# plt.yticks(rotation = 0)
# plt.show()

#Precision Recall Curve
average_precision = average_precision_score(y_test, y_prob)
disp = plot_precision_recall_curve(classifier, x_test, y_test)
# plt.title('Precision-Recall Curve')
# plt.show()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
y_prob = classifier.predict_proba(x_test)[:,1]
cm = confusion_matrix(y_test, y_pred)

# print('GaussianNB classification report:')
# print(classification_report(y_test, y_pred))
# print('Accuracy Score: ',accuracy_score(y_test, y_pred))

# Visualizing Confusion Matrix
# plt.figure(figsize = (6, 6))
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}, 
            yticklabels = ['Healthy', 'Diabetic'], xticklabels = ['Predicted Healthy', 'Predicted Diabetic'])
# plt.yticks(rotation = 0)
# plt.show()

## **Decision Tree**
# Fitting DecisionTreeClassifier Model

classifier = DecisionTreeClassifier(random_state= 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
y_prob = classifier.predict_proba(x_test)[:,1]
cm = confusion_matrix(y_test, y_pred)
# print('DecisionTreeClassifier classification report:')
# print(classification_report(y_test, y_pred))
# print('Accuracy Score: ',accuracy_score(y_test, y_pred))
 
# Visualizing Confusion Matrix
# plt.figure(figsize = (6, 6))
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}, 
            yticklabels = ['Healthy', 'Diabetic'], xticklabels = ['Predicted Healthy', 'Predicted Diabetic'])
# plt.yticks(rotation = 0)
# plt.show()

## **Random Forest**
# Fitting RandomForestClassifier Model
classifier = RandomForestClassifier( random_state= 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

y_prob = classifier.predict_proba(x_test)[:,1]
cm = confusion_matrix(y_test, y_pred)

# print('RandomForestClassifier classification report:')
# print(classification_report(y_test, y_pred))
# print('Accuracy Score: ',accuracy_score(y_test, y_pred))

# Visualizing Confusion Matrix
# plt.figure(figsize = (6, 6))
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}, 
            yticklabels = ['Healthy', 'Diabetic'], xticklabels = ['Predicted Healthy', 'Predicted Diabetic'])
# plt.yticks(rotation = 0)
# plt.show()







