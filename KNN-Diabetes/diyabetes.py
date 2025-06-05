import pandas as pd 
import numpy as np
import  sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('/Users/omerfarukteker/Desktop/my_test/KNN-Diabetes/diabetes.csv')
print(len(dataset))
print(dataset.head())

# replace zeros 
zero_not_accedpted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in zero_not_accedpted:
    dataset[column] = dataset[column].replace(0, np.nan)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.nan, mean)
print(dataset.head())

print(dataset["Glucose"])