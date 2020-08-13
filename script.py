import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

passengers = pd.read_csv('passengers.csv')

passengers['Sex'] = passengers['Sex'].map({'female': 1, 'male': 0})
passengers['Age'].fillna(inplace=True, value=round(passengers['Age'].mean()))
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)

features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

train_features, test_features, train_labels, test_labels = train_test_split(features, survival)
scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

model = LogisticRegression()
model.fit(train_features, train_labels)
your_age = input("What is your age? ")
def find_gender():
    gender_input = input("What is your gender? ")
    if gender_input in ["Male", "male", "Man", "man", "Boy", "boy", "m", "M"]:
        return 0.0
    elif gender_input in ["Female", "female", "Woman", "woman", "Girl", "girl", "f", "F"]:
        return 1.0
    else:
        print("Invalid input. Please try again.")
your_gender = find_gender()
def find_class():
    class_input = input("""
    Which class would you have been in on the Titanic?
    A: First Class
    B: Second Class
    C: Third Class
    """)
    if class_input in ["A", "a"]:
        return (1.0, 0.0)
    if class_input in ["B", "b"]:
        return (0.0, 1.0)
    if class_input in ["C", "c"]:
        return (0.0, 0.0)
    else:
        print("Invalid input. Please try again.")
your_class = find_class()

Jack = np.array([0.0, 20.0, 0.0, 0.0])
Rose = np.array([1.0, 17.0, 1.0, 0.0])
You = np.array([your_gender, float(your_age), your_class[0], your_class[1]])

new_passengers = np.array([Jack, Rose, You])
new_passengers_transformed = scaler.transform(new_passengers)

if model.predict(new_passengers_transformed)[2] == 0:
    print("Rose survived, but unfortunately you and Jack did not.")
if model.predict(new_passengers_transformed)[2] == 1:
    print("Jack did not survive, but fortunately you and Rose lived!")
