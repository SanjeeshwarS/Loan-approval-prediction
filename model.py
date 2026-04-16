import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("loan_approval_1000.csv")

print("===============\nDataset loaded successfully!\n=================")
#data exploration
print(df.head())

print("===============\nDataset Information: Null Values\n=================")
#checking for missing values
print(df.isnull().sum())

print("===============\nCalculating Mean and Mode for Missing Values\n=================")
#calculating mean and mode for filling missing values
mean = df["Income"].mean()
print("Mean Income:", mean)

mode = df["Employment_Type"].mode()[0]
print("Mode Employment Type : ", mode)

print("===============\nFilling Missing Values\n=================")
#filling missing values with mean and 
df["Income"] = df["Income"].fillna(mean)
df["Employment_Type"] = df["Employment_Type"].fillna(mode)


#checking for missing values after filling
print(df.isnull().sum())

print("===============\nEncoding Categorical Variables\n=================")
#encoding categorical variables
le = LabelEncoder()
df['Employment_Type'] = le.fit_transform(df["Employment_Type"])
print(le.classes_)

print("===============\nDataset after Preprocessing\n=================")
#checking the dataset after preprocessing
print(df.head())

print("===============\nModel Training\n=================")
#separating features and target variable
x = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]
print("Features (X):")
print(x.head())
print("Target (y):")
print(y.head())

#splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#training the logistic regression model
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(x_train, y_train)

#predicting on the test set
y_pred = lr.predict(x_test)

print("===============\nModel Evaluation\n=================")
#calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: ",  accuracy)


#classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

