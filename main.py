import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("loan_approval_1000.csv")

print("===============\nDataset loaded successfully!\n=================")
#data exploration
print(df.head())
print("===============================\nDataset info: null values\n===============================")
print(df.isnull().sum())

print("===============================\nCalculating mean and mode for missing values\n===============================")
mean = df["Income"].mean()
print("Mean Income:", mean)

mode = df["Employment_Type"].mode()[0]
print("Mode Employment Type : ", mode)

#filling missing values with mean and 
df["Income"] = df["Income"].fillna(mean)
df["Employment_Type"] = df["Employment_Type"].fillna(mode)

print("===============================\nMissing values filled\n===============================")

#checking for missing values after filling
print(df.isnull().sum())

print("===============================\nEncoding categorical variables\n===============================")
#encoding categorical variables
le = LabelEncoder()
df['Employment_Type'] = le.fit_transform(df["Employment_Type"])
print(le.classes_)
