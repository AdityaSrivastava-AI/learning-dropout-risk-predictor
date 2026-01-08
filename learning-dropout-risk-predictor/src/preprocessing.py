import pandas as pd

## Load dataset
df = pd.read_csv("data/student_engagement_data.csv")
##Basic data inspection
print('first 5 rows are:\n')
print(df.head())
print("\nttl no. of rows and clmns are:", df.shape)

print("BASIC INFO-----")
print(df.info())

print("\nMISSING VALUES ------:")
print(df.isnull().sum())

print("\nDUPLICATES--------")
print("Duplicate rows:", df.duplicated().sum())

## now we seperate feature and target vars
#asix=1(1 means clmn, 0 means row)
X = df.drop("dropout", axis=1)
y = df["dropout"]

print("\nFeatures shape:", X.shape)
print("\nTarget shape:", y.shape)

##train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    #80/20
    test_size=0.2,
    #to always have same target data to know model performance
    random_state=42,
    #stratify ditributes the data to train and test in equal ratio (20% topper in both)
    stratify=y
)

print("\nTrain set size:", X_train.shape)
print("Test set size:", X_test.shape)

##now we scale our features(X)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#scaling X train (fit-tho get mean and std d)
X_train_scaled = scaler.fit_transform(X_train)
#scaling Xtest (use mean and std d from Xtrain (so no cheating))
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed")

##Now we will do EDA 
import matplotlib.pyplot as plt
import seaborn as sns

# Making a copy of DataFrame for EDA
eda_df = df.copy()

# Split by dropout status
dropped = eda_df[eda_df["dropout"] == 1]
active = eda_df[eda_df["dropout"] == 0]

print("\nAverage values (Dropped vs Active):")
print(pd.concat([
    dropped.mean().rename("Dropped"),
    active.mean().rename("Active")
], axis=1))

##visualization

features = [
    "login_frequency",
    "time_spent_hours",
    "assignments_completed",
    "quiz_score",
    "days_inactive"
]
# canvas size(dimensions)
plt.figure(figsize=(12, 8))

for i, feature in enumerate(features, 1):
    #divides screen (2x3)
    plt.subplot(2, 3, i)
    #box plot from seaborn 
    sns.boxplot(x="dropout", y=feature, data=eda_df)
    plt.title(feature)
#no overlapping
plt.tight_layout()
plt.show()

##correlation heatmaps (corelation matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(eda_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


