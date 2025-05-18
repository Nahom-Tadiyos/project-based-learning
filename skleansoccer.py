import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('datasets\ME.csv')

df['is_goal'] = df['result'].apply(lambda x: 1 if x == 'Goal' else 0)

X = df[['minute', 'X', 'Y', 'xG', 'player_id', 'situation', 'shotType']]
y = df['is_goal']

le = LabelEncoder()
df['shotType_encoded'] = le.fit_transform(df['shotType'])
X = X.assign(shotType=df['shotType_encoded'])
X = pd.get_dummies(X, columns=['situation'], drop_first=True)

scaler = StandardScaler()
numerical_features = ['minute', 'X', 'Y', 'xG', 'player_id']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))