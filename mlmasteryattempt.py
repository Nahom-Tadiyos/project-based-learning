from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns

url = "datasets\ME.csv"
data = read_csv(url)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='X', y='Y', hue='result', data=data)
plt.title('Shot Locations by Result')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().invert_yaxis() # Adjust y-axis to represent the football pitch
plt.show()

# You can add more visualizations based on your specific interests.
# For example, to see the distribution of xG:
plt.figure(figsize=(8, 5))
sns.histplot(data['xG'], kde=True)
plt.title('Distribution of Expected Goals (xG)')
plt.xlabel('Expected Goals (xG)')
plt.ylabel('Frequency')
plt.show()