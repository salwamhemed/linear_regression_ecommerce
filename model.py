import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , mean_absolute_error
import math

df = pd.read_csv("C:\\Users\\salwa\\OneDrive\\Desktop\\machine learning projects\\linear regression - ecommerce\\Ecommerce Customers")
#df.head()
#df.info()
#df.describe()

#EDA
#sns.jointplot(X= 'Time on Website', y= 'Yearly Amount Spent', data=df, alpha=0.5)
#sns.jointplot(X='Time on App', y='Yearly Amount Spent', data=df, alpha=0.5)

#sns.pairplot(df, kind='scatter',plot_kws={'alpha': 0.4})
#sns.lmplot(X='Length of Membership', y='Yearly Amount Spent', data=df, scatter_kws={'alpha':0.3})
X = df[['Avg. Session Length','Time on App','Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.3, random_state=42)


lm = LinearRegression()
lm.fit(X_train, y_train)


odf = pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])
predictions = lm.predict(X_test)

sns.scatterplot(x=predictions, y= y_test)
plt.xlabel("predictions")
plt.ylabel("yearly amount spent")
plt.title("evaluation of the model")
plt.show()

print("Mean absolute error", mean_absolute_error(y_test, predictions))
print("Mean squared error", mean_squared_error(y_test, predictions))

#residuals
residuals = y_test - predictions
sns.displot(residuals,bins=30)


