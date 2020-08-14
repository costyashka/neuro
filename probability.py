from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,random_state=1)
gbc = GradientBoostingClassifier(random_state=1)

gbc.fit(X_train,y_train)

print("Form: {}".format(gbc.decision_function(X_test).shape))
print(gbc.decision_function(X_test)[:6])
print(gbc.predict_proba(X_test)[:6])
print(gbc.predict_proba(X_test)[:6,:].sum(axis=1))