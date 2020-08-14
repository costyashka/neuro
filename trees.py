import graphviz
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
#print(cancer)
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data , cancer.target, stratify=cancer.target, random_state=1)

tree = DecisionTreeClassifier(max_depth=4,random_state=1)
tree.fit(X_train, y_train)

print('Правильность на обучающем наборе: {:.3f}'.format(tree.score(X_train,y_train)))
print('Правильность на тестовом наборе: {:.3f}'.format(tree.score(X_test,y_test)))
def vizualize():
    export_graphviz(tree, out_file='tree.dot', class_names=['malignant','benign'],
    feature_names=cancer.feature_names,impurity=True, filled=True)

    with open('tree.dot') as f:
        dot_graph = f.read()
    a = graphviz.Source(dot_graph)
    a.view()
def importants():
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), tree.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel('признак')
    plt.ylabel('важность')
    plt.show()
importants()