from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_lfw_people
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

people = fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask] / 255
y_people = people.target[mask]

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people)

pca = PCA(n_components=75,whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train_pca,y_train)
print(model.score(X_test_pca,y_test))

fix, axes = plt.subplots(3, 5, figsize=(15,12), subplot_kw={'xticks': (),'yticks': ()})

for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape),cmap='viridis')
    ax.set_title(f'{i+1} component')
plt.show()