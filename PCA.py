from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
fig, axes = plt.subplots(15,2,figsize=(10,20))
maligant = cancer.data[cancer.target==0]
benign = cancer.data[cancer.target==0]

ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:i],bins=50)
    ax[i].hist(maligant[:,i], bins=bins)
    ax[i].hist(benign[:,i], bins = bins)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())

fig.tight_layout()
plt.show(axes)
