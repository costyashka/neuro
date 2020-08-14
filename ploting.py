import matplotlib.pyplot as plt
import numpy as np 

x = np.linspace(-10,10,20)
y = np.arange(-10,10)
y = y*y

plt.plot(x,y)
plt.show()