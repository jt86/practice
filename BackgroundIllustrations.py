from matplotlib import pyplot as plt
import numpy as np
import random
from Get_Full_Path import get_full_path

np.random.seed(5)


fig, axes = plt.subplots(1,2, figsize = [6,5])
N=10

ax = axes.flat[0]
x1, x2 = np.random.rand(N, )+0.25, np.random.rand(N)-0.25
y1, y2 = np.random.rand(N)+0.25, np.random.rand(N)-0.25
ax.scatter(x1,y1)
ax.scatter(x2,y2)
xline = np.linspace(x1.max(),x2.min())
yline = np.linspace(y2.min(),y1.max())
ax.plot(xline,yline, color='k')
ax.plot(xline,yline+0.1, color='k', linestyle='--', linewidth=0.9)
ax.plot(xline,yline-0.1, color='k', linestyle='--', linewidth=0.9)
ax.set_title('Primary feature space')

ax = axes.flat[1]
x1, x2 = np.random.rand(N), np.random.rand(N)+0.3
y1, y2 = np.random.rand(N)-0.25, np.random.rand(N)
ax.scatter(x1,y1)
ax.scatter(x2,y2)
all_x = np.concatenate([x1,x2])
all_y = np.concatenate([y1,y2])
plt.plot(np.unique(all_x), np.poly1d(np.polyfit(all_x, all_y, 1))(np.unique(all_x)), color='k')
ax.set_title('Privileged feature space')
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/Background/featurespaces.pdf'),format='pdf')
plt.show()