from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path

# maxval=50
# y_values1 = [(0.9**x)+0.1 for x in range(1,maxval+1)]+([0.105,0.104,0.103,0.102,0.101]+[0.1]*5)
# print(y_values1)
# y_values2 = [(0.85**x)+0.1 for x in range(1,maxval+1)]+([0.100]*10)
# print(y_values2)
# fig,ax = plt.subplots(figsize=[6,4])
# ax.plot(range(len(y_values1)),y_values1, label='Standard SVM')
# ax.plot(range(len(y_values2)),y_values2, label='Oracle SVM')
# ax.legend()
# ax.set_ylabel('Error rate')
# ax.set_xlabel('Number of training instances')
# ax.set_xticklabels([r'$10^{}$'.format(i) for i in range(9)])
# plt.axvline(x=20,linestyle='--',color='k', lw=0.7)
# plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/relatedwork/learningcurve.pdf'),format='pdf')
# plt.show()
#
# print((0.9**20)+0.1)
# print((0.85**20)+0.1)

maxval=50
y_values1 = [(0.9**x)+0.1 for x in range(1,maxval+1)]+([0.105,0.104,0.103,0.102,0.101]+[0.1]*5)
print(y_values1)
y_values2 = [(0.85**x)+0.1 for x in range(1,maxval+1)]+([0.100]*10)
print(y_values2)
fig,ax = plt.subplots(figsize=[6,4])
ax.plot(range(len(y_values1)),y_values1, label='Standard SVM')
ax.plot(range(len(y_values2)),y_values2, label='Oracle SVM')
ax.legend()
ax.set_ylabel('Error rate')
ax.set_xlabel('Number of training instances')
ax.set_xticklabels([r'$10^{}$'.format(i) for i in range(9)])
plt.axvline(x=20,linestyle='--',color='k', lw=0.7)
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/relatedwork/learningcurve.pdf'),format='pdf')
plt.show()

print((0.9**20)+0.1)
print((0.85**20)+0.1)