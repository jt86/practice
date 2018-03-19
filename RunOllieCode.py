from sklearn.metrics import accuracy_score
import numpy as np
from New import svm_problem,svm_u_problem
from Models import SVMdp, SVMu

X=np.load('SVMdelta/Data/Dataset219/tech219-0-0-train_normal.npy')
Y=np.load('SVMdelta/Data/Dataset219/tech219-0-0-train_labels.npy')
Xtest=np.load('SVMdelta/Data/Dataset219/tech219-0-0-test_normal.npy')
Xstar = np.load('SVMdelta/Data/Dataset219/tech219-0-1-train_normal.npy')
Ytest=np.load('SVMdelta/Data/Dataset219/tech219-0-0-test_labels.npy')

problem = svm_problem(X,Xstar,Y,C=10)
problem2 = svm_u_problem(X, Xstar, Xstar,Y)

# s = SVM()
# s.train(x=X,prob=problem)



s2= SVMdp()
c2 = s2.train(prob=problem)
s3 = SVMu()
c3 = s3.train(problem2)
print(c3.support_vectors.shape)
print(c3.predict(Xtest[0]))

def get_accuracy_score(classifier, test_data, true_labels):
    predictions = [classifier.predict(Xtest[i]) for i in range(test_data.shape[0])]
    print(true_labels,'\n',[int(item) for item in predictions])
    return accuracy_score(true_labels,predictions)

print(get_accuracy_score(c3,Xtest,Ytest))