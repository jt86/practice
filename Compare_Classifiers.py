from Models import *
from timeit import default_timer as timer
from Kernels import *
import logging
from multiprocessing import Pool as ThreadPool
import pickle
from operator import itemgetter
from sklearn.model_selection import KFold
import math as mt

def get_array(file):
    return np.load(file)

def get_accuracy(tp, fp, fn, tn):
    return (tp+tn)/(tp+fp+fn+tn)#+0.000001)

def get_error(tp, fp, fn, tn):
    return (fp+fn)/(tp+fp+fn+tn)#+0.000001)

def get_recall(tp, fp, fn, tn):
    return (tp)/(tp+fn) if tp+fn > 0 else 0

def get_specificity(tp, fp, fn, tn):
    return (tn)/(fp+tn) if fp+tn > 0 else 0

def get_precision(tp, fp, fn, tn):
    return (tp)/(tp+fp) if tp+fp > 0 else 0

def get_prevalence(tp, fp, fn, tn):
    return (tp+fn)/(tp+fp+fn+tn)#+0.000001)

def get_fscore(pre, rec):
    return 2*((pre*rec)/(pre+rec)) if (pre+rec) > 0 else 0

def t(data, model, test_x, test_y):
    svm = model
    test_y = np.asarray(test_y).reshape(-1)

    if isinstance(svm, SVM):
        clf = svm.train(data.X, data)
    else:
        clf = svm.train(data)
    predictions = []
    for test_point in test_x:
        predictions.append(clf.predict(test_point))

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(test_y)):
        if test_y[i] == 1 and predictions[i] == 1:
            tp += 1
        if test_y[i] == -1 and predictions[i] == 1:
            fp += 1
        if test_y[i] == 1 and predictions[i] == -1:
            fn += 1
        if test_y[i] == -1 and predictions[i] == -1:
            tn += 1
    return (tp, fp, fn, tn)

def comp(clf, prob, test_x, test_y):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    average_time = 0
    #for fold in probs:
    start = timer()
    a, b, c, d = t(prob, clf, test_x, test_y)
    average_time += timer() - start
    tp += a
    fp += b
    fn += c
    tn += d
    return tp,fp,fn,tn,average_time

def svm_comp(clf, prob, test_x, test_y):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    average_time = 0
    start = timer()
    a, b, c, d = t(prob, clf, test_x, test_y)
    average_time += timer() - start
    tp += a
    fp += b
    fn += c
    tn += d
    return tp,fp,fn,tn,average_time

def grid_search(C, Delta, Gamma, dataset):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    xkerns = [Gaussian(), Polynomial(), Linear()]
    xskerns = [Gaussian(), Polynomial(), Linear()]

    results = []

    for i in range(3):
        prob_data = []
        test_labels = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_labels.npy")
        test_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_normal.npy")
        train_y = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_labels.npy").reshape(-1,1)
        train_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_normal.npy")
        train_xS = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_priv.npy")
        prob_data.append([test_labels, test_x, train_y, train_x, train_xS])

        prob_data = np.array(prob_data)
        conc_data = []
        for j in range(len(prob_data)):
            conc_data = np.hstack((train_x, train_xS))
            conc_data = np.hstack((conc_data, train_y))
        np.random.shuffle(conc_data)

        kf = KFold(shuffle=True, n_splits=5)
        inner_folds_train_index = []
        inner_folds_test_index = []
        inner_folds_train = []
        inner_folds_test = []
        for train, test in kf.split(conc_data):
            inner_folds_train_index.append(train)
            inner_folds_test_index.append(test)

        for j in range(5):
            inner_folds_train.append(np.array(conc_data[inner_folds_train_index[j]]))
            inner_folds_test.append(np.array(conc_data[inner_folds_test_index[j]]))

        inner_probs = []
        for j in range(5):
            inner_x = inner_folds_train[j][:,0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1]+train_xS.shape[1]]
            inner_y = inner_folds_train[j][:, train_x.shape[1]+train_xS.shape[1]:train_x.shape[1]+train_xS.shape[1]+train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:, :test_x.shape[1]]#test_x.shape[1]:test_x.shape[1]+train_xS.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])

        inner_probs = np.array(inner_probs)

        svm_prob = [(data[3], data[4], data[2], data[1], data[0], c, xk, data[5], data[6]) for data in inner_probs for c in C for xk in xkerns]
        svmp_prob = [(data[3], data[4], data[2], data[1], data[0], c, g, xk, xsk, data[5], data[6]) for data in inner_probs for c in C for g in Gamma for xk in xkerns for xsk in xskerns]
        svmdps_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, xk, xsk, data[5], data[6], 0) for data in inner_probs for c in C for d in Delta for xk in xkerns for xsk in xskerns]
        svmdp_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, g, xk, xsk, data[5], data[6], 0) for data in inner_probs for c in C for d in Delta for g in Gamma for xk in xkerns for xsk in xskerns]

        print(len(svmp_prob))

        pool = ThreadPool(4)

        #resultsSVM = pool.map(svmThread, svm_prob)
        resultsSVMp = pool.map(svmpThread, svmp_prob)
        #resultsSVMdpsa = pool.map(svmdpsaThread, svmdps_prob)
        #resultsSVMdp = pool.map(svmdpThread, svmdp_prob)

        pool.close()
        pool.join()
        results = results+resultsSVMp#resultsSVM+resultsSVMp+resultsSVMdpsa+resultsSVMdp

    with open('SVMPLUSoutfile', 'wb') as fp:
        pickle.dump(results, fp)



def svmThread(p):
    #print("SVM Thread >>> ", p[0].shape, p[1].shape, p[3].shape, " <<<<")
    prob = svm_problem(p)
    logging.info("Entered multicore process")
    svm_tp, svm_fp, svm_fn, svm_tn, svm_avg_time = svm_comp(SVM(), prob, p[3], p[4])
    logging.info("model trained"+" "+str(svm_tp)+" "+str(svm_fp)+" "+str(svm_fn)+" "+str(svm_tn))
    svm_acc = get_accuracy(svm_tp, svm_fp, svm_fn, svm_tn)
    svm_pre = get_precision(svm_tp, svm_fp, svm_fn, svm_tn)
    svm_rec = get_recall(svm_tp, svm_fp, svm_fn, svm_tn)
    svm_fsc = get_fscore(svm_pre, svm_rec)
    return ("SVM", p[7], p[8], svm_acc, svm_fsc, p[5], p[6].getName(), svm_avg_time)

def svmpThread(p):
    prob = svm_problem(p)
    logging.info("Entered multicore process")
    svmp_tp, svmp_fp, svmp_fn, svmp_tn, svmp_avg_time = comp(SVMp(), prob, p[3], p[4])
    logging.info("model trained")
    svmp_acc = get_accuracy(svmp_tp, svmp_fp, svmp_fn, svmp_tn)
    svmp_pre = get_precision(svmp_tp, svmp_fp, svmp_fn, svmp_tn)
    svmp_rec = get_recall(svmp_tp, svmp_fp, svmp_fn, svmp_tn)
    svmp_fsc = get_fscore(svmp_pre, svmp_rec)
    logging.info("Completed")
    return ("SVM+", p[9], p[10], svmp_acc, svmp_fsc, p[5], p[6], p[7].getName(), p[8].getName(), svmp_avg_time)

def svmdpsaThread(p):
    prob = svm_problem(p)
    logging.info("Entered multicore process")
    svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn, svmdpsa_avg_time = comp(SVMdp_simp(), prob, p[3], p[4])
    logging.info("model trained")
    svmdpsa_acc = get_accuracy(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_pre = get_precision(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_rec = get_recall(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_fsc = get_fscore(svmdpsa_pre, svmdpsa_rec)
    logging.info("Completed")
    return ("SVMd+ - simp", p[9], p[10], svmdpsa_acc, svmdpsa_fsc, p[5], p[6], p[7].getName(), p[8].getName(), svmdpsa_avg_time)

def svmdpThread(p):
    prob = svm_problem(p)
    logging.info("Entered multicore process")
    svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn, svmdp_avg_time = comp(SVMdp(), prob, p[3], p[4])
    logging.info("model trained")
    svmdp_acc = get_accuracy(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_pre = get_precision(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_rec = get_recall(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_fsc = get_fscore(svmdp_pre, svmdp_rec)
    logging.info("Completed")
    return ("SVMd+ - simp", p[10], p[11], svmdp_acc, svmdp_fsc, p[5], p[6], p[7], p[8].getName(), p[9].getName(), svmdp_avg_time)


#grid_search([0.1, 1, 10], [0.1, 1, 10], [0.1, 1, 10], 137)


'''
with open ('SVMoutfile', 'rb') as fp:
    itemlist = pickle.load(fp)
itemlist = sorted(itemlist, key=itemgetter(6,1,2,4,3))
print("===== SVM =====")
for ker in ["Gaussian", "Linear", "Polynomial p=2"]:
    print("\n",ker)
    print("====================")
    for c in [0.1, 1, 10]:
        ca = 0
        cf = 0
        for i in range(3):
            temp = [(x[3],x[4]) for x in itemlist if x[5] == c and x[6] == ker and x[1] == i]
            ca += np.mean([x[0] for x in temp])
            cf += np.mean([x[1] for x in temp])
        print("C =", c, "Average accuracy:", ca/3, "F-Score:", cf/3)


with open ('SVMPLUSoutfile', 'rb') as fp:
    itemlist = pickle.load(fp)
itemlist = sorted(itemlist, key=itemgetter(6,1,2,4,3))
print("===== SVM+ =====")
for xker in ["Gaussian", "Linear", "Polynomial p=2"]:
    for xsker in ["Gaussian", "Linear", "Polynomial p=2"]:
        print("\nX: ",xker, "X*:", xsker)
        print("====================")
        for c in [0.1, 1, 10]:
            for g in [0.1, 1, 10]:
                gca = 0
                gcf = 0
                for i in range (3):
                    temp = [(x[3],x[4]) for x in itemlist if x[5] == c and x[6] == g and x[7] == xker and x[8] == xsker and x[1] == i]
                    gca += np.mean([x[0] for x in temp])
                    gcf += np.mean([x[1] for x in temp])
                print("C =", c, "Gam =", g, "Average accuracy:", gca/3, "F-Score:", gcf/3)


with open ('SVMdpSIMPoutfile', 'rb') as fp:
    itemlist = pickle.load(fp)
itemlist = sorted(itemlist, key=itemgetter(6,1,2,4,3))
print("===== SVMd+ simp =====")
for xker in ["Gaussian", "Linear", "Polynomial p=2"]:
    for xsker in ["Gaussian", "Linear", "Polynomial p=2"]:
        print("\nX: ",xker, "X*:", xsker)
        print("====================")
        for c in [0.1, 1, 10]:
            for d in [0.1, 1, 10]:
                dca = 0
                dcf = 0
                for i in range (3):
                    temp = [(x[3],x[4]) for x in itemlist if x[5] == c and x[6] == d and x[7] == xker and x[8] == xsker and x[1] == i]
                    dca += np.mean([x[0] for x in temp])
                    dcf += np.mean([x[1] for x in temp])
                print("C =", c, "Del =", d, "Average accuracy:", dca/3, "F-Score:", dcf/3)


with open ('SVMdpoutfileC', 'rb') as fp:
    itemlist = pickle.load(fp)
itemlist = sorted(itemlist, key=itemgetter(6,1,2,4,3))
print("===== SVMd+ simp =====")
for xker in ["Gaussian", "Linear", "Polynomial p=2"]:
    for xsker in ["Gaussian", "Linear", "Polynomial p=2"]:
        print("\nX: ",xker, "X*:", xsker)
        print("====================")
        for c in [0.1, 1, 10]:
            for d in [0.1, 1, 10]:
                for g in [0.1, 1, 10]:
                    gdca = 0
                    gdcf = 0
                    for i in range (3):
                        temp = [(x[3],x[4]) for x in itemlist if x[5] == c and x[6] == d and x[7] == g and x[8] == xker and x[9] == xsker and x[1] == i]
                        gdca += np.mean([x[0] for x in temp])
                        gdcf += np.mean([x[1] for x in temp])
                    print("C =", c, "Del = ", d, "Gam =", g, "Average accuracy:", gdca/3, "F-Score:", gdcf/3)
'''
