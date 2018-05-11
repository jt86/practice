import numpy as np
import pprint

class Experiment_Setting:
    def __init__(self, classifier, datasetnum, lupimethod, featsel, stepsize=0.1, foldnum='all', topk=300, dataset='tech', skfseed=1, kernel='linear',
                 cmin=-3, cmax=3, numberofcs=7, percent_of_priv=100, percentageofinstances=100, take_top_t='top'):

        assert classifier in ['baseline','featselector','lufe','lufereverse','svmreverse','luferandom','lufeshuffle',
                              'svmtrain','lufetrain','lufenonlincrossval']
        assert lupimethod in ['nolufe','svmplus','dp','dsvm'], 'lupi method must be nolufe, svmplus or dp'
        assert featsel in ['nofeatsel','rfe','mi','anova','chi2','bahsic'], 'feat selection method not valid'

        self.cmin=cmin
        self.cmax=cmax
        self.foldnum = foldnum
        self.topk = topk
        self.dataset = dataset
        self.datasetnum = datasetnum
        self.kernel = kernel
        # self.cvalues = np.logspace(*[int(item)for item in cvalues.split('a')])
        self.cvalues = (np.logspace(cmin,cmax,numberofcs))
        self.skfseed = skfseed
        self.percent_of_priv = percent_of_priv
        self.percentageofinstances = percentageofinstances
        self.take_top_t = take_top_t
        self.lupimethod =lupimethod
        self.stepsize = stepsize
        self.featsel = featsel
        self.classifier = classifier


        # if self.classifier == 'baseline':
        #     self.lupimethod='nolufe'
        #     self.featsel='nofeatsel'
        #     self.topk='all'
        # if self.classifier == 'featsel' or 'svmreverse':
        #     self.lupimethod='nolufe'


        if stepsize==0.1:
            if self.classifier=='lufenonlincrossval':
                self.name = '{}{}{}-{}-{}-{}selected-{}{}priv'.format(self.classifier, self.cmin,self.cmax, self.lupimethod, self.featsel, self.topk,
                                                          self.take_top_t, self.percent_of_priv)
            else:
                self.name = '{}-{}-{}-{}selected-{}{}priv'.format(self.classifier, self.lupimethod, self.featsel, self.topk,
                                                          self.take_top_t, self.percent_of_priv)
        if percentageofinstances != 100:
            print('\n percent not 100',self.name)
            self.name = '{}-{}-{}-{}selected-{}{}priv-{}instances'.format(self.classifier, self.lupimethod, self.featsel, self.topk,
                                                          self.take_top_t, self.percent_of_priv, self.percentageofinstances)
        else:
            self.name = '{}-{}-{}-{}selected-{}{}priv-{}'.format(self.classifier, self.lupimethod, self.featsel, self.topk,
                                                          self.take_top_t, self.percent_of_priv, self.stepsize)

        if self.dataset!='tech':
            self.name = '{}-{}-{}-{}selected-{}{}priv'.format(self.classifier, self.lupimethod, self.featsel, self.topk,
                                                              self.take_top_t, self.percent_of_priv)
    def print_all_settings(self):
        pprint(vars(self))
        # print(self.k,self.top)