# Some useful parameters which will come in handy later on
#ntrain = train.shape[0]
#ntest = test.shape[0]
#SEED = 0 # for reproducibility
#NFOLDS = 5 # set folds for out-of-fold prediction
#kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
import numpy as np
import pandas as pd


def undersample(x, y, x_active, y_active, times):
    
    inactive_ind = np.where(y==0)[0]
    undersample_ind = np.random.choice(inactive_ind, (times * len(y_active)), replace=False)
    
    lst_x = np.zeros(x.shape)
    lst_y = np.zeros(y.shape)
    ind = 0
    for i in range(len(undersample_ind)):
        lst_x[ind] = x[undersample_ind[i]]
        lst_y[ind] = y[undersample_ind[i]]
        ind += 1
        
    new_x = np.concatenate((np.array(lst_x), x_active))
    new_y = np.concatenate((np.array(lst_y), y_active)) 
    
    return new_x, new_y

def get_oof_under(clf, x_train, y_train, x_test,ntrain, ntest, kf, times=3):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    x_active = x_train[y_train == 1]
    y_active = y_train[y_train == 1]

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]       

        x_tr, y_tr = undersample(x_tr, y_tr, x_active, y_active, times)

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        if type(params) != type(dict()):
            raise ValueError
            
        try:    
            params['random_state'] = seed
            self.clf = clf(**params)
        except Exception:
            params.pop('random_state')
            self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_    
    

'''
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)    
'''

'''
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
'''
from sklearn.model_selection import KFold
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction


class model_combin:
    def __init__(self, lst_of_models,lst_of_models_names, lst_of_dct_params):
        self.lst_of_dct_params = lst_of_dct_params
        self.lst_of_models_names = lst_of_models_names    
        self.lst_of_models = [SklearnHelper(clf=lst_of_models[i], seed=SEED, params=self.lst_of_dct_params[i]) for i in range(len(lst_of_models))]
        self.dct_of_models = {self.lst_of_models_names[i] : SklearnHelper(clf=lst_of_models[i], seed=SEED, params=self.lst_of_dct_params[i]) for i in range(len(lst_of_models))}

class KFold_for_combining_first_level:
    def __init__(self, lst_of_models_names, lst_of_models, lst_of_dct_params, train, test, target, kf=KFold(n_splits=NFOLDS, random_state=None, shuffle=False)):
    
        self.model_combin  = model_combin(lst_of_models,lst_of_models_names, lst_of_dct_params)
        self.KFold = kf
        self.lst_of_models_names = self.model_combin.lst_of_models_names
        self.lst_of_models = self.model_combin.lst_of_models
        
        # data
        self.__train = train
        self.__test = test
        
        # data preparation 
        self.y_train = self.__train[target].ravel()
        self.y_test = self.__test[target].ravel()
        
        train_x = self.__train.drop([target], axis=1)
        self.x_train = train_x.values # Creates an array of the train data
        
        test_x = self.__test.drop([target], axis=1)
        self.x_test = test_x.values # Creats an array of the test data
        
    def get_test_train(self, ntrain, ntest, times=3):
        lst_train = []
        lst_test = []
        for i in range(len(self.lst_of_models)):
            oof_train, oof_test = get_oof_under(self.lst_of_models[i], self.x_train, self.y_train, self.x_test, ntrain, ntest, self.KFold,times)
            lst_train.append(oof_train)
            lst_test.append(oof_test)
        return np.concatenate(tuple(lst_train), axis=1), np.concatenate(tuple(lst_test), axis=1)
    
    
            
        
        