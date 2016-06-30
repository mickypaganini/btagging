from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import deepdish.io as io
import cPickle
import sys

CROSS_VAL = False

# -----------------------------------------------------------------

def train():
    '''
    '''
    data = io.load(open('train_data.h5', 'rb'))
    #data = remove_tau(data)
    
    if CROSS_VAL:
        param_grid = {'n_estimators':[50, 100], 'max_depth':[3, 5, 10], 'min_samples_split':[2, 5]}
        fit_params = {
                         'sample_weight' : data['w'],
                     }
        metaclassifier = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid, fit_params=fit_params, \
            cv=2, n_jobs=4, verbose=2)#, scoring=roc_score)
        metaclassifier.fit(data['X'], data['y'])
        classifier = metaclassifier.best_estimator_
        print 'Best classifier:', metaclassifier.best_params_

    else:
        classifier = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=10, verbose=1)
        classifier.fit(data['X'], data['y'], sample_weight=data['w'])

    joblib.dump(classifier, 'sklBDT_trk2.pkl', protocol=cPickle.HIGHEST_PROTOCOL)

# -----------------------------------------------------------------

def test():
    '''
    '''
    data = io.load(open('test_data.h5', 'rb'))
    #data = remove_tau(data)

    # -- Load scikit classifier
    classifier = joblib.load('sklBDT_trk2.pkl')
    
    # -- Get classifier predictions
    yhat = classifier.predict_proba(data['X'])[:, 2]

    io.save(open('yhat_test.h5', 'wb'), yhat)

# -----------------------------------------------------------------

def roc_score(clf, X, y):
    '''
    '''
    from sklearn.metrics import roc_curve, auc

    yhat = clf.predict_proba(X)[:, 2] # make sure you select out the right column! (depends on whether we are doing binary or multiclass classification) 
    
    bl_sel = (y == 0) | (y == 5)
    fpr, eff, _ = roc_curve(y[bl_sel] == 5, yhat[bl_sel])
    rej = 1 / fpr
    
    select = (eff > 0.5) & (eff < 1.0)
    rej = rej[select]
    eff = eff[select]
    
    return auc(eff, rej)

# ----------------------------------------------------------------- 

def remove_tau(data):

    data['X'] = data['X'][data['y'] != 15]
    data['y'] = data['y'][data['y'] != 15]
    data['w'] = data['w'][data['y'] != 15]

    return data

# ----------------------------------------------------------------- 

if __name__ == '__main__':

    # -- read in arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='train, test or traintest')
    args= parser.parse_args()

    if args.mode == 'train':
        sys.exit(train())
    elif args.mode == 'test':
        sys.exit(test())
    elif args.mode == 'traintest':
        train()
        sys.exit(test())
    else:
        sys.exit('Error: unknown mode')
