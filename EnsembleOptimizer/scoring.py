import pandas as pd
import numpy as np
import sklearn.ensemble as ens
import sklearn.model_selection as msl
import xgboost as xgb

from .testing import rocauc

# scoring functions (regular average vs. weighted average) of whole dataframe

def ensemble_average(dataframe):
    dataframe.insert(dataframe.columns.size,'eA',dataframe.iloc[:,1:].mean(axis=1))
    return dataframe

def ensemble_best(dataframe):
    dataframe.insert(dataframe.columns.size,'eB',dataframe.iloc[:,1:].min(axis=1))
    return dataframe

def rank_average(dataframe):
    df_ranked = dataframe.rank(axis=0)
    df_ranked.insert(dataframe.columns.size,'rA',df_ranked.iloc[:,1:].mean(axis=1))
    return df_ranked

def rank_best(dataframe):
    df_ranked = dataframe.rank(axis=0)
    df_ranked.insert(dataframe.columns.size,'rB',df_ranked.iloc[:,1:].min(axis=1))
    return df_ranked

def get_unweighted(dataframe,scoring_scheme):
    if scoring_scheme == 'eA':
        return ensemble_average(dataframe)
    elif scoring_scheme == 'eB':
        return ensemble_best(dataframe)
    elif scoring_scheme == 'rA':
        return rank_average(dataframe)
    elif scoring_scheme == 'rB':
        return rank_best(dataframe)

### weighting

def unknown_ligs_method(unw_frame,scoring_scheme,top_method='top_n'):
    if top_method == 'top_n':
        # for n_known is the number of highest scorers
        n_known = int(0.02*len(unw_frame))
        top_n = unw_frame[scoring_scheme].sort_values()[n_known]
        top_ligs = unw_frame[scoring_scheme] < top_n
        
    elif top_method == 'sample':    
        # for n_known selection of 2% from top 10%
        known_range = range(int(len(unw_frame)*0.1))
        n_known = np.random.default_rng().choice(known_range,size=int(len(unw_frame)*0.02))
        top_ligs = np.isin(np.arange(len(unw_frame)),n_known)
    
    else:
        n_known_dict = {'COMT':41,
                        'GLCM':54,
                        'GLCM8':54,
                        'HXK4':92,
                        'KIF11':116,
                        'TGFR1':133,
                        'TRYB1':148,
                        'XIAP':100}
        n_known = n_known_dict[top_method]
        top_n = unw_frame[scoring_scheme].sort_values()[n_known]
        top_ligs = unw_frame[scoring_scheme] < top_n
    
    return top_ligs.astype(int)


def get_weights_RF(dataframe,known_ligs,scoring_scheme):
    
    unw_ens = get_unweighted(dataframe,scoring_scheme)
    rfc = ens.RandomForestClassifier()

    # get weights for NO KNOWN ligands
    if np.sum(known_ligs) == 0:
        top_ligs = unknown_ligs_method(unw_ens,scoring_scheme)
        rfc.fit(unw_ens.to_numpy()[:,1:-1],top_ligs)

        aucs = None
    
    # get weights for KNOWN ligands
    else:
        # cv
        cv_results = cv(rfc,unw_ens.to_numpy()[:,1:-1],known_ligs) 
        rfc = cv_results[0]
        aucs = cv_results[1]

    wts = rfc.feature_importances_
    pred = rfc.predict_proba(unw_ens.to_numpy()[:,1:-1])
    return (np.matmul(unw_ens.to_numpy()[:,1:-1],wts), pd.Series(wts,index=unw_ens.columns[1:-1]), pred[:,1], aucs)
        

def get_weights_XGB(dataframe,known_ligs,scoring_scheme):
    
    xgbc = xgb.XGBClassifier(n_estimators=15,verbosity=0,use_label_encoder=False)
    unw_ens = get_unweighted(dataframe,scoring_scheme)

    # get weights for NO KNOWN ligands
    if np.sum(known_ligs) == 0:
        top_ligs = unknown_ligs_method(unw_ens,scoring_scheme)
        xgbc_p = xgbc.fit(unw_ens.to_numpy()[:,1:-1],top_ligs)

        aucs = None

    # get weights for KNOWN ligands -- 3fCV
    else:
        # cv
        cv_results = cv(xgbc,unw_ens.to_numpy()[:,1:-1],known_ligs)
        xgbc_p = cv_results[0]
        aucs = cv_results[1]

    wts = xgbc_p.feature_importances_
    pred = xgbc_p.predict_proba(unw_ens.to_numpy()[:,1:-1])
    return (np.matmul(unw_ens.to_numpy()[:,1:-1],wts), pd.Series(wts,index=unw_ens.columns[1:-1]), pred[:,1], aucs)


def cv(classifier_instance,dataframe,known_ligs):
    models = []
    aucs = []
    s = msl.StratifiedShuffleSplit(n_splits=3,test_size=0.35)
    tt_split = s.split(np.zeros(len(dataframe)),known_ligs)
    
    for i, tdat in enumerate(tt_split):
        m = classifier_instance.fit(dataframe[tdat[0]],known_ligs[tdat[0]])
        w = m.feature_importances_
        p = m.predict_proba(dataframe[tdat[1]])
        aucval = rocauc(known_ligs[tdat[1]],p[:,1])
        models.append(m)
        aucs.append(aucval[0])
    
    cv_model = models[np.argmax(aucs)]
    
    return (cv_model, aucs)















