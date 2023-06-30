import pandas as pd
import numpy as np
import sklearn.ensemble as ens
import sklearn.model_selection as msl
import xgboost as xgb

from .evaluation import rocauc
from .evaluation import unknown_ligs_method

# scoring functions (regular average vs. weighted average) of whole dataframe

def ensemble_average(dataframe):
    """Computes the average score across all conformations in ensemble.
    Called internally during scoring.
    
    Args:
        dataframe (pandas DataFrame): docking score matrix.

    Returns:
        pandas DataFrame: docking score matrix with ensemble average column added.
    """
    dataframe.insert(dataframe.columns.size,'eA',dataframe.iloc[:,1:].mean(axis=1))
    return dataframe

def ensemble_best(dataframe):
    """Computes the best (default: lowest, can be changed by user option) 
    score across all conformations in ensemble.
    Called internally during scoring.
    
    Args:
        dataframe (pandas DataFrame): docking score matrix.

    Returns:
        pandas DataFrame: docking score matrix with ensemble best column added.
    """
    if args.invert_score_sign is True:
        dataframe.insert(dataframe.columns.size,'eB',dataframe.iloc[:,1:].max(axis=1))
    else:    
        dataframe.insert(dataframe.columns.size,'eB',dataframe.iloc[:,1:].min(axis=1))
    return dataframe

def rank_average(dataframe,args):
    """Computes the average *rank* score across all conformations in ensemble.
    
    Ranking is done across all conformations (e.g. scores for each row/compound are 
    replaced with numbers 1-M for M conformations in the ensemble)
    Called internally during scoring.
    
    Args:
        dataframe (pandas DataFrame): docking score matrix.

    Returns:
        pandas DataFrame: docking score matrix with ranked average column added.
    """
    df_ranked = dataframe.rank(axis=0,ascending=(not args.invert_score_sign))
    df_ranked.insert(dataframe.columns.size,'rA',df_ranked.iloc[:,1:].mean(axis=1))
    return df_ranked

def rank_best(dataframe,args):
    """Computes the best (default: lowest, can be changed by user option) 
    score *rank* across all conformations in ensemble.
    
    Ranking is done across all conformations (e.g. scores for each row/compound are 
    replaced with numbers 1-M for M conformations in the ensemble)
    Called internally during scoring.
    
    Args:
        dataframe (pandas DataFrame): docking score matrix.

    Returns:
        pandas DataFrame: docking score matrix with ranked best column added.
    """
    df_ranked = dataframe.rank(axis=0,ascending=(not args.invert_score_sign))
    df_ranked.insert(dataframe.columns.size,'rB',df_ranked.iloc[:,1:].min(axis=1))
    return df_ranked

def get_unweighted(dataframe,args):
    """Wrapper for the scoring scheme functions above.
    Selects function based on user-provided scoring scheme.

    Args:
        dataframe (pandas DataFrame): docking score matrix.
        scoring_scheme (string): scoring scheme selection.

    Returns:
        pandas DataFrame: docking score matrix with scoring scheme column added.
    """
    if args.scoring_scheme == 'eA':
        return ensemble_average(dataframe)
    elif args.scoring_scheme == 'eB':
        return ensemble_best(dataframe)
    elif args.scoring_scheme == 'rA':
        return rank_average(dataframe,args)
    elif args.scoring_scheme == 'rB':
        return rank_best(dataframe,args)

### weighting


def get_weights_RF(dataframe,known_ligs,args):
    """Fits Random Forest model from scikit-learn to docking score data.
    
    Calls CV function to perform 3-fold cross validation when known ligands
    are provided. If no known ligands are provided, the top N best-scoring 
    ligands are assumed to be actives. See Methods in the paper for more information.
    Model fitting is only called if weighted scoring is requested (default: True).

    Args:
        dataframe (pandas dataframe): docking score matrix.
        known_ligs (numpy array): bool mask of known ligands.
        scoring_scheme (string): scoring scheme selection.

    Returns:
        tuple of (
            numpy array: weighted single score for each row/compound.
            pandas Series: conformation weights from RF feature importances.
            numpy array: predicted probability of being an 'active' per compound.
            list: ROCAUCs for three models built in CV.
            )
    """
    
    unw_ens = get_unweighted(dataframe,args)
    rfc = ens.RandomForestClassifier()

    # get weights for NO KNOWN ligands
    if np.sum(known_ligs) == 0:
        top_ligs = unknown_ligs_method(unw_ens,args)
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
        

def get_weights_XGB(dataframe,known_ligs,args):
    """Fits gradient-boosted tree model from XGBoost to docking score data.
    
    Calls CV function to perform 3-fold cross validation when known ligands
    are provided. If no known ligands are provided, the top N best-scoring 
    ligands are assumed to be actives. See Methods in the paper for more information.
    Model fitting is only called if weighted scoring is requested (default: True).

    Args:
        dataframe (pandas dataframe): docking score matrix.
        known_ligs (numpy array): bool mask of known ligands.
        scoring_scheme (string): scoring scheme selection.

    Returns:
        tuple of (
            numpy array: weighted single score for each row/compound.
            pandas Series: conformation weights from XGB feature importances.
            numpy array: predicted probability of being an 'active' per compound.
            list: ROCAUCs for three models built in CV.
            )
    """
    
    xgbc = xgb.XGBClassifier(n_estimators=15,verbosity=0,use_label_encoder=False)
    unw_ens = get_unweighted(dataframe,args)

    # get weights for NO KNOWN ligands
    if np.sum(known_ligs) == 0:
        top_ligs = unknown_ligs_method(unw_ens,args)
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
    """Performs 3-fold cross validation for tree classifiers.
    
    Uses scikit-learn's data split/shuffle to generate three left-out 
    validation sets, and trains a tree model on each set of remaining
    data. Returns the best model and all model evaluations.
    Calls rocauc function from evaluation.py.

    Args:
        classifier_instance (RF or XGB classifier): Description of param1.
        dataframe (pandas dataframe): docking score matrix with scoring scheme.
        known_ligs (numpy array): bool mask of known ligands.

    Returns:
        tuple of (
            RF/XGB classifier: selected best model from 3-fold CV.
            list: ROCAUCs for three tree models built. 
            )
    """
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















