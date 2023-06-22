import pandas as pd
import numpy as np
import sklearn.ensemble as ens
import sklearn.model_selection as msl
import xgboost as xgb

from .testing import rocauc
from .testing import unknown_ligs_method

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
    dataframe.insert(dataframe.columns.size,'eB',dataframe.iloc[:,1:].min(axis=1))
    return dataframe

def rank_average(dataframe):
    """Computes the average *rank* score across all conformations in ensemble.
    
    Ranking is done across all conformations (e.g. scores for each row/compound are 
    replaced with numbers 1-M for M conformations in the ensemble)
    Called internally during scoring.
    
    Args:
        dataframe (pandas DataFrame): docking score matrix.

    Returns:
        pandas DataFrame: docking score matrix with ranked average column added.
    """
    df_ranked = dataframe.rank(axis=0)
    df_ranked.insert(dataframe.columns.size,'rA',df_ranked.iloc[:,1:].mean(axis=1))
    return df_ranked

def rank_best(dataframe):
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
    df_ranked = dataframe.rank(axis=0)
    df_ranked.insert(dataframe.columns.size,'rB',df_ranked.iloc[:,1:].min(axis=1))
    return df_ranked

def get_unweighted(dataframe,scoring_scheme):
    """Wrapper for the scoring scheme functions above.
    Selects function based on user-provided scoring scheme.

    Args:
        dataframe (pandas DataFrame): docking score matrix.
        scoring_scheme (string): scoring scheme selection.

    Returns:
        pandas DataFrame: docking score matrix with scoring scheme column added.
    """
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
    """Function to generate 'active' ligand heuristics with no known ligands.

    Identifies a heuristic-based set of 'actives' for tree model fitting
    based on one of three methods:
        - top N compounds labeling the top 2% scorers as 'actives'
        - top N sample labeling a random sample of 2% out of the top 10%
          of scorers as 'actives'
        - (for paper and evaluation ONLY) top N as identified from actual 
          actives in the dataset when running label-blind heuristic tests.

    Args:
        unw_frame (pandas dataframe): docking score matrix with scoring scheme.
        scoring_scheme (string): scoring scheme selection.
        top_method (string): top-N heuristic selection.

    Returns:
        pandas Series: mask of top-scoring rows/compounds by index in docking matrix.
    """
    if top_method == 'top_n':
        # for n_known is the number of highest scorers
        n_known = int(0.02*len(unw_frame))
        top_n = unw_frame[scoring_scheme].sort_values()[n_known]
        top_ligs = unw_frame[scoring_scheme] < top_n
        print(top_ligs.dtype) 
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















