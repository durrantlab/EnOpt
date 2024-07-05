import pandas as pd
import numpy as np
import sklearn.ensemble as ens
import sklearn.model_selection as msl
import xgboost as xgb

from .evaluation import rocauc, prauc, bedroc, topn

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

def ensemble_best(dataframe,args):
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
    dataframe.insert(dataframe.columns.size,'rA',df_ranked.iloc[:,1:].mean(axis=1))
    return dataframe

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
    dataframe.insert(dataframe.columns.size,'rB',df_ranked.iloc[:,1:].min(axis=1))
    return dataframe

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
        return ensemble_best(dataframe,args)
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
    
    if args.tree_params != None:
        params = args.tree_params
    else: # default parameters
        params = {'n_estimators': 15,
                    'max_depth': 6,
                    } 

    if args.hyperparam:
        # params cv
        cv_results = hyperparams_tuning(rfc,unw_ens.to_numpy()[:,1:-1],known_ligs,args)
        rfc = cv_results[0]
        params = cv_results[1]
        print(params)

    rfc = rfc.set_params(**params)
    
    # cv
    cv_results = cv(unw_ens.to_numpy()[:,1:-1],known_ligs,20,rfc) 
    rfc_models = cv_results[0]
    wts = cv_results[1]
    aucs = cv_results[2]
    test_splits = cv_results[3]

    pred = np.zeros(len(unw_ens))
    mult = np.zeros(len(unw_ens))
    model = np.zeros(len(unw_ens))
    for i in range(3):
        pred[test_splits[i]] = rfc_models[i].predict_proba(unw_ens.to_numpy()[test_splits[i],1:-1])[:,1]
        mult[test_splits[i]] = np.matmul(unw_ens.to_numpy()[:,1:-1][test_splits[i]],wts[i])
        model[test_splits[i]] = i+1

    return (mult, pd.DataFrame(wts,columns=unw_ens.columns[1:-1]), pred, aucs, model)

    
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
    
    xgbc = xgb.XGBClassifier(verbosity=0,use_label_encoder=False)
    unw_ens = get_unweighted(dataframe,args)
    
    if args.tree_params != None:
        params = args.tree_params
    else: # default parameters
        params = {'n_estimators': 15,
                    'max_depth': 6,
                    'learning_rate': 0.3,
                    'colsample_bytree': 1,
                    } 

    if args.hyperparam:
        # params cv
        cv_results = hyperparams_tuning(xgbc,unw_ens.to_numpy()[:,1:-1],known_ligs,args)
        xgbc = cv_results[0]
        params = cv_results[1]
       
        print(params)
        #print params and hyperparam cv results

    xgbc = xgbc.set_params(**params)
    
    # cv
    cv_results = cv(unw_ens.to_numpy()[:,1:-1],known_ligs,20,xgbc)
    xgbc_models = cv_results[0]
    wts = cv_results[1]
    aucs = np.array(cv_results[2])
    test_splits = cv_results[3] 

    pred = np.zeros(len(unw_ens))
    mult = np.zeros(len(unw_ens))
    model = np.zeros(len(unw_ens))
    for i in range(3):
        pred[test_splits[i]] = xgbc_models[i].predict_proba(unw_ens.to_numpy()[test_splits[i],1:-1])[:,1]
        mult[test_splits[i]] = np.matmul(unw_ens.to_numpy()[:,1:-1][test_splits[i]],wts[i])
        model[test_splits[i]] = i+1

    return (mult, pd.DataFrame(wts,columns=unw_ens.columns[1:-1]), pred, aucs, model)


def cv(dataframe,known_ligs,topn_value,classifier_instance):
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
    weights = []
    aucs = []
    test_splits = []
    s = msl.StratifiedShuffleSplit(n_splits=3,test_size=0.35)
    tt_split = s.split(np.zeros(len(dataframe)),known_ligs)
    for i, tdat in enumerate(tt_split):
        m = classifier_instance.fit(dataframe[tdat[0]],known_ligs[tdat[0]])
        p = m.predict_proba(dataframe[tdat[1]])
        aucval = rocauc(known_ligs[tdat[1]],p[:,1]) 
        prcval = prauc(known_ligs[tdat[1]],p[:,1])
        brcval = bedroc(known_ligs[tdat[1]],pred_ligs=p[:,1])
        efval = topn(topn_value,known_ligs[tdat[1]],pred_ligs=p[:,1])
        
        models.append(m)
        weights.append(m.feature_importances_)
        aucs.append([aucval[0],prcval[0],brcval,efval])
        test_splits.append(tdat[1])
    #cv_model = models[np.argmax(aucs)]
    
    return (models,weights,aucs,test_splits)


def hyperparams_tuning(classifier_instance,dataframe,known_ligs,args):
    """Wraps tree model training functions when hyperparam tuning
    is requested, and returns best hyperparameters for tree model
    fitting.

    Uses sklearn model selection gridsearch module.
    
    Args:
        classifier_instance
        dataframe
        known_ligs

    Returns:
        params dictionary
    """
    if args.opt_method == 'XGB':
        params_dict = {'n_estimators': [15,50,100,150,200],
                        'max_depth': [2,5,10],
                        'learning_rate': [0.05,0.1,0.15],
                        'cosample_bynode': [0.2,0.5,1]
                        }
    elif args.opt_method == 'RF':
        params_dict = {'n_estimators': [15,50,100,200],
                        'max_depth': [2,5,10],
                        'max_features': [0.2,0.5,1]
                        }

    s = msl.StratifiedShuffleSplit(n_splits=3,test_size=0.35)
    tt_split = s.split(np.zeros(len(dataframe)),known_ligs)
    
    grid_search = msl.GridSearchCV(classifier_instance,params_dict,cv=tt_split)
    grid_results = grid_search.fit(dataframe,known_ligs)
    
    return (grid_results.best_estimator_,grid_results.best_params_,grid_results.cv_results_)

   
def single_conformation_scores(dataframe,known_ligs,args,topn_value=20):
    """Computes the rank-based predictions and evaluation
    metrics for each single conformation.
    
    Args:
        dataframe
        known_ligs

    Returns:
        params dictionary
    """
    aucs_dict = {}
    for col in dataframe.columns[1:]:
        if args.invert_score_sign:
            single_col = pd.DataFrame(dataframe[col].sort_values(ascending=False))
        else:
            single_col = pd.DataFrame(dataframe[col].sort_values(ascending=True))
        
        aucval = rocauc(known_ligs,single_col)
        prcval = prauc(known_ligs,single_col)
        brcval = bedroc(known_ligs,score_matrix=single_col,metric=col,invert=args.invert_score_sign)
        efval = topn(topn_value,known_ligs,score_matrix=single_col,metric=col,invert=args.invert_score_sign)
        
        aucs_dict[col] = [aucval[0],prcval[0],brcval,efval]

    return aucs_dict
