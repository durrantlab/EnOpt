# testing functions used in evaluation and benchmarking
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

try:
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
except:
    print('RDKit dependency missing, required for BEDROC caclulation. Skipping BEDROC metric.')


# compute ROCAUC
def rocauc(kn_ligs,pred_ligs):
    """Compute ROCAUC using sklearn.
    
    Args:
        kn_ligs (np.ndarray): known ligand indices.
        pred_ligs (np.ndarray): predicted ligand indices.

    Returns:
        tuple: containing auc and roc-curve data
        np.float: auc score for the input labels and predictions.
        np.ndarray: roc plot values for the input labels and predictions.
    """
    curve_data = roc_curve(kn_ligs,pred_ligs)
    auc = roc_auc_score(kn_ligs,pred_ligs)
    return (auc, curve_data)

def topn(topn,kn_ligs,pred_ligs=None,score_matrix=None,metric=None):
    """Compute top-n (3%) enrichment ratio.
    
    Args:
        score_matrix (pd.DataFrame): predicted prob. and weighted score matrix.
        kn_ligs (np.ndarray): known ligand indices.
        topn (int): top % to use when computing enrichment.
        metric (str): column of score matrix (eA/eB/rA/rB or predicted prob.)
                      used to determined top N.

    Returns:
        float: ratio of known ligands in the top 3% of ranked output.
    """
    if pred_ligs is None:
        # computes true positives in top n
        lig_ranking = np.argsort(score_matrix[metric])
    else:
        lig_ranking = np.argsort(pred_ligs) 
    
    if metric == 'Predicted probability' or metric == 'inverse':
        # highest n probabilities
        topn_cutoff = float(1-(topn/100))*len(kn_ligs)
        top_n_enriched = np.sum(np.multiply(lig_ranking,kn_ligs) > topn_cutoff)
    else: 
        # lowest n scores
        topn_cutoff = float(topn/100)*len(kn_ligs)
        top_n_enriched = np.sum(np.multiply(lig_ranking,kn_ligs) < topn_cutoff)
    
    return top_n_enriched/topn_cutoff


def bedroc(kn_ligs,pred_ligs=None,score_matrix=None,metric=None):
    """Compute BEDROC (Boltzmann enhanxced discrimination of ROC) score.
    
    Args:
        kn_ligs (np.ndarray): known ligand indices.
        pred_ligs (np.ndarray): predicted ligand indices.
        score_matrix (pd.DataFrame): predicted prob. and weighted score matrix.
        metric (str): column of score matrix (eA/eB/rA/rB or predicted prob.)

    Returns:
        float: BEDROC score.
    """
    if pred_ligs is None:
        cols = len(score_matrix.columns)
        score_input = score_matrix.insert(cols,'Known label',kn_ligs.astype(bool))
        if metric == 'Predicted probability' or metric == 'inverse':
            # highest n probabilities
            score_input = score_input.sort_values(by=metric,ascending=False)
        else: 
            # lowest n scores
            score_input = score_input.sort_values(by=metric,ascending=True)
        
        bedroc = CalcBEDROC(score_input[[metric,'Known label']].values,1,20)
    else:
        score_input = np.column_stack([kn_ligs,pred_ligs])
        indices = np.argsort(score_input[:,-1])
        score_input = score_input[indices]
        bedroc = CalcBEDROC(score_input,0,20)
    
    return bedroc

def prauc(kn_ligs,pred_ligs):
    """Compute PRAUC (precision-recall AUC) score.
    
    Args:
        kn_ligs (np.ndarray): known ligand indices.
        pred_ligs (np.ndarray): predicted ligand indices.

    Returns:
        tuple: containing auc and roc-curve data
        np.float: prauc score for the input labels and predictions.
        np.ndarray: pr plot values for the input labels and predictions.
    """
    curve_data = roc_curve(kn_ligs,pred_ligs)
    prauc = auc(curve_data[1],curve_data[0])
    return (prauc, curve_data)
