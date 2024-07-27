# testing functions used in evaluation and benchmarking
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from rdkit.ML.Scoring.Scoring import CalcBEDROC

def inversion_old(kn_ligs,pred_ligs,invert):
    score_input = np.column_stack([kn_ligs,pred_ligs])

    if invert == False: # lower values are better
        indices = np.argsort(score_input[:,-1])
    else: # higher values are better (incl probabilities)
        indices = np.argsort(score_input[:,-1]*-1)
    
    score_input = score_input[indices] # ordered predicted scores w corresponding label
    return score_input

def inversion(kn_ligs,pred_ligs,invert):

    if invert == False: # lower values are better
        indices = np.argsort(pred_ligs)
        score_input = np.column_stack([kn_ligs,pred_ligs*-1])
    else: # higher values are better (incl probabilities)
        indices = np.argsort(pred_ligs*-1)
        score_input = np.column_stack([kn_ligs,pred_ligs])
    
    score_input = score_input[indices] # ordered predicted scores w corresponding label
    return score_input

# compute ROCAUC
def rocauc(kn_ligs,pred_ligs,invert=True):
    """Compute ROCAUC using sklearn.
    
    Args:
        kn_ligs (np.ndarray): known ligand indices.
        pred_ligs (np.ndarray): predicted ligand scores.
        invert (bool): whether scores for the metric are inverted (i.e., more
                       positive is better)

    Returns:
        tuple: containing auc and roc-curve data
        np.float: auc score for the input labels and predictions.
        np.ndarray: roc plot values for the input labels and predictions.
    """
    score_input = inversion(kn_ligs,pred_ligs,invert)
    curve_data = roc_curve(score_input[:,0],score_input[:,1])
    auc = roc_auc_score(score_input[:,0],score_input[:,1])
    return (auc, curve_data)


def topn(topn_value,kn_ligs,pred_ligs,invert=True):
    """Compute top-n % enrichment ratio.
    Input arguments kn_ligs, metric, and invert are required, and either pred_ligs
    or score_matrix is also required.
    
    Args:
        topn (int): top % to use when computing enrichment.
        kn_ligs (np.ndarray): known ligand indices.
        pred_ligs (np.ndarray): predicted ligand scores.
        invert (bool): whether scores for the metric are inverted (i.e., more
                       positive is better)

    Returns:
        float: enrichment factor, or ratio of known ligands in the top N% of ranked output.
    """

    score_input = inversion(kn_ligs,pred_ligs,invert)
    topn_cutoff = int(score_input.shape[0]*(float(topn_value)/100.0))
    enrichment = np.sum(score_input[:topn_cutoff,0])
    total = np.sum(kn_ligs)
    return (enrichment/total)


def bedroc(kn_ligs,pred_ligs,invert=True):
    """Compute BEDROC (Boltzmann enhanxced discrimination of ROC) score.
    Input arguments kn_ligs, metric, and invert are required, and either pred_ligs
    or score_matrix is also required.
    
    Args:
        kn_ligs (np.ndarray): known ligand indices.
        pred_ligs (np.ndarray): predicted ligand scores.
        invert (bool): whether scores for the metric are inverted (i.e., more 
                       positive is better)

    Returns:
        float: BEDROC score.
    """
    score_input = inversion(kn_ligs,pred_ligs,invert)
    
    bedroc = CalcBEDROC(score_input,0,5)
    
    return bedroc

def prauc(kn_ligs,pred_ligs,invert=True):
    """Compute PRAUC (precision-recall AUC) score.
    
    Args:
        kn_ligs (np.ndarray): known ligand indices.
        pred_ligs (np.ndarray): predicted ligand scores.
        invert (bool): whether scores for the metric are inverted (i.e., more
                       positive is better)

    Returns:
        tuple: containing auc and roc-curve data
        np.float: prauc score for the input labels and predictions.
        np.ndarray: pr plot values for the input labels and predictions.
    """
    score_input = inversion(kn_ligs,pred_ligs,invert)

    curve_data = precision_recall_curve(score_input[:,0],score_input[:,1])
    prauc = auc(curve_data[1],curve_data[0])
    return (prauc, curve_data)
