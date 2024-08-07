# testing functions used in evaluation and benchmarking
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from rdkit.ML.Scoring.Scoring import CalcBEDROC


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

def topn(topn_value, kn_ligs, pred_ligs, invert=True):
    """Compute top-n % enrichment factor.
    
    This function calculates the enrichment factor for the top N% of compounds.
    The enrichment factor is the ratio of the proportion of actives in the top N%
    to the proportion of actives in the entire dataset.
    
    Args:
        topn_value (int): Top percentage to use when computing enrichment (e.g., 20 for top 20%).
        kn_ligs (np.ndarray): Known ligand indicators (1 for active, 0 for inactive).
        pred_ligs (np.ndarray): Predicted ligand scores.
        invert (bool): Whether scores for the metric are inverted (i.e., more positive is better).
                       Default is True.

    Returns:
        float: Enrichment factor. A value greater than 1 indicates better than random performance.
    """

    # Invert scores if necessary (assuming inversion function is defined elsewhere)
    score_input = inversion(kn_ligs, pred_ligs, invert)

    # Calculate total number of compounds
    total_compounds = score_input.shape[0]
    
    # Calculate the index cutoff for the top N%
    topn_cutoff = int(total_compounds * (float(topn_value) / 100.0))
    
    # Count the number of active compounds in the top N%
    actives_in_top = np.sum(score_input[:topn_cutoff, 0])
    
    # Count the total number of active compounds
    total_actives = np.sum(kn_ligs)
    
    # Calculate the fraction corresponding to the top N%
    fraction_top = topn_value / 100.0
    
    # Calculate the enrichment factor
    # (proportion of actives in top N%) / (proportion of actives in entire dataset)
    ef = (actives_in_top / (total_compounds * fraction_top)) / (total_actives / total_compounds)
    
    return ef



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

    # Save to tmp.csv
    # np.savetxt('tmp.csv',score_input,delimiter=',')
    # print(score_input)
    # import pdb; pdb.set_trace()
    
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
