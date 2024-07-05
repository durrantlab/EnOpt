# testing functions used in evaluation and benchmarking
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

try:
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
    from rdkit.ML.Scoring.Scoring import CalcEnrichment
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


def topn(topn_value,kn_ligs,pred_ligs=None,score_matrix=None,metric='Predicted probability',invert=False):
    """Compute top-n % enrichment ratio.
    Input arguments kn_ligs, metric, and invert are required, and either pred_ligs
    or score_matrix is also required.
    
    Args:
        score_matrix (pd.DataFrame): predicted prob. and weighted score matrix.
        kn_ligs (np.ndarray): known ligand indices.
        topn (int): top % to use when computing enrichment.
        metric (str): column of score matrix (eA/eB/rA/rB or predicted prob.)
                      used to determined top N.
        invert (bool): whether scores for the metric are inverted (i.e., more
                       positive is better)

    Returns:
        float: ratio of known ligands in the top N% of ranked output.
    """
    if pred_ligs is None:
        # computes true positives in top n
        lig_ranking = np.argsort(score_matrix[metric])+1
    else:
        lig_ranking = np.argsort(pred_ligs)+1 
    
    if metric == 'Predicted probability' or invert == True:
        # highest n probabilities
        topn_cutoff = (float(topn_value)/100.0)*len(kn_ligs)
        top_n_mask = lig_ranking < topn_cutoff
        top_n_enriched = np.sum(np.multiply(top_n_mask,kn_ligs))

    else: 
        # lowest n scores
        topn_cutoff = (1-(float(topn_value)/100.0))*len(kn_ligs)
        top_n_mask = lig_ranking > topn_cutoff
        top_n_enriched = np.sum(np.multiply(top_n_mask,kn_ligs))
    return top_n_enriched/np.sum(kn_ligs)


def bedroc(kn_ligs,pred_ligs=None,score_matrix=None,metric='Predicted probability',invert=False):
    """Compute BEDROC (Boltzmann enhanxced discrimination of ROC) score.
    Input arguments kn_ligs, metric, and invert are required, and either pred_ligs
    or score_matrix is also required.
    
    Args:
        kn_ligs (np.ndarray): known ligand indices.
        pred_ligs (np.ndarray): predicted ligand indices.
        score_matrix (pd.DataFrame): predicted prob. and weighted score matrix.
        metric (str): column of score matrix (eA/eB/rA/rB or predicted prob.)
        invert (bool): whether scores for the metric are inverted (i.e., more 
                       positive is better)

    Returns:
        float: BEDROC score.
    """
    if pred_ligs is None:
        known_labels = pd.Series(kn_ligs.astype(bool),name='Known label')
        score_input = pd.concat([score_matrix,known_labels],axis=1)
        if metric == 'Predicted probability' or invert == True:
            # highest n probabilities
            score_input = score_input.sort_values(by=metric,ascending=False)
        else: 
            # lowest n scores
            score_input = score_input.sort_values(by=metric,ascending=True)
        bedroc = CalcBEDROC(score_input[[metric,'Known label']].values,1,5)
    else:
        score_input = np.column_stack([kn_ligs,pred_ligs])
        indices = np.argsort(score_input[:,-1]*-1)
        score_input = score_input[indices]
        bedroc = CalcBEDROC(score_input,0,5)
    
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
    curve_data = precision_recall_curve(kn_ligs,pred_ligs)
    prauc = auc(curve_data[1],curve_data[0])
    return (prauc, curve_data)
