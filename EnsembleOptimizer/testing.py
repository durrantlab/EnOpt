# testing functions used in evaluation and benchmarking
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# compute ROCAUC
def rocauc(kn_ligs,pred_ligs):
    curve_data = roc_curve(kn_ligs,pred_ligs)
    auc = roc_auc_score(kn_ligs,pred_ligs)
    return (auc, curve_data)

def topn(score_matrix,kn_ligs,metric):
    # computes true positives in top n
    lig_ranking = np.argsort(score_matrix[metric])
    if metric == 'Predicted probability':
        # highest n probabilities
        topn_cutoff = 0.97*len(kn_ligs)
        top_n_enriched = np.sum(np.multiply(lig_ranking,kn_ligs) > topn_cutoff)
    else: 
        # lowest n scores
        topn_cutoff = 0.03*len(kn_ligs)
        top_n_enriched = np.sum(np.multiply(lig_ranking,kn_ligs) < topn_cutoff)

    return top_n_enriched/topn_cutoff


