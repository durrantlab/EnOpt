# testing functions used in evaluation and benchmarking
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# compute ROCAUC
def rocauc(kn_ligs,pred_ligs):
    """Compute ROCAUC using sklearn.
    
    Args:
        kn_ligs (np.ndarray): known ligand indices.
        pred_ligs (np.ndarray): predicted ligand indices.

    Returns:
        tuple: containing auc and roc-curve data
        np.ndarray: auc score for the input labels and predictions.
        np.ndarray: roc plot values for the input labels and predictions.
    """
    curve_data = roc_curve(kn_ligs,pred_ligs)
    auc = roc_auc_score(kn_ligs,pred_ligs)
    return (auc, curve_data)

def topn(score_matrix,kn_ligs,metric):
    """Compute top-n (3%) enrichment ratio.
    
    Args:
        score_matrix (pd.DataFrame): predicted prob. and weighted score matrix.
        kn_ligs (np.ndarray): known ligand indices.
        metric (str): column of score matrix (eA/eB/rA/rB or predicted prob.)
                      used to determined top N.

    Returns:
        float: ratio of known ligands in the top 3% of ranked output.
    """
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


def unknown_ligs_method(unw_frame,args,top_method='top_n'):
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
        top_n = unw_frame[args.scoring_scheme].sort_values(ascending=(not args.invert_score_sign))[n_known]
        top_ligs = unw_frame[args.scoring_scheme] < top_n
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
        top_n = unw_frame[args.scoring_scheme].sort_values(ascending=(not args.invert_score_sign))[n_known]
        top_ligs = unw_frame[args.scoring_scheme] < top_n
    
    return top_ligs.astype(int)

def single_confs(score_data,args):
    """Single conformation testing code.

    Runs predictions given known ligands for each single-conformation
    in ensemble as if it were the full ensemble. 
    Solely used for testing heuristics and tree model efficacy.

    Args:
        score_data (tuple): Tuple of original docking score data, known ligands.
        args (arg Namespace): user-provided arguments.

    Returns:
        list: prediction aucs (used for testing) ordered by index from scoring dataframe.

    Notes:
        ONLY called in testing mode
    """
    # single conformations
    pred_list = []
    knowns_empty = []
    for i in score_data[0].columns[1:-1]:
        sc_score_data = score_data[0][['Ligand',i]]
        score_matrix, weights, pred, aucs = generate_scores(sc_score_data,knowns_empty,args)
        pred_list.append(rocauc(score_data[1],pred))
    return pred_list
