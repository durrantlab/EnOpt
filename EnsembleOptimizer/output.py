# file writing and plot saving
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as pg
from plotly.subplots import make_subplots


def write_matrix(matrix,prefix='output',weights_index=False):
    """Correctly format and write DataFrame or array files as csv.
    
    Args:
        matrix (pd.DataFrame or np.ndarray): matrix or vector object.
        prefix (str): user-provided output file name. Default 'output'.
        weights_index (bool): row index for dataframes. included for 
                              conformation weights output.
        
    Returns:
        None
    """
    if type(matrix) == pd.DataFrame or type(matrix) == pd.Series:
        matrix.to_csv(prefix+'.csv',index=weights_index)
    elif type(matrix) == np.ndarray:
        np.savetxt(prefix+'.csv',matrix,delimiter=',')

# output potential ligands ranked by predicted probability
def output_ligands_ranked(score_data,score_matrix,weights,pred,args):
    """Output potential ligands, ranked by predicted probability.
    
    Args:
        score_data (tuple): Tuple of original docking score data, known ligands.
        score_matrix (np.ndarray): scoring scheme and weighted score data.
        weights (np.ndarray): conformation weights from tree models.
        pred (np.ndarray): predicted probabilities from tree models.
        args (Namespace): user-provided arguments.

    Returns:
        pd.DataFrame: final_scores, all data sorted by predicted probability.
        pd.DataFrame: unknown_scores, outputs filtered to include only decoys/
                      unknown compounds. Also sorted by predicted probability.
    """
    final_scores = pd.concat([score_data[0],pd.Series(score_matrix,name=args.opt_method),pd.Series(pred,name='Predicted probability')],axis=1)
    unknown_scores = final_scores[np.invert(score_data[1].astype(bool))].sort_values(by='Predicted probability',ascending=False)
    final_scores = final_scores.sort_values(by='Predicted probability',ascending=False)
    write_matrix(final_scores,prefix=args.out_file)
    return final_scores, unknown_scores

# output conformation weights, ranked by feature importance in predictive model
def output_best_confs(score_data,weights,args):
    """Output top conformations from tree model feature importances.
    
    Args:
        score_data (tuple): Tuple of original docking score data, known ligands.
        weights (np.ndarray): conformation weights from tree models.
        args (Namespace): user-provided arguments.

    Returns:
        pd.DataFrame: conformations with weights and top N indication included.
    """
    top3 = np.argsort(weights)[-3:]
    confs_rank = np.zeros(len(weights)).astype(bool)
    confs_rank[top3] = True
    confs_rank = pd.Series(confs_rank,index=weights.index,name='Best subensemble (3)')
   
    weights.name = 'Conformation weight from '+args.opt_method
    confs_weights_out = pd.concat([weights,confs_rank],axis=1)
    write_matrix(confs_weights_out,prefix=args.out_file+'_conformations',weights_index=True)
    return confs_weights_out

# make output of scoring include auc data
def interactive_summary(score_matrix,conf_weights,aucs,args):
    """Output interactive summary of ensemble optimization results.

    Includes: top 20 compound probability values with hover info;
              top 20 compound docking score distributions w hover info;
              best-scoring conformations (with top 3 most predictive indicated);
              (where applicable) AUC of tree models from 3-fold CV.
    
    Args:
        score_matrix (pd.DataFrame): "final" (ranked, score/probability data included) matrix.
        conf_weights (np.ndarray): conformation weights from tree models.
        aucs (np.ndarray): ROCAUC values from 3-fold tree model cross validation.
        args (Namespace): user-provided arguments.

    Returns:
        None: outputs html file of interactive summary. 
    """
    fig = make_subplots(4,1,
                        subplot_titles=['Top 20 Compounds Predicted Active Probability',
                                        'Top 20 Compounds Docking Score Distribution',
                                        'Best-score Frequency Per Conformation',
                                        ' '])
    fig.update_layout(width=950,height=2000)

    # top N% compound Prob. values (Bar) and ranges (Box)
    topn = score_matrix[:20]
    bar_hover = []
    box_hover = []
    for lig in topn.itertuples():
        prob = lig[-1]
        ea = lig[-2]
        eb = np.max(lig[2:-3])
        bar_hover.append("Predicted probability: %s\n Ensemble average: %s\n Ensemble best: %s"%(prob, ea, eb))
        
        fig.add_trace(pg.Box(x0=lig[1],y=lig[2:-3],hoverinfo='y',hovertext=box_hover),row=2,col=1)

    fig.add_trace(pg.Bar(x=topn['Ligand'],y=topn['Predicted probability'],hoverinfo='text',hovertext=bar_hover),row=1,col=1)

    # top 3 conformations info
    top_conf_counts = score_matrix[score_matrix.columns[2:-3]].idxmin(axis=1).value_counts()
    confs_dict = {'colors':{'True': 'limegreen', 'False': 'orangered'},
                  'text':{'True': 'Top 3 predicitive conformations (from tree model)', 'False': None}}
    confs_hover = []
    confs_color = []
    for i in conf_weights['Best subensemble (3)']:
        confs_hover.append(confs_dict['text'][str(i)])
        confs_color.append(confs_dict['colors'][str(i)])

    fig.add_trace(pg.Bar(x=top_conf_counts.index,y=top_conf_counts,marker_color=confs_color,hoverinfo='text',hovertext=confs_hover),row=3,col=1)

    if aucs is not None:
        # ROCAUC of fitted tree model (if applicable)
        fig.add_trace(pg.Bar(x=['Model 1','Model 2','Model 3'],y=aucs,hoverinfo='y'),row=4,col=1)

        fig.update_xaxes(title_text='Tree model (from 3-fold CV)', row=4,col=1)
        fig.update_yaxes(title_text='AUC from included known ligands', row=4,col=1)

        fig.layout.annotations[3].update(text='AUCs of Tree Models')

    fig.update_traces(showlegend=False)

    fig.update_xaxes(title_text='Compound (of top 20)', row=1,col=1)
    fig.update_xaxes(title_text='Compound (of top 20)', row=2,col=1)
    fig.update_xaxes(title_text='Protein conformation', row=3,col=1)

    fig.update_yaxes(title_text='Predicted probability', row=1,col=1)
    fig.update_yaxes(title_text='Docking score', row=2,col=1)
    fig.update_yaxes(title_text='Frequency of lowest conformation score', row=3,col=1)
    
    # export as html (interactive)
    fig.write_html(args.out_file+'_interactive_summary.html',include_plotlyjs='cdn')
    
    fig.show()

def organize_output(score_data,score_matrix,weights,pred,aucs,args):
    """Output interactive summary of ensemble optimization results.
    
    Wrapper for all output functions/organizing output.

    Includes: top 20 compound probability values with hover info;
              top 20 compound docking score distributions w hover info;
              best-scoring conformations (with top 3 most predictive indicated);
              (where applicable) AUC of tree models from 3-fold CV.
    
    Args:
        score_data (tuple): Tuple of original docking score data, known ligands.
        score_matrix (pd.DataFrame): "final" (ranked, score/probability data included) matrix.
        weights (np.ndarray): conformation weights from tree models.
        pred (np.ndarray): predicted probabilities from tree models.
        aucs (np.ndarray): ROCAUC values from 3-fold tree model cross validation.
        args (Namespace): user-provided arguments.

    Returns:
        None: writes all output files.
    """
    # for either knowns or no knowns 
    # output final score matrix file, csv
    ranked_scores, ranked_unknowns = output_ligands_ranked(score_data,score_matrix,weights,pred,args)

    # output weights table (all confs), csv
    best_confs = output_best_confs(score_data,weights,args)

    # save CV data (for knowns only)
    np.savetxt(args.out_file+'_cv.csv',aucs)
    
    # output image summary
    # currently only includes the non-actives
    interactive_summary(ranked_unknowns,best_confs,aucs,args)


    
