# file writing and plot saving
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as pg
import plotly.colors as pc
from plotly.subplots import make_subplots


def write_matrix(matrix,prefix='enopt',weights_index=False):
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

# ranked scores based ONLY on scoring scheme
def output_scores_ranked(score_matrix,args):
    """Output potential ligands, ranked by scoring scheme only.
    
    Args:
        score_matrix (np.ndarray): scoring scheme and weighted score data.
        args (Namespace): user-provided arguments.

    Returns:
        pd.DataFrame: final_scores, all data sorted by scoring scheme, ranked.
    """
    if args.invert_score_sign is True:
        final_scores = score_matrix.sort_values(by=args.scoring_scheme,ascending=False)
    elif args.invert_score_sign is False:
        final_scores = score_matrix.sort_values(by=args.scoring_scheme,ascending=True)
        
    write_matrix(final_scores,prefix=args.out_file)
    return final_scores

# output potential ligands ranked by predicted probability
def output_ligands_ranked(score_data,score_matrix,weights,pred,model,args):
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
    final_scores = pd.concat([score_data[0],pd.Series(score_matrix,name=args.opt_method),pd.Series(pred,name='Predicted probability'),pd.Series(model,name='Prediction source model')],axis=1)
    unknown_scores = final_scores[np.invert(score_data[1].astype(bool))].sort_values(by='Predicted probability',ascending=False)
    known_scores = final_scores[score_data[1].astype(bool)].sort_values(by='Predicted probability',ascending=False)
    final_scores = final_scores.sort_values(by='Predicted probability',ascending=False)
    write_matrix(final_scores,prefix=args.out_file)
    return final_scores, known_scores, unknown_scores

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
    confs_weights_out = pd.DataFrame(index=['Conformation weight Model 1','Best subens. Model 1','Conformation weight Model 2','Best subens. Model 2','Conformation weight Model 3','Best subens. Model 3'],columns=weights.columns)
    for i in range(len(weights)):
        topn = np.argsort(weights.iloc[i,:])[-(int(args.topn_confs)):]
        confs_rank = np.zeros(len(weights.iloc[i,:])).astype(bool)
        confs_rank[topn] = True
        confs_weights_out.iloc[i*2,:] = weights.iloc[i,:]
        confs_weights_out.iloc[i*2+1,:] = confs_rank
    write_matrix(confs_weights_out,prefix=args.out_file+'_conformations',weights_index=True)
    return confs_weights_out.iloc[[1,3,5],:].sum(axis=0)

# make output of scoring include auc data
def interactive_summary(known_scores,unknown_scores,score_matrix,conf_weights,aucs,args):
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
                        subplot_titles=["Top %s Compounds Predicted Active Probability"%(str(args.top_known_out+args.top_unknown_out)),
                                        "Top %s Compounds Docking Score Distribution"%(str(args.top_known_out+args.top_unknown_out)),
                                        'Best-score Frequency Per Conformation',
                                        ' '])
    fig.update_layout(width=950,height=2000)

    # top N% compound Prob. values (Bar) and ranges (Box)
    topn_known = known_scores[:args.top_known_out]
    topn_unknown = unknown_scores[:args.top_unknown_out]
    topn = pd.concat([topn_known,topn_unknown]).sort_values(by='Predicted probability',ascending=False)
    bar_hover = []
    box_hover = []
    color_index = 0
    for lig in topn.itertuples():
        prob = lig[-2]
        ea = lig[-3]
        eb = np.max(lig[2:-4])
        bar_hover = ["Predicted probability: %s<br> Ensemble average: %s<br> Ensemble best: %s"%(round(prob,4), round(ea,3), round(eb,3))]

        ligs_color = pc.qualitative.Light24[color_index%(len(pc.qualitative.Light24))]
        fig.add_trace(pg.Bar(x=[lig[1]],name=lig[1],y=[prob],hoverinfo='text',hovertext=bar_hover,marker_color=ligs_color),row=1,col=1)
        fig.add_trace(pg.Box(name=lig[1],y=lig[2:-4],hoverinfo='y',marker_color=ligs_color),row=2,col=1)
        color_index +=1 
        
        #fig.add_trace(pg.Box(x0=lig[1],y=lig[2:-3],hoverinfo='y',hovertext=box_hover),row=2,col=1)

    #fig.add_trace(pg.Bar(x=topn['Ligand'],y=topn['Predicted probability'],hoverinfo='text',hovertext=bar_hover),row=1,col=1)
    
    # top 3 conformations info
    if args.invert_score_sign is True:
        top_conf_counts = score_matrix[score_matrix.columns[1:-4]].idxmax(axis=1).value_counts()
    else:
        top_conf_counts = score_matrix[score_matrix.columns[1:-4]].idxmin(axis=1).value_counts()
    
    #confs_dict = {'colors':{'3': 'limegreen', '2': 'limegreen', '1': 'yellow', '0': 'orangered'},
    #              'text':{'True': 'Top %s predictive conformations (from tree model)'%(args.topn_confs), 'False': None}}
    confs_dict = {'colors':{'True': 'limegreen', 'False': 'orangered'},
                  'text':{'True': 'Top %s predictive conformations (from tree model)'%(args.topn_confs), 'False': None}}
    confs_max = conf_weights.argsort().values >= (len(conf_weights)-int(args.topn_confs))
    
    confs_hover = []
    confs_color = []
    for i in range(len(conf_weights)):
        confs_hover.append(confs_dict['text'][str(confs_max[i])])
        confs_color.append(confs_dict['colors'][str(confs_max[i])])

    fig.add_trace(pg.Bar(x=top_conf_counts.index,y=top_conf_counts,marker_color=confs_color,hoverinfo='text',hovertext=confs_hover),row=3,col=1)

    if aucs is not None:
        # ROCAUC of fitted tree model (if applicable)
        fig.add_trace(pg.Bar(x=['Model 1','Model 2','Model 3'],y=np.round(aucs,3),hoverinfo='y'),row=4,col=1)

        fig.update_xaxes(title_text='Tree model (from 3-fold CV)', row=4,col=1)
        fig.update_yaxes(title_text='AUC from included known ligands', row=4,col=1)

        fig.layout.annotations[3].update(text='AUCs of Tree Models')

    fig.update_traces(showlegend=False)

    fig.update_xaxes(title_text="Compound (of top %s)"%(str(args.top_known_out+args.top_unknown_out)), row=1,col=1)
    fig.update_xaxes(title_text="Compound (of top %s)"%(str(args.top_known_out+args.top_unknown_out)), row=2,col=1)
    fig.update_xaxes(title_text='Protein conformation', row=3,col=1)

    fig.update_yaxes(title_text='Predicted probability', row=1,col=1)
    fig.update_yaxes(title_text='Docking score', row=2,col=1)
    fig.update_yaxes(title_text='Frequency of lowest conformation score', row=3,col=1)
    
    # export as html (interactive)
    fig.write_html(args.out_file+'_interactive_summary.html',include_plotlyjs='cdn')
    
    fig.show()

def organize_output(score_data,score_matrix,weights,pred,aucs,model,args):
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
    ranked_scores, ranked_knowns, ranked_unknowns = output_ligands_ranked(score_data,score_matrix,weights,pred,model,args)

    # output weights table (all confs), csv
    best_confs = output_best_confs(score_data,weights,args)

    # save CV data (for knowns only)
    auc_out = pd.DataFrame(aucs,index=['Model 1','Model 2','Model 3'],columns=['AUROC','PRAUC','BEDROC','Enrichment Factor'])
    auc_out.to_csv(args.out_file+'_cv.csv')
    
    # output image summary
    # currently only includes the non-actives
    interactive_summary(ranked_knowns,ranked_unknowns,ranked_scores,best_confs,auc_out['AUROC'],args)


    
