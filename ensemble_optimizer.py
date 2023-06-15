# main file for ensemble optimizer
import numpy as np
import pandas as pd

from EnsembleOptimizer import inputs
from EnsembleOptimizer import scoring 
from EnsembleOptimizer import output
from EnsembleOptimizer import testing

# read input (commandline, generate input dataframe)
# from input.py
def load_data():
    argparser = inputs.create_argparser()
    args = inputs.handle_command_line(argparser)
    score_data = inputs.read_input(args.file,args.known_ligs)
    return score_data, args

# based on commline options, generate scores  
# from scoring.py
# Options: (1) scores or ranks, (2) weighted or unweighted
def generate_scores(dataframe,known_ligs,args):
    scoring_dict = {'RF': scoring.get_weights_RF, 'XGB': scoring.get_weights_XGB}
    if args.weighted_score == True:
        score_matrix, weights, pred, aucs = scoring_dict[args.opt_method](dataframe,known_ligs,args.scoring_scheme)
    elif args.weighted_score == False:
        score_matrix = scoring.get_unweighted(dataframe,args.scoring_scheme)
        weights = None
        pred = None
        aucs = None
    return (score_matrix, weights, pred, aucs)

def single_confs(score_data,args):
    # single conformations
    pred_list = []
    knowns_empty = []
    for i in score_data[0].columns[1:-1]:
        sc_score_data = score_data[0][['Ligand',i]]
        score_matrix, weights, pred, aucs = generate_scores(sc_score_data,knowns_empty,args)
        pred_list.append(testing.rocauc(score_data[1],pred))
    return pred_list


def main(test=''):
    if test == 'sc':
        # Get the command-line parameters and load the ligand-score data.
        score_data, args = load_data() 

        # single conformation tests
        single_conf_list = single_confs(score_data,args)

        # score based on user input -- all conformations 
        score_matrix, weights, pred = generate_scores(score_data[0],[],args)
        # add all confs data to end
        single_conf_list.append(testing.rocauc(score_data[1],pred))
       
        # output
        np.save(args.out_file+'_aucdat.npy',np.array(single_conf_list,dtype=object))
    
    
    #no test main
    else:
        # Get the command-line parameters and load the ligand-score data.
        score_data, args = load_data() 
        
        # score based on user input -- all conformations 
        score_matrix, weights, pred, aucs = generate_scores(score_data[0],score_data[1],args)

        # output
        if args.weighted_score:
            output.organize_output(score_data,score_matrix,weights,pred,aucs,args)
        else:
            output.write_matrix(score_matrix,prefix=args.out_file)

    return 0

if __name__ == "__main__":
    main()
