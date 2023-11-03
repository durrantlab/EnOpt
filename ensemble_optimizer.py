# main file for ensemble optimizer
import numpy as np
import pandas as pd

from EnsembleOptimizer import inputs
from EnsembleOptimizer import scoring 
from EnsembleOptimizer import output
from EnsembleOptimizer import evaluation

# read input (commandline, generate input dataframe)
# from input.py
def load_data():
    """Loads tool paramters and inputs command line or json.

    Requires datafile only; all other parameters have defaults set.
    Called internally using inputs.py. 

    Returns:
        tuple (pandas DataFrame, numpy array): 
            contains docking score data in [0] and known ligand data in [1].
        argparser Namespace object: contains json or command line arguments.
    """
    argparser = inputs.create_argparser()
    args = inputs.handle_command_line(argparser)
    score_data = inputs.read_input(args.file,args.known_ligs)
    return score_data, args

# based on commline options, generate scores  
# from scoring.py
# Options: (1) scores or ranks, (2) weighted or unweighted
def generate_scores(dataframe,known_ligs,args):
    """Generates ensemble scores from docking matrix data.
        
    Generates ensemble score matrix and tree model weights/
    predictions based on user-specified optimization method (RF or XGB),
    scoring scheme (eA/eB/rA/rB), score format (negative/positive better),
    and inclusion of known ligands.
    Called internally using scoring.py.    

    Args:
        dataframe (pandas DataRrame): Docking score matrix in df format.
        known_ligs (numpy array): Mask of booleans (in int form)
                                  identifying known ligands by df index.
        args (arg Namespace): user-provided arguments.

    Returns:
        tuple (pandas DataFrame): includes weighted scores,
                                    conformation weights, 
                                    predicted probabilities,
                                    aucs.
    """
    scoring_dict = {'RF': scoring.get_weights_RF, 'XGB': scoring.get_weights_XGB}
    if args.weighted_score == True:
        score_matrix, weights, pred, aucs = scoring_dict[args.opt_method](dataframe,known_ligs,args)
    elif args.weighted_score == False:
        score_matrix = scoring.get_unweighted(dataframe,args)
        weights = None
        pred = None
        aucs = None
    return (score_matrix, weights, pred, aucs)


def main():
    """Runs Ensemble Optimizer. Calls all other methods internally based on options.

    Returns:
        None
    """
    # Get the command-line parameters and load the ligand-score data.
    score_data, args = load_data() 
    
    # score based on user input -- all conformations 
    score_matrix, weights, pred, aucs = generate_scores(score_data[0],score_data[1],args)

    # output
    if args.weighted_score:
        output.organize_output(score_data,score_matrix,weights,pred,aucs,args)
    else:
        output.output_scores_ranked(score_matrix,args)

    return 0

if __name__ == "__main__":
    main()
