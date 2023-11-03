# command line and input file reading

import argparse
import json
import os
import sys

import numpy as np 
import pandas as pd 

### COMMAND LINE ###
def create_argparser():
    """Make argparser and define all arguments.

    Returns:
        argparser: Command-line or json argument parser.
    """
    # create argument parser
    optimizer_args = argparse.ArgumentParser(description = "Given an input \
        matrix of docking scores, containing rows of ligands (both known and \
        unknown) against a protein in several conformations, run multiple \
        mathematical and statistical functions to potentially discover new \
        ligands from known ligands. By default, all input and output data is \
        saved in CSV format.")
   
    ### REQUIRED  ###
    required_args = optimizer_args.add_argument_group("Required Arguments")

    required_args.add_argument("-f", "--file", required = True, help = "The \
    CSV file containing the docking matrix that will be analyzed.", 
        action = 'store', dest = "file")
    
    ### IN/OUT ###
    in_out_args = optimizer_args.add_argument_group('input_output')

    in_out_args.add_argument("--outFile", help = "Prefix of output file.",
        action = "store", dest = "out_file")
    
    in_out_args.add_argument("-l", "--knownLigs", required = False, 
        help = "File containing names of known ligands separated by commas.",
        action = 'store', dest = "known_ligs")
    
    ### SCORING ###
    scoring_args = optimizer_args.add_argument_group('scoring')

    # create optional flag for weight optimization scoring function choice
    scoring_args.add_argument("--scoringScheme", help = "Scoring scheme to use for \
        combining scores across conformations. One of 'eA', 'eB', 'rA', or 'rB'. \
        'eA' uses the average score across all conformations in the ensemble. 'eB' uses \
        the best score across all conformations. 'rA' uses the average of the score rank \
        for each conformation. 'rB' uses the best-ranked score across all conformations. \
        Default: eA.",
        action = "store", dest = "scoring_scheme")
    
    # create optional flag for output of weighted ensemble score functions
    scoring_args.add_argument("--weightedScore", help = "Whether or not to \
        compute weights optimized using tree models. Optimization is done \
        using known ligands if included, and score rankings if not included.",
        action = "store_true", dest = "weighted_score")

    scoring_args.add_argument("--invertScoreSign", help = "Whether to use \
        higher (more positive) scores as describing stronger binding. This is \
        dependent on the docking system used; for example, smina uses more negative \
        scores to represent stronger binding. Default: False (meaning that more negative \
        scores represent stronger binding).",
        action = "store_true", dest = "invert_score_sign")
    
    ### OPTIMIZATION ###
    optimization_args = optimizer_args.add_argument_group('optimization')

    optimization_args.add_argument("--optimizationMethod", help = "Method to determine \
        weighted scores. One of 'RF' (Random Forest) or 'XGB' (Gradient-boosted trees). \
        Default: RF.",
        action = "store", dest = "opt_method")

    optimization_args.add_argument("--topConformations", help = "Number of top conformations \
        to include in the 'best subensemble'. \
        Default: 3.",
        action = "store", dest = "topn_confs")
    

    # return the argument parser
    return optimizer_args


def handle_command_line(argument_parser):
    """Check for errors/inconsistencies and set defaults for args.

    Reads args and ensures that required inputs are in the correct 
    format. Sets defaults for non-required inputs.
    
    Args:
        argument_parser (argparser): Passed from create_argparser when
                                     reading input.

    Returns:
        namespace: argument parser with defaults set. Dict formatted.
    """
    # check args for inconsistencies or issues
    args = argument_parser.parse_args()

    # inputs: file exists and is correct format
    if not os.path.exists(args.file) or not ".csv" in args.file:
        print("Score matrix file is required and must be in .csv file format.")
        sys.exit(0) 

    # check known ligands param
    if args.known_ligs == None:
        print("It is highly recommended to include known ligands in your\
                dataset. If you do not include known ligands, EnOpt will\
                not be able to train a tree classifier to identify potential\
                novel actives.")
        args.weighted_score = False
    if args.known_ligs != None and not ".csv" in args.known_ligs:
        print("Known ligands file must be in .csv format.")
        sys.exit(0)

    # scoring: scoring scheme is properly specified, or set default
    if args.scoring_scheme is None:
    	args.scoring_scheme = 'eA'
    if args.scoring_scheme != None and args.scoring_scheme not in ['eA','eB','rA','rB']:
        print("Scoring scheme must be one of 'eA' (Ensemble Average), \
            'eB' (Ensemble Best), 'rA' (Ranked Average), or 'rB' (Ranked Best).")
        sys.exit(0)

    if args.invert_score_sign is None:
        args.invert_score_sign = False

    # optimization: method is properly specified, or set default
    if args.opt_method is None:
        args.opt_method = 'RF'
    elif args.opt_method not in ['RF','XGB']:
        print("Optimization method must be one of 'RF' or 'XGB'.")
        sys.exit(0)
    else:
        if args.known_ligs != None and args.weighted_score is False:
            args.weighted_score = True

    if args.topn_confs is None:
        args.topn_confs = 3
    else:
        try:
            int(args.topn_confs)
        except:
            print("Top N conformations must specify an integer. Default value is 3.")
            sys.exit(0)

    # output: set default if not specified
    if args.out_file is None:
        args.out_file = "enopt"

    print(vars(args))
    return args 

### INPUT FILE ###

def read_input(filename,known_ligs):
    """Reads input csv and known ligands, constructs docking matrix 
    and known ligand list.

    If no knowns, produces boolean mask of all zeros. 

    Args:
        filename (str): path to file containing docking matrix.
        known_ligs (str): path to file containing list of known
                             ligands.

    Returns:
        tuple: of docking score matrix and known ligands, containing
        pd.DataFrame: pd format of input scoring matrix.
        np.array: mask of 0/1 indicating known ligands by index in 
                  docking score dataframe.
    """
    # reads input csv and known ligands, constructs docking matrix and known ligand list
    docking_score_matrix = pd.read_csv(filename)    
    knowns_list = []
    if known_ligs != None:
        known_ligs_list = pd.read_csv(known_ligs,header=0)
        for line in docking_score_matrix.iloc[:,0]:
            if line in known_ligs_list.values.astype('str'):
                knowns_list.append(True)
            else:
                knowns_list.append(False)

        knowns_list = np.array(knowns_list).astype(int)
    else:
        knowns_list = np.zeros(len(docking_score_matrix))
    return (docking_score_matrix, knowns_list)

    
