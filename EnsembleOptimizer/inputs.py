# command line and input file reading

import argparse
import json
import os
import sys

import numpy as np 
import pandas as pd 

### COMMAND LINE ###
def create_argparser():
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
    
    in_out_args.add_argument("-l", "--known_ligs", required = False, 
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

    ### OPTIMIZATION ###
    optimization_args = optimizer_args.add_argument_group('optimization')

    optimization_args.add_argument("--optimizationMethod", help = "Method to determine \
        weighted scores. One of 'RF' (Random Forest) or 'XGB' (Gradient-boosted trees). \
        Default: RF.",
        action = "store", dest = "opt_method")

    # return the argument parser
    return optimizer_args


def handle_command_line(argument_parser):
    # check args for inconsistencies or issues
    args = argument_parser.parse_args()

    # inputs: file exists and is correct format
    if not os.path.exists(args.file) or not ".csv" in args.file:
        print("Score matrix file is required and must be in .csv file format.")
        sys.exit(0) 

    # check known ligands param
    if args.known_ligs != None and not ".csv" in args.known_ligs:
        print("Known ligands file must be in .csv format.")
        sys.exit(0)

    # scoring: scoring scheme is properly specified, or set default
    if args.scoring_scheme is None:
    	args.scoring_scheme = 'eA'
    elif args.scoring_scheme not in ['eA','eB','rA','rB']:
        print("Scoring scheme must be one of 'eA' (Ensemble Average), \
            'eB' (Ensemble Best), 'rA' (Ranked Average), or 'rB' (Ranked Best).")
        sys.exit(0)

    # optimization: method is properly specified, or set default
    if args.opt_method is None:
        args.opt_method = 'RF'
    elif args.opt_method not in ['RF','XGB']:
        print("Performance metric must be one of 'RF' or 'XGB'.")
        sys.exit(0)
    else:
        if args.weighted_score is False:
            args.weighted_score = True

    # output: set default if not specified
    if args.out_file is None:
        args.out_file = "enopt_"

    print(vars(args))
    return args 

### INPUT FILE ###

def read_input(filename,known_ligs):
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

    
