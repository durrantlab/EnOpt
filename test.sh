#!/bin/bash

# no optimization
python ensemble_optimizer.py -f demo_files/demo_input_scores.csv --outFile basic_test

# defaults
python ensemble_optimizer.py -f demo_files/demo_input_scores.csv -l demo_files/demo_input_known.csv \
	--weightedScore \
	--hyperparameterOpt \
	--outFile default_params_test


# all options
python ensemble_optimizer.py -f demo_files/demo_input_scores.csv -l demo_files/demo_input_known.csv \
	--scoringScheme rA \
	--weightedScore \
	--invertScoreSign \
	--optimizationMethod RF \
	--topConformations 5 \
	--treeParameterDict demo_files/demo_tree_params.json \
	--outFile all_params_test

