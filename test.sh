#!/bin/bash

# no optimization
python ensemble_optimizer.py -f demo_files/demo_input_scores.csv --out_file basic_test

# defaults
python ensemble_optimizer.py -f demo_files/demo_input_scores.csv -l demo_files/demo_input_known.csv \
	--weighted_score \
	--hyperparam \
	--out_file default_params_test


# all options
python ensemble_optimizer.py -f demo_files/demo_input_scores.csv -l demo_files/demo_input_known.csv \
	--scoring_scheme rA \
	--weighted_score \
	--invert_score_sign \
	--opt_method RF \
	--topn_confs 5 \
	--tree_params demo_files/demo_tree_params.json \
	--out_file all_params_test

# json input
python ensemble_optimizer.py --json_input demo_files/demo_json_input.json


