This directory contains demo and testing files for EnOpt.

# Example inputs:
`demo_input_scores.csv` is a file containing an example score matrix for input to EnOpt.
This score matrix was generated using structures and compounds of HXK4 from the [DUD-E database](https://dude.docking.org/targets/hxk4).

`demo_input_knowns.csv` is a file containing an example list of positive controls (known ligands) for HXK4,
to be used with `demo_input_scores.csv` as input to EnOpt.

`demo_tree_params.json` is an example json file containing parameters for training the tree models. This is an
optional input to EnOpt for when hyperparameter optimization is not used.

`demo_json_input.json` is an example json file containing parameters for running EnOpt. This is an
alternative to the command line input format to EnOpt, and does not require any other flags.

# Example outputs:
`demo_output.csv` is an example output file containing the input compounds as ranked by EnOpt.

`demo_output_conformations.csv` is an example output file containing the conformation importances as computed by EnOpt.

`demo_output_cv.csv` is an example output file containing 3-fold cross-validation AUROC, PRAUC, BEDROC, and EF values from EnOpt model training.

`demo_output_single_conformations.csv` is an example output file containing the AUROC, PRAUC, BEDROC, and EF values for each single conformation from the ensemble used independently.

`demo_output_interactive_summary.html` is an example output file containing the interactive summary from a run of EnOpt using the 
HXK4 screen inputs provided.

