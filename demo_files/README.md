This directory contains demo and testing files for EnOpt.

# Example inputs:
`demo\_input\_scores.csv` is a file containing an example score matrix for input to EnOpt.
This score matrix was generated using structures and compounds of HXK4 from the DUD-E database.

`demo\_input\_knowns.csv` is a file containing an example list of positive controls (known ligands) for HXK4,
to be used with `demo/_input/_scores.csv` as input to EnOpt.

`demo\_tree\_params.json` is an example json file containing parameters for training the tree models. This is an
optional input to EnOpt for when hyperparameter optimization is not used.

# Example outputs:
`demo\_output.csv` is an example output file containing the input compounds as ranked by EnOpt.

`demo\_output\_conformations.csv` is an example output file containing the conformation importances as computed by EnOpt.

`demo\_output\_cv.csv` is an example output file containing 3-fold cross-validation AUROC values from EnOpt model training.

`demo\_output\_interactive\_summary.html` is an example output file containing the interactive summary from a run of EnOpt using the 
HXK4 screen inputs provided.

