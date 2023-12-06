# EnOpt
Ensemble Optimizer (EnOpt) is a fast, accessible tool that
streamlines ensemble-docking and consensus-score analysis. EnOpt takes as input a matrix of docking scores
from an ensemble virtual screen, organized as compounds (rows) X protein conformations (columns), and uses 
simple, interpretable machine learning to identify most-predictive subensembles and an ensemble composite score. 

## Usage and options
Prior to using EnOpt, you will need to ensure that you have all a python enviroment with all necessary packages (such as NumPy, Pandas, SciPy, etc.)
installed and accessible. We have provided a conda specification file to make it easier to set up an environment with all necessary packages.
To do so, you can use the command `conda create --name [environment name] --file conda_spec_file.txt`

`python ensemble_optimizer.py --help` will print a guide with all standard options and their usage.

### Simple usage 
The simplest use of EnOpt uses the command `python ensembe_optimizer.py -f [input file matrix]`

### Standard options
**Required:**

-f INPUT\_FILE
> Input CSV file containing the ensemble docking score matrix. Can be obtained by running `generate_score_matrix.py` in a nested directory of docking output.
See `generate_score_matrix.py` for details. 

**Input and output:**

--outFile OUT\_FILE                         
> Prefix of output file.

-l KNOWN\_LIGS, --knownLigs KNOWN\_LIGS     
> File containing names of known ligands separated by commas.

**Scoring:**

--scoringScheme SCORING\_SCHEME
> Scoring scheme to use for combining scores across conformations. One of 'eA', 'eB', 'rA', or 'rB'. 'eA' uses the average score across all conformations
in the ensemble. 'eB' uses the best score across all conformations. 'rA' uses the average of the score rank for each conformation. 'rB' uses the best-
ranked score across all conformations. Default: eA.

--weightedScore       
> Whether or not to compute weights optimized using tree models. Optimization is done using known ligands if included, and score rankings if not included.

--invertScoreSign     
> Whether to use higher (more positive) scores as describing stronger binding. This is dependent on the docking system used; for example, smina uses more negative scores to represent stronger binding. Default: False (meaning that more negative scores represent stronger binding).

**Optimization:**

--optimizationMethod OPT\_METHOD
> Method to determine weighted scores. One of 'RF' (Random Forest) or 'XGB' (Gradient-boosted trees). Default: RF.

--topConformations TOPN\_CONFS
> Number of top conformations to include in the 'best subensemble'. Default: 3.
input and output:
--outFile OUT\_FILE                         Prefix of output file.
-l KNOWN\_LIGS, --knownLigs KNOWN\_LIGS     File containing names of known ligands separated by commas.

## Paper/lab link
Find more tools for analysis of protein-ligand binding at https://durrantlab.pitt.edu/durrant-lab-software/. 

## Contact info
For questions, suggestions, or problems with the tool contact Roshni Bhatt at rob108@pitt.edu. 

## Acknowledgements
This work was supported by the National Institute of Health (1R01GM132353-01) and the University of Pittsburgh’s Center for Research Computing, RRID:SCR\_022735 (supported by NSFOAC-2117681). We would like to thank Yogindra Raghav for his contributions in generating initial proof-of-concept code. We also thank Darian Yang for assistance in collating and pruning ideas.

