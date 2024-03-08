# EnOpt

Ensemble Optimizer (EnOpt) is a fast, accessible tool that streamlines
ensemble-docking and consensus-score analysis. EnOpt takes as input a matrix of
docking scores from an ensemble virtual screen, organized as compounds (rows) X
protein conformations (columns). It uses simple, interpretable machine learning
to identify most-predictive subensembles and an ensemble composite score.

## Setup

Before using EnOpt, ensure that you have installed a python enviroment with all
necessary packages (e.g., NumPy, Pandas, SciPy, etc.). We have provided a conda
specification file to make it easier to set up an environment with all necessary
packages:

```bash
conda create --name [environment name] --file conda_spec_file.txt
```

To print a guide with all standard options and their usage:

```bash
python ensemble_optimizer.py --help
```

## Simple usage

An example of the simplest use of EnOpt:

```bash
python ensembe_optimizer.py -f [input file matrix]
```

## Options

### Input options

The input CSV file containing the ensemble docking score matrix (required):

```
-f INPUT_FILE
```

A file containing the names of known ligands, separated by commas:

```
-l KNOWN_LIGS, --knownLigs KNOWN_LIGS
```

### Output options

The prefix of the output file:

```
--outFile OUT_FILE
```

### Scoring options

The scoring scheme to use for combining scores across conformations:

```
--scoringScheme SCORING_SCHEME
```

<sup>(One of "eA", "eB", "rA", or "rB". "eA" uses the average score across all
conformations in the ensemble. "eB" uses the best score across all
conformations. "rA" uses the average of the score rank for each conformation.
"rB" uses the best-ranked score across all conformations. Default: eA.)</sup>

Whether to compute weights optimized using tree models:

```
--weightedScore
```

<sup>(EnOpt performs optimization using known ligands if included. Otherwise, it
uses score rankings; not recommended.)</sup>

Whether higher (more positive) scores describing stronger binding:

```
--invertScoreSign
```

<sup>(The scheme depends on the docking program used; for example, smina uses
more negative scores to represent stronger binding. Default: False, meaning that
more negative scores represent stronger binding.)</sup>

### Optimization options

Method to determine weighted scores:

```
--optimizationMethod OPT_METHOD
```

<sup>(One of "RF", Random Forest, or "XGB", Gradient-boosted trees. Default: RF.)</sup>

Number of top conformations to include in the "best subensemble":

```
--topConformations TOPN_CONFS
```

<sup>(Default: 3)</sup>

<!-- input and output:

```
--outFile OUT_FILE                         Prefix of output file.
-l KNOWN_LIGS, --knownLigs KNOWN_LIGS     File containing names of known ligands separated by commas.
``` -->

## Paper/lab link

Find more tools for analysis of protein-ligand binding at
[https://durrantlab.pitt.edu/durrant-lab-software/](https://durrantlab.pitt.edu/durrant-lab-software/).

## Contact info

For questions, suggestions, or problems with the tool contact Roshni Bhatt at
[rob108@pitt.edu](mailto:rob108@pitt.edu).

## Acknowledgements

This work was supported by the National Institute of Health (1R01GM132353-01)
and the University of Pittsburgh's Center for Research Computing,
RRID:SCR\_022735 (supported by NSFOAC-2117681). We would like to thank Yogindra
Raghav for his contributions in generating initial proof-of-concept code. We
also thank Darian Yang for assistance in collating and pruning ideas.
