# structured_optimal_policies

This repository includes the code and data sets used in ***An Analysis of Structured Optimal Policies for Hypertension Treatment
Planning: The Tradeoff Between Optimality and Interpretability***. The repositoy contains three main directories.

## A. Data
This directory includes the input files for the analyses conducted in the paper. The directory contains the following files:
1. lifedata.csv - this file contains the life expectancy for adults in the USA with ages 40 to 60
3. strokedeathdata.csv - this file contains the likelihood of death due to a stroke for adults in the USA with ages 40 to 60
4. chddeathdata.csv - this file contains the likelihood of death due to a coronary heart disease (CHD) event for adults in the USA with ages 40 to 60
5. alldeathdata.csv - this file contains the likelihood of death not related to atherosclerotic cardiovascular disease for adults in the USA with ages 40 to 60
6. Continuous NHANES - this sub-directory contains the following files:
    - Continuous_NHANES_Data_Extraction_and_Filtering.R - this R script contains our code to obtain our final patient dataset from the files publicly-available at: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx
    - Continuous NHANES Forecasted Dataset.csv - this file is the end result of the Continuous_NHANES_Data_Extraction_and_Filtering.R script, which was used as the patient dataset in our analyses
  
## B. Analyses
This directory includes the scripts used to conduct the analyses in the paper. The directory constains the following files:
1. hypertension_treatment_monotone_mdp.py - this Python script serves as the **master file** for our analyses. Interested readers are encouraged to start with this file.
2. patient_simulation.py - this script is essentially a continuation of the hypertension_treatment_monotone_mdp.py file. It was made a function to allow for parallel computing.
