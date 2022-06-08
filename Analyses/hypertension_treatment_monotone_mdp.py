# ******************************************************
# Monotone Policies Case Study - Hypertension Treatment
# ******************************************************

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # array operations
import itertools as it  # recursive operations
import time as tm  # timing code
import patient_simulation as pt_sim # risk, transition probabilities, and policy calculations
import multiprocessing as mp  # parallel computations
import pickle as pk  # saving results

# Establishing directories (change to appropriate path)
home_dir = os.path.abspath(os.environ['HOME'] + '/Documents/Monotone Policies/Python')
data_dir = os.path.abspath(os.environ['HOME'] + '/Documents/Monotone Policies/Data')
results_dir = os.path.abspath(os.environ['HOME'] + '/Documents/Monotone Policies/Python/Results')
sens_dir = os.path.abspath(os.environ['HOME']+'/Documents/Monotone Policies/Python/Results/Sensitivity Analyses')
fig_dir = os.path.abspath(os.environ['HOME']+'/Documents/Monotone Policies/Python/Figures')

# =============
# Loading data
# =============

# Loading life expectancy and death likelihood data
# (first column age, second column male, third column female)
os.chdir(data_dir)
lifedata = pd.read_csv('lifedata.csv', header=None)
strokedeathdata = pd.read_csv('strokedeathdata.csv', header=None)
chddeathdata = pd.read_csv('chddeathdata.csv', header=None)
alldeathdata = pd.read_csv('alldeathdata.csv', header=None)

# Loading risk slopes (first column age, second column CHD, third column stroke)
riskslopedata = pd.read_csv('riskslopes.csv', header=None)

# Loading 2009-2016 Continuous NHANES dataset (ages 40-60)
os.chdir(data_dir + '/Continuous NHANES')
ptdata = pd.read_csv('Continuous NHANES Forecasted Dataset.csv') # using sampling weights as recorded in the NHANES dataset (to reduce computational burden)


# =======================
# Initializing parameters
# =======================

# Selecting number of cores for parallel processing
cores = mp.cpu_count() - 1

# Risk parameters
ascvd_hist_mult = [3, 3]  # multiplier to account for history of CHD and stroke, respectively

# Transition probability parameters
numhealth = 10  # Number of health states
years = 10  # Number of years (non-stationary stages)
events = 2  # Number of events considered in model
numeds = 5 # Maximum number of drugs combined

# Treatment parameters
sbpmin = 120  # minimum allowable SBP
dbpmin = 55  # minimum allowable DBP
sbpmax = 150  # maximum allowable untreated SBP
dbpmax = 90  # maximum allowable untreated DBP

# AHA's guideline parameters
targetrisk = 0.1
targetsbp = 130
targetdbp = 80

# Generating list of treatment options
##One drug at a time
drugs = ["ACE", "ARB", "BB", "CCB", "TH"]

##Two drugs simultaneously
drugcomb2 = it.combinations_with_replacement(drugs, 2)
drugcomb2 = np.array(list(drugcomb2))
drugcomb2 = np.sort(drugcomb2, axis=1)
drugcomb2 = drugcomb2[drugcomb2[:, 1].argsort(kind='mergesort')]
drugcomb2 = drugcomb2[drugcomb2[:, 0].argsort(kind='mergesort')]

##Three drugs simultaneously
drugcomb3 = np.array(np.meshgrid(drugs, drugs, drugs)).reshape(3, len(drugs) ** 3).T
drugcomb3 = np.sort(drugcomb3, axis=1)
drugcomb3 = np.unique(drugcomb3, axis=0)

##Four drugs simultaneously
drugcomb4 = np.array(np.meshgrid(drugs, drugs, drugs, drugs)).reshape(4, len(drugs) ** 4).T
drugcomb4 = np.sort(drugcomb4, axis=1)
drugcomb4 = np.unique(drugcomb4, axis=0)

##Five drugs simultaneously (could extend to n number of drugs - limited by having to repeat the "drugs" list inside np.meshgrid)
drugcomb5 = np.array(np.meshgrid(drugs, drugs, drugs, drugs, drugs)).reshape(5, len(drugs) ** 5).T
drugcomb5 = np.sort(drugcomb5, axis=1)
drugcomb5 = np.unique(drugcomb5, axis=0)

###Removing potentially dangerous drug combinations (could extend to any simulataneous number of drugs)
drugcomb2 = np.delete(drugcomb2, np.intersect1d(np.unique(np.where(drugcomb2 == "ACE")[0]), np.unique(np.where(drugcomb2 == "ARB")[0])), axis=0)
drugcomb3 = np.delete(drugcomb3, np.intersect1d(np.unique(np.where(drugcomb3 == "ACE")[0]), np.unique(np.where(drugcomb3 == "ARB")[0])), axis=0)
drugcomb4 = np.delete(drugcomb4, np.intersect1d(np.unique(np.where(drugcomb4 == "ACE")[0]), np.unique(np.where(drugcomb4 == "ARB")[0])), axis=0)
drugcomb5 = np.delete(drugcomb5, np.intersect1d(np.unique(np.where(drugcomb5 == "ACE")[0]), np.unique(np.where(drugcomb5 == "ARB")[0])), axis=0)

##Combining all treatment choices in a list
drugs.insert(0, "NT") # incorporating no treatment
alldrugs = drugs + list(drugcomb2) + list(drugcomb3) + list(drugcomb4) + list(drugcomb5) # number of treatments to consider (196 trts: no treatment plus 1 to 5 drugs from 5 different types at standard dosage - excluding combinations of ACE and ARB)

# Immediate rewards - QALY parameters (Kohli-Lynch et al. 2019)
QoL = {"40-44": [1, 0.9348, 0.8835, 0.9348*0.8835, 0.8970*(1/12)+0.9348*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
       "45-54": [1, 0.9374, 0.8835, 0.9374*0.8835, 0.8862*(1/12)+0.9374*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
       "55-64": [1, 0.9376, 0.8835, 0.9376*0.8835, 0.8669*(1/12)+0.9376*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
       "65-74": [1, 0.9372, 0.8835, 0.9372*0.8835, 0.8351*(1/12)+0.9372*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
       "75-84": [1, 0.9364, 0.8835, 0.9363*0.8835, 0.7946*(1/12)+0.9363*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
       }

# Disutility parameters
disut = 0.002  # base treatment disutility per drug at standard drug
drugs_perdisut = [i / 100 for i in [9.9, 7.5, 3.9, 0, 8.3]]  # percentage of people showing side effects per drug (same order as drugs list)
drugs_disut = [disut * (1 + i) for i in drugs_perdisut]  # treatment disutility per drug type at standard drug

##Generating lists of disutility per drug combination
trtharm = []
for d in range(len(alldrugs)):

    # Making sure evaluated treatment is in a list or string format
    if type(alldrugs[d]) == str or type(alldrugs[d]) == list:
        drugcomb = alldrugs[d]
    else:
        drugcomb = list(alldrugs[d])

    # Counting number of times a drug is being given
    th = drugcomb.count('TH')
    bb = drugcomb.count('BB')
    ace = drugcomb.count('ACE')
    a2ra = drugcomb.count('ARB')
    ccb = drugcomb.count('CCB')

    # Calculating treatment harm per drug combination
    trtharm.append(th*drugs_disut[0] + bb*drugs_disut[1] + ace*drugs_disut[2] + a2ra*drugs_disut[3] + ccb*drugs_disut[4])

# MDP parameters
##Discounting factor
gamma = 0.97

## Terminal reward
### Terminal QoL weights
QoLterm = {"40-44": [1, 0.9348, 0.8835, 0.9348*0.8835, 0.9348, 0.8835, 0, 0, 0, 0],
           "45-54": [1, 0.9374, 0.8835, 0.9374*0.8835, 0.9374, 0.8835, 0, 0, 0, 0],
           "55-64": [1, 0.9376, 0.8835, 0.9376*0.8835, 0.9376, 0.8835, 0, 0, 0, 0],
           "65-74": [1, 0.9372, 0.8835, 0.9372*0.8835, 0.9372, 0.8835, 0, 0, 0, 0],
           "75-84": [1, 0.9364, 0.8835, 0.9363*0.8835, 0.9364, 0.8835, 0, 0, 0, 0]
           }

### Terminal condition standardized mortality rate (first list males, second list females)
mortality_rates = {"Males <2 CHD events":    [1, 1/1.6, 1/2.3, (1/1.6)*(1/2.3), 1/1.6, 1/2.3, 0, 0, 0, 0],
                   "Females <2 CHD events":  [1, 1/2.1, 1/2.3, (1/2.1)*(1/2.3), 1/2.1, 1/2.3, 0, 0, 0, 0],
                   "Males >=2 CHD events":   [1, 1/3.4, 1/2.3, (1/3.4)*(1/2.3), 1/3.4, 1/2.3, 0, 0, 0, 0],
                   "Females >=2 CHD events": [1, 1/2.5, 1/2.3, (1/2.5)*(1/2.3), 1/2.5, 1/2.3, 0, 0, 0, 0]
                   }

# =====================
# Sensitivity analyses
# =====================

# State order scenarios (nonincreasing [base case] and nondecreasing order of rewards) - each order is repeated to match the state classes scenarios
## Initial order: (0) healthy, (1) history of CHD, (2) history of stroke, (3) history of CHD and stroke, (4) surviving a CHD,
# (5) surviving a stroke, (6) dying from non-ASCVD related cause, (7) dying from a CHD, (8) dying from a stroke, and (9) death)
s_order_sens = [[0, 1, 4, 2, 5, 3, 6, 7, 8, 9],
                [0, 1, 4, 2, 5, 3, 6, 7, 8, 9],
                [9, 8, 7, 6, 3, 5, 2, 4, 1, 0],
                [9, 8, 7, 6, 3, 5, 2, 4, 1, 0]]
s_order_sens_len = len(s_order_sens) # number of state order scenarios (including base case)

# State classes scenarios
healthy_sens, alive_sens, chd_hist_sens, stroke_hist_sens, dead_sens, event_ind, s_class_sens = [[] for _ in range(7)]
ascvd_hist_sens = [np.ones([numhealth, events]) for _ in s_order_sens]
for i, so in enumerate(s_order_sens):
    ## Identification of states
    healthy_sens.append(int(np.where(np.array(so) == 0)[0])) # states at which the patient has not experienced ASCVD events
    alive_sens.append(np.where([j in range(6) for j in so])[0]) # states at which the patient is alive
    chd_hist_sens.append(np.where([j in [1, 3, 4] for j in so])[0]) # states where chd risk is higher
    stroke_hist_sens.append(np.where([j in [2, 3, 5] for j in so])[0]) # states where stroke risk is higher
    dead_sens.append(np.where([j in range(6, 10) for j in so])[0]) # states at which the patient is dead_sens

    # Identification of states in where ASCVD events happen
    tmp = np.zeros(numhealth) # vector of zeros
    tmp[np.where([j in [4, 5, 7, 8] for j in so])[0]] = 1 # replacing appropiate indexes with 1
    event_ind.append(tmp.astype(int).tolist()); del tmp # states in where ASCVD events happen

    # Scaling odds to account for history of adverse events
    ascvd_hist_sens[i][chd_hist_sens[i], 0] = ascvd_hist_mult[0]  # Attaching CHD multiplier
    ascvd_hist_sens[i][stroke_hist_sens[i], 1] = ascvd_hist_mult[1]  # Attaching stroke multiplier
    ascvd_hist_sens[i][dead_sens[i], :] = 0  # risk becomes 0 if the patient is not alive

## Nonincreasing cases
s_class_sens.append([[healthy_sens[0]], list(np.delete(chd_hist_sens[0], 2)), list(np.delete(stroke_hist_sens[0], 2)), [stroke_hist_sens[0][2]], list(dead_sens[0])]) # healthy, CHD events, stroke events, both events, death [base case]
s_class_sens.append([[healthy_sens[1]], list(np.unique(np.stack([chd_hist_sens[1], stroke_hist_sens[1]]))), list(dead_sens[1])]) # healthy, ASCVD events, death

## Nondecreasing cases
s_class_sens.append([[healthy_sens[2]], list(np.delete(chd_hist_sens[2], 0)), list(np.delete(stroke_hist_sens[2], 0)), [stroke_hist_sens[2][0]], list(dead_sens[2])][::-1]) # death, both events, stroke events, CHD events, healthy
s_class_sens.append([[healthy_sens[3]], list(np.unique(np.stack([chd_hist_sens[3], stroke_hist_sens[3]]))), list(dead_sens[3])][::-1]) # death, ASCVD events, healthy

# Action order scenarios (obtained from the transition_probabilities.py file retroactively by running a patient with the average BP [SBP = 154, DBP = 97] in Law et al. (2009))
action_order_sbp = [0, 1, 4, 5, 3, 2, 6, 9, 8, 19, 18, 17, 7, 15, 16, 14, 12, 13, 11, 10, 20, 23, 22, 29, 28, 21, 27, 46,
                    49, 48, 47, 25, 26, 43, 44, 24, 45, 41, 42, 40, 37, 38, 39, 35, 36, 34, 32, 33, 31, 30, 50, 53, 52,
                    59, 51, 58, 57, 69, 68, 56, 67, 55, 66, 100, 104, 103, 102, 101, 54, 63, 64, 65, 96, 97, 61, 98, 62,
                    99, 93, 60, 94, 95, 91, 92, 86, 87, 88, 90, 89, 83, 84, 85, 81, 82, 77, 80, 78, 79, 75, 76, 74, 72,
                    73, 71, 105, 108, 107, 114, 106, 113, 112, 70, 124, 111, 123, 110, 122, 121, 139, 138, 109, 120, 137,
                    119, 136, 118, 135, 191, 193, 190, 192, 195, 194, 116, 131, 117, 132, 133, 134, 185, 186, 115, 128,
                    187, 129, 188, 130, 189, 181, 126, 182, 127, 183, 184, 178, 125, 179, 170, 180, 171, 172, 176, 173,
                    166, 177, 174, 167, 175, 168, 163, 169, 164, 156, 165, 161, 157, 162, 158, 153, 160, 159, 154, 151,
                    155, 147, 152, 148, 150, 145, 149, 146, 144, 142, 143, 141, 140] # ordering actions accroding to their SBP reduction (equivalent to ordering by RRR) [base case]

action_order_dbp = [0, 5, 1, 2, 4, 3, 19, 9, 6, 13, 18, 8, 16, 7, 10, 12, 17, 11, 49, 15, 29, 23, 20, 14, 39, 48, 28,
                    22, 45, 26, 21, 33, 38, 47, 27, 104, 36, 69, 30, 44, 25, 59, 32, 37, 53, 46, 50, 24, 31, 42, 89, 35,
                    103, 43, 68, 58, 34, 52, 41, 99, 65, 56, 51, 79, 88, 195, 102, 139, 40, 67, 57, 124, 85, 73, 98, 114,
                    64, 55, 78, 108, 87, 105, 54, 101, 174, 62, 66, 194, 76, 95, 70, 84, 138, 72, 97, 63, 123, 77, 86,
                    113, 189, 100, 134, 71, 120, 107, 82, 111, 61, 159, 106, 75, 94, 173, 83, 193, 96, 60, 137, 74, 122,
                    81, 169, 92, 112, 149, 93, 188, 133, 119, 110, 158, 172, 109, 117, 192, 80, 130, 91, 136, 155, 184,
                    143, 121, 168, 148, 187, 132, 118, 157, 171, 90, 116, 191, 146, 165, 129, 140, 135, 154, 183, 142,
                    115, 167, 147, 186, 131, 156, 127, 141, 170, 152, 190, 145, 164, 128, 180, 153, 182, 166, 185, 144,
                    126, 151, 162, 163, 179, 181, 125, 150, 161, 178, 177, 160, 176, 175] # ordering actions accroding to their DBP reduction

a_order_sens = [action_order_sbp, action_order_sbp, action_order_dbp]
a_order_sens_len = len(a_order_sens) - 1 # number of action order scenarios (excluding base case)

# Action classes scenarios (obtained from the transition_probabilities.py file retroactively by running a patient with the average BP [SBP = 154, DBP = 97] in Law et al. (2009))
action_class_meds = [[0], list(np.array(a_order_sens[0])[range(1, len(drugs))]),
                     list(np.array(a_order_sens[0])[range(len(drugs), len(list(drugcomb2))+len(drugs))]),
                     list(np.array(a_order_sens[0])[range(len(list(drugcomb2))+len(drugs), len(list(drugcomb3))+len(list(drugcomb2))+len(drugs))]),
                     list(np.array(a_order_sens[0])[range(len(list(drugcomb3))+len(list(drugcomb2))+len(drugs),
                                                          len(list(drugcomb4))+len(list(drugcomb3))+len(list(drugcomb2))+len(drugs))]),
                     list(np.array(a_order_sens[0])[range(len(list(drugcomb4))+len(list(drugcomb3))+len(list(drugcomb2))+len(drugs),
                                                          len(list(drugcomb5))+len(list(drugcomb4))+len(list(drugcomb3))+len(list(drugcomb2))+len(drugs))])] # creating classes according to the number of medications [base case]
action_class_meds = [sorted(x) for x in action_class_meds] # sorting lists of actions
action_class_sbp = [[0], [1, 3, 4, 5], [2], [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                    [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53, 57, 58, 59, 68, 69],
                    [54, 55, 56, 60, 61, 62, 63, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                     85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105],
                    [70, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
                     125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 145, 147, 148, 150, 151,
                     152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                     172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
                     192, 193, 194, 195],
                    [140, 141, 142, 143, 144, 146, 149]] # creating classes based on "reasonable" splits on the effect of medications in patients' SBP (used increments of 5 mm Hg)
action_class_dbp = [[0], [1, 2, 4, 5], [3, 6, 9, 19], [7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 49],
                    [14, 20, 21, 22, 23, 26, 27, 28, 29, 33, 38, 39, 45, 47, 48, 104],
                    [24, 25, 30, 31, 32, 34, 35, 36, 37, 40, 41, 42, 43, 44, 46, 50, 51, 52, 53, 56, 58, 59, 65, 67, 68,
                     69, 79, 88, 89, 99, 102, 103, 139, 195],
                    [54, 55, 57, 60, 61, 62, 63, 64, 66, 70, 71, 72, 73, 74, 75, 76, 77, 78, 82, 83, 84, 85, 86, 87, 94,
                     95, 96, 97, 98, 100, 101, 105, 106, 107, 108, 111, 113, 114, 120, 122, 123, 124, 134, 137, 138, 159,
                     173, 174, 189, 193, 194],
                    [80, 81, 90, 91, 92, 93, 109, 110, 112, 115, 116, 117, 118, 119, 121, 127,
                     128, 129, 130, 131, 132, 133, 135, 136, 140, 141, 142, 143, 145, 146, 147,
                     148, 149, 152, 153, 154, 155, 156, 157, 158, 164, 165, 167, 168, 169, 170,
                     171, 172, 180, 182, 183, 184, 186, 187, 188, 190, 191, 192],
                    [125, 126, 144, 150, 151, 160, 161, 162, 163, 166, 175, 176, 177, 178, 179, 181, 185]] # creating classes based on "reasonable" splits on the effect of medications in patients' DBP (used increments of 3 mm Hg)
a_class_sens = [action_class_meds, action_class_sbp, action_class_dbp]

# State distribution scenarios
mult = [0.01/(numhealth*(years+1)), 1/(numhealth*(years+1))] # multipliers
alpha_sens = [np.zeros((numhealth, (years+1)))] # matrix of zeros for sensitivity analysis added after 1st revision
alpha_sens[0][healthy_sens[0], 0] = 1 # all weight focused on year 1 and healthy condition - weight according to NHANES [base case after revision]
alpha_sens.append(np.ones((numhealth, (years+1)))*mult[0]) # matrix of ones for scenario with most weight in first year and healthy condition
alpha_sens[1][healthy_sens[0], 0] += (1-mult[0]) # most weight focused on year 1 and healthy condition - weight almost according to NHANES [former base case]
alpha_sens.append(np.ones((numhealth, (years+1)))*mult[0]) # matrix of ones for scenarios with most weight in first year
alpha_sens[2][:, 0] += (1-mult[0])/numhealth # most of the weight focused on year 1
alpha_sens.append(np.ones((numhealth, (years+1)))*mult[1]) # uniform weight across all health conditions and years
alpha_sens = [x/np.sum(x) for x in alpha_sens] # making sure weights add up to one
alpha_sens_len = len(alpha_sens) - 1 # number of state distribution scenarios (excluding base case)

# Sensitivity scenario summary
sens_sc = s_order_sens_len + a_order_sens_len + alpha_sens_len
sens_id = ['base case', 'combined state classes nonincreasing s', 'segregated nondecreasing s', 'combined state classes nondecreasing s',
           'SBP reduction action aclasses', 'DBP reduction action aclasses',
           'most weight in year 1 and healthy condition', 'most weight at year 1', 'uniform weight distribution']

# ==================
# Patient simulation
# ==================

os.chdir(home_dir)
keys = ('pt_id', 'V_notrt', 'e_notrt',
        'V_opt', 'd_opt', 'occup', 'J_opt', 'e_opt',
        'V_mopt_epochs', 'd_mopt_epochs', 'J_mopt_epochs', 'e_mopt_epochs',
        'V_class_mopt_epochs', 'd_class_mopt_epochs', 'J_class_mopt_epochs', 'e_class_mopt_epochs',
        'V_aha', 'd_aha', 'J_aha', 'e_aha'
        ) # keys for dictionary to store results

pt_ids = range(len(ptdata.id.unique()))  # patient ids # range(len(ptdata.id.unique()))

if __name__ == '__main__': # Ensuring the code is run as the main module

    for sc in range(0, sens_sc):  # base case and all sensitivity scenarios

        print("Running sensitivity analysis scenario", sc) # keeping track of progress

        # Extracting parameters (index 0 is the base case)
        if sc < s_order_sens_len: # state order and state classes sensitivity scenarios

            # State ordering
            state_order = s_order_sens[sc]

            # State classes
            S_class = s_class_sens[sc]

            # Identification of states
            healthy = healthy_sens[sc] # states at which the patient has not experienced ASCVD events
            alive = alive_sens[sc] # states at which the patient is alive
            dead = dead_sens[sc] # states at which the patient is dead
            event_states = event_ind[sc] # indicators of states where patients are experiencing ASCVD events

            # Base case parameters
            chd_hist = chd_hist_sens[0]  # states where chd risk is higher (order not changed for risk calculations to avoid numerical errors in transition probabilities)
            stroke_hist = stroke_hist_sens[0]  # states where stroke risk is higher (order not changed for risk calculations to avoid numerical errors in transition probabilities)
            ascvd_hist = ascvd_hist_sens[0]  # odds scaling to account for history of adverse events (order not changed for risk calculations to avoid numerical errors in transition probabilities)
            action_order = a_order_sens[0] # action ordering
            A_class = a_class_sens[0] # action classes
            alpha = alpha_sens[0] # state distribution

        elif s_order_sens_len <= sc < s_order_sens_len + a_order_sens_len: # action order and action classes sensitivity scenarios

            # Action ordering
            action_order = a_order_sens[sc-s_order_sens_len+1] # action ordering

            # Action classes
            A_class = a_class_sens[sc-s_order_sens_len+1]

            # Base case parameters
            state_order = s_order_sens[0] # state ordering
            S_class = s_class_sens[0]  # state classes
            healthy = healthy_sens[0]  # states at which the patient has not experienced ASCVD events
            alive = alive_sens[0]  # states at which the patient is alive
            chd_hist = chd_hist_sens[0]  # states where chd risk is higher
            stroke_hist = stroke_hist_sens[0]  # states where stroke risk is higher
            ascvd_hist = ascvd_hist_sens[0] # odds scaling to account for history of adverse events
            dead = dead_sens[0]  # states at which the patient is dead
            event_states = event_ind[0]  # indicators of states where patients are experiencing ASCVD events
            alpha = alpha_sens[0]  # state distribution

        elif s_order_sens_len + a_order_sens_len <= sc < s_order_sens_len + a_order_sens_len + alpha_sens_len: # state distribution sensitivity scenarios

            # State distribution
            alpha = alpha_sens[sc-s_order_sens_len-a_order_sens_len+1]

            # Base case parameters
            state_order = s_order_sens[0]  # state ordering
            S_class = s_class_sens[0]  # state classes
            healthy = healthy_sens[0]  # states at which the patient has not experienced ASCVD events
            alive = alive_sens[0]  # states at which the patient is alive
            chd_hist = chd_hist_sens[0]  # states where chd risk is higher
            stroke_hist = stroke_hist_sens[0]  # states where stroke risk is higher
            ascvd_hist = ascvd_hist_sens[0]  # odds scaling to account for history of adverse events
            dead = dead_sens[0]  # states at which the patient is dead
            event_states = event_ind[0]  # indicators of states where patients are experiencing ASCVD events
            action_order = a_order_sens[0]  # action ordering
            A_class = a_class_sens[0] # action classes

        else: # using base for any scenario not accounted for
            print("Scenario", sc, "was not taken into account")

            # Base case parameters
            state_order = s_order_sens[0]  # state ordering
            S_class = s_class_sens[0]  # state classes
            healthy = healthy_sens[0]  # states at which the patient has not experienced ASCVD events
            alive = alive_sens[0]  # states at which the patient is alive
            chd_hist = chd_hist_sens[0]  # states where chd risk is higher
            stroke_hist = stroke_hist_sens[0]  # states where stroke risk is higher
            ascvd_hist = ascvd_hist_sens[0]  # odds scaling to account for history of adverse events
            dead = dead_sens[0]  # states at which the patient is dead
            event_states = event_ind[0] # indicators of states where patients are experiencing ASCVD events
            action_order = a_order_sens[0]  # action ordering
            A_class = a_class_sens[0] # action classes
            alpha = alpha_sens[0] # state distribution

        # # Running patient simulation sequentially
        # start_time_par = tm.time()
        # results = list(itertools.starmap(pt_sim.patient_sim, [(i, ptdata[ptdata.id == i],
        #                                                           numhealth, healthy, dead, years, events,
        #                                                           stroke_hist, ascvd_hist, event_states, lifedata, mortality_rates,
        #                                                           chddeathdata, strokedeathdata, alldeathdata, riskslopedata,
        #                                                           sbpmin, dbpmin, sbpmax, dbpmax, alldrugs, trtharm,
        #                                                           QoL, QoLterm, alpha, gamma,
        #                                                           state_order, S_class, action_order, A_class, action_class_meds,
        #                                                           targetrisk, targetsbp, targetdbp, numeds)
        #                                                           for i in pt_ids]))
        # end_time_par = tm.time()
        # ptresults = results
        # print("--- %s minutes ---" % ((end_time_par-start_time_par)/60))

        # # Running patient simulation in parallel
        start_time_par = tm.time()
        with mp.Pool(cores) as pool: # Creating pool of parallel workers
            par_results = pool.starmap_async(pt_sim.patient_sim, [(i, ptdata[ptdata.id == i],
                                             numhealth, healthy, dead, years, events,
                                             stroke_hist, ascvd_hist, event_states, lifedata, mortality_rates,
                                             chddeathdata, strokedeathdata, alldeathdata, riskslopedata,
                                             sbpmin, dbpmin, sbpmax, dbpmax, alldrugs, trtharm,
                                             QoL, QoLterm, alpha, gamma,
                                             state_order, S_class, action_order, A_class, action_class_meds,
                                             targetrisk, targetsbp, targetdbp, numeds)
                                             for i in pt_ids]).get()
        end_time_par = tm.time()
        print("--- %s minutes ---" % ((end_time_par - start_time_par) / 60))

        # Storing results in a dictionary
        values = ([d[res] for d in par_results] for res in range(len(par_results[0])))
        ptresults = dict(zip(keys, values))

        # Removing dead states (kept for debugging purposes)
        keys_to_extract = ['V_notrt',
                           'V_opt', 'd_opt', 'e_opt',
                           'V_class_mopt_epochs', 'd_class_mopt_epochs', 'e_class_mopt_epochs',
                           'V_mopt_epochs', 'd_mopt_epochs', 'e_mopt_epochs',
                           'V_aha', 'd_aha', 'e_aha'
                           ]
        tmp_dict = {k: [np.delete(x, dead, axis=0) for x in ptresults[k]] for k in keys_to_extract}
        ptresults.update(tmp_dict); del tmp_dict

        # Removing temporary variables (to make sure they are not recycled in subsequent scenarios)
        del state_order, S_class, healthy, alive, chd_hist, stroke_hist, dead, ascvd_hist, event_states, action_order, A_class, alpha

        # Saving results (sensitivity analysis scenario 0 is the base case)
        os.chdir(home_dir)
        if not os.path.isdir(results_dir):
            os.mkdir("Results")
        if not os.path.isdir(sens_dir):
            os.mkdir("Sensitivity Analyses")
        os.chdir(sens_dir)
        with open('Sensitivity analysis results for scenario ' + str(sc) + ' using ' + str(len(pt_ids)) + ' patients with 1 hour time limit and 0.001 absolute MIP gap.pkl', 'wb') as f:
            pk.dump(ptresults, f, protocol=3)
