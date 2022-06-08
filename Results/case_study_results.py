# =======================================================
# Summary of Results - Hypertension treatment case study
# =======================================================

# Loading modules
import os  # directory changes
import numpy as np  # array operations
from functools import reduce # union of multiple arrays
import pandas as pd  # data frame operations
import pickle as pk  # saving results
import gc # clearing memory
import statsmodels.api as sm
from case_study_plots import plot_trt_dist, freq_drug, qalys_events, price_interpret, price_interpret_dist, plot_policies_state # plot functions
from ascvd_risk import arisk  # risk calculations

# Importing parameters from main module
from hypertension_treatment_monotone_mdp import results_dir, fig_dir, sens_dir, ptdata, alive_sens, \
    ascvd_hist_mult, alldrugs, a_order_sens, a_class_sens
alive = alive_sens[0]; A_class = a_class_sens[0]; action_order = a_order_sens[0]

# Presenting results using per capita rate
per_capita = 100000 # results per per_capita number of patients

# Combining strings representing drugs of same type
meds = [0, 1, 2, 3, 4, 5] # number of medications considered in analysis
## Merging list of strings for every treatment choice
drugs_concat = [' + '.join(alldrugs[6:][x]) for x in range(len(alldrugs[6:]))] # merging with '+' sign
drugs_concat = alldrugs[:6] + drugs_concat # incoporating single drug treatments

## Creating tuple of strings to replace (first list are the strings to be replaced and second list are the replacement strings)
temp = tuple(zip(['ACE + ACE', 'ARB + ARB', 'BB + BB', 'CCB + CCB', 'TH + TH',
                  '2 ACE + ACE', '2 ARB + ARB', '2 BB + BB', '2 CCB + CCB', '2 TH + TH',
                  '2 ACE + 2 ACE', '2 ARB + 2 ARB', '2 BB + 2 BB', '2 CCB + 2 CCB', '2 TH + 2 TH',
                  '2 ACE + 3 ACE', '2 ARB + 3 ARB', '2 BB + 3 BB', '2 CCB + 3 CCB', '2 TH + 3 TH'],
                 ['2 ACE', '2 ARB', '2 BB', '2 CCB', '2 TH', '3 ACE', '3 ARB', '3 BB', '3 CCB', '3 TH', 
                  '4 ACE', '4 ARB', '4 BB', '4 CCB', '4 TH', '5 ACE', '5 ARB', '5 BB', '5 CCB', '5 TH']))

## Replacing strings
for j in range(len(temp)): # looping through tuple of strings
    drugs_concat = [x.replace(temp[j][0], temp[j][1]) for x in drugs_concat] # list comprehension of elements in list of drug combinations
del temp

# Creating 'jitter' to distinguishing treatment types with the same number of medications
tmp5 = np.linspace(start=2, stop=8, num=len(A_class[5])) # sequence to represent all the medication combinations with five drugs # start=4.8, stop=5.2 for all medications in plots # start=3, stop=7 for 1 medication only in plots
jitt = np.unique(np.diff(tmp5).round(10))[0] # difference among each treatment option
tmp4 = 4 + np.arange(-len(A_class[4])/2, len(A_class[4])/2)*jitt # sequence to represent all the medication combinations with four drugs
tmp3 = 3 + np.arange(-len(A_class[3])/2, len(A_class[3])/2)*jitt # sequence to represent all the medication combinations with three drugs
tmp2 = 2 + np.arange(-len(A_class[2])/2, len(A_class[2])/2)*jitt # sequence to represent all the medication combinations with two drugs
tmp1 = 1 + np.arange(-len(A_class[1])/2, len(A_class[1])/2)*jitt # sequence to represent each drug
meds_jitt = np.concatenate([np.array([0]), tmp1, tmp2, tmp3, tmp4, tmp5]) # merging sequences
del tmp5, tmp4, tmp3, tmp2, tmp1

# Extracting first year data
ptdata1 = ptdata.groupby('id').first().reset_index(drop=False, inplace=False)
del ptdata; gc.collect()

## Estimating mean arterial pressure
ptdata1['map'] = ptdata1.apply(lambda row: (1/3)*(row['sbp']+2*row['dbp']), axis=1)

## Identifying BP categories
bp_cats = [(ptdata1.sbp < 120) & (ptdata1.dbp < 80),
            (ptdata1.sbp >= 120) & (ptdata1.dbp < 80) & (ptdata1.sbp < 130),
            ((ptdata1.sbp >= 130) | (ptdata1.dbp >= 80)) & ((ptdata1.sbp < 140) & (ptdata1.dbp < 90)),
            (ptdata1.sbp >= 140) | (ptdata1.dbp >= 90)]
bp_cat_labels = ['Normal BP', 'Elevated BP', 'Stage 1 Hypertension', 'Stage 2 Hypertension'] # Hypertension
ptdata1['bp_cat'] = np.select(bp_cats, bp_cat_labels)

## Calculating 1-year risks
### CHD
ptdata1['risk_chd'] = ptdata1.apply(lambda row: arisk(0, row['sex'], row['race'], row['age'], row['sbp'], row['smk'],
                                                      row['tc'], row['hdl'], row['diab'], 0, 1), axis=1)

### Stroke
ptdata1['risk_stroke'] = ptdata1.apply(lambda row: arisk(1, row['sex'], row['race'], row['age'], row['sbp'], row['smk'],
                                                         row['tc'], row['hdl'], row['diab'], 0, 1), axis=1)

### Overall
ptdata1['risk'] = ptdata1.apply(lambda row: row['risk_chd']+row['risk_stroke'], axis=1)

### Calculating odds
ptdata1['odds_chd'] = ptdata1.apply(lambda row: row['risk_chd']/(1-row['risk_chd']), axis=1)
ptdata1['odds_stroke'] = ptdata1.apply(lambda row: row['risk_stroke']/(1-row['risk_stroke']), axis=1)

### Adjusting odds of each ASCVD event for history of CHD and stroke and converting back to risk
ptdata1['adj_risk_chd'] = ptdata1.apply(lambda row: (row['odds_chd']*ascvd_hist_mult[0])/(1+(row['odds_chd']*ascvd_hist_mult[0])), axis=1) # CHD
ptdata1['adj_risk_stroke'] = ptdata1.apply(lambda row: (row['odds_stroke']*ascvd_hist_mult[1])/(1+(row['odds_stroke']*ascvd_hist_mult[1])) if row['age']<60
                                                         else (row['odds_stroke']*2)/(1+(row['odds_stroke']*2)), axis=1) # stroke (see patient_simulation.py for explanation)

### Calculating risk for people with history of CHD, stroke or both
ptdata1['risk_hist_chd'] = ptdata1.apply(lambda row: row['adj_risk_chd']+row['risk_stroke'], axis=1)
ptdata1['risk_hist_stroke'] = ptdata1.apply(lambda row: row['risk_chd']+row['adj_risk_stroke'], axis=1)
ptdata1['risk_hist_ascvd'] = ptdata1.apply(lambda row: row['adj_risk_chd']+row['adj_risk_stroke'], axis=1)

### Removing unnecesary columns
ptdata1 = ptdata1.drop(['risk_chd', 'risk_stroke', 'odds_chd', 'odds_stroke', 'adj_risk_chd', 'adj_risk_stroke'], axis=1)

## Calculating 10-year risks
### CHD
ptdata1['risk_chd10'] = ptdata1.apply(lambda row: arisk(0, row['sex'], row['race'], row['age'], row['sbp'], row['smk'],
                                                        row['tc'], row['hdl'], row['diab'], 0, 10), axis=1)

### Stroke
ptdata1['risk_stroke10'] = ptdata1.apply(lambda row: arisk(1, row['sex'], row['race'], row['age'], row['sbp'], row['smk'],
                                                           row['tc'], row['hdl'], row['diab'], 0, 10), axis=1)

### Overall
ptdata1['risk10'] = ptdata1.apply(lambda row: row['risk_chd10']+row['risk_stroke10'], axis=1)

### Calculating odds
ptdata1['odds_chd'] = ptdata1.apply(lambda row: row['risk_chd10']/(1-row['risk_chd10']), axis=1)
ptdata1['odds_stroke'] = ptdata1.apply(lambda row: row['risk_stroke10']/(1-row['risk_stroke10']), axis=1)

### Adjusting odds of each ASCVD event for history of CHD and stroke and converting back to risk
ptdata1['adj_risk_chd'] = ptdata1.apply(lambda row: (row['odds_chd']*ascvd_hist_mult[0])/(1+(row['odds_chd']*ascvd_hist_mult[0])), axis=1) # CHD
ptdata1['adj_risk_stroke'] = ptdata1.apply(lambda row: (row['odds_stroke']*ascvd_hist_mult[1])/(1+(row['odds_stroke']*ascvd_hist_mult[1])) if row['age']<60
                                                         else (row['odds_stroke']*2)/(1+(row['odds_stroke']*2)), axis=1) # stroke (see patient_simulation.py for explanation)

### Calculating risk for people with history of CHD, stroke or both
ptdata1['risk10_hist_chd'] = ptdata1.apply(lambda row: row['adj_risk_chd']+row['risk_stroke10'], axis=1)
ptdata1['risk10_hist_stroke'] = ptdata1.apply(lambda row: row['risk_chd10']+row['adj_risk_stroke'], axis=1)
ptdata1['risk10_hist_ascvd'] = ptdata1.apply(lambda row: row['adj_risk_chd']+row['adj_risk_stroke'], axis=1)

### Removing unnecesary columns
ptdata1 = ptdata1.drop(['risk_chd10', 'risk_stroke10', 'odds_chd', 'odds_stroke', 'adj_risk_chd', 'adj_risk_stroke'], axis=1)

# ---------------------------
# Population-level analysis
# ---------------------------

# Overall demographic information
## Race and sex
pd.concat([(ptdata1[['wt', 'race', 'sex']].groupby(['race', 'sex']).sum()/1e06).round(2),
           (ptdata1[['wt', 'race', 'sex']].groupby(['race', 'sex']).sum()/ptdata1.wt.sum()*100).round(2)],
          axis=1)

## BP category
pd.concat([(ptdata1[['wt', 'bp_cat']].groupby(['bp_cat']).sum()/1e06).round(2),
           (ptdata1[['wt', 'bp_cat']].groupby(['bp_cat']).sum()/ptdata1.wt.sum()*100).round(2)],
          axis=1)

## Race and BP category
(ptdata1[['wt', 'race', 'bp_cat']].groupby(['race', 'bp_cat']).sum()/1e06).reset_index(drop=False, inplace=False).round(2)

## Sex and BP category
(ptdata1[['wt', 'sex', 'bp_cat']].groupby(['sex', 'bp_cat']).sum()/1e06).reset_index(drop=False, inplace=False).round(2)

# Loading MP and CMP results (scenario 0 is the base case)
os.chdir(sens_dir)
with open('Sensitivity analysis results for scenario 0 using 4590 patients with 1 hour time limit and 0.001 absolute MIP gap.pkl',
          'rb') as f:
    pt_sim = pk.load(f)

# Excluding NaN values (patients that the MIP was not able to find a solution in 1 hour)
# Indexes of patients with incomplete policies
nan_index = reduce(np.union1d, (np.where([np.isnan(x).any() for x in pt_sim['V_class_mopt_epochs']])[0],
                                np.where([np.isnan(x).any() for x in pt_sim['V_mopt_epochs']])[0]))

## Indexes of patients with MIP solutions in all policies
not_nan_index = np.delete(np.arange(len(pt_sim['pt_id'])), nan_index)

## Keeping only patients with all policies
pt_sim = {k: [v[i] for i in not_nan_index] for k, v in pt_sim.items()}

## Making sure that equal polcicies do not result in different total discounted reward (to avoid numerical issues)
tmp_mopt, tmp_class_mopt = [[] for _ in range(2)]
for y, x in enumerate(pt_sim['d_opt']):

    ## Making sure the total discounted reward of CMP is at least the total discounted reward of MP
    if np.all(pt_sim['d_class_mopt_epochs'] == pt_sim['d_mopt_epochs'][y]):
        tmp_mopt.append(pt_sim['J_class_mopt_epochs'][y])
        tmp_class_mopt.append(pt_sim['J_class_mopt_epochs'][y])
    else:
        tmp_mopt.append(pt_sim['J_mopt_epochs'][y])
        tmp_class_mopt.append(pt_sim['J_class_mopt_epochs'][y])

    ## Making sure total discounted reward of OP is at least the total discounted reward of CMP
    if np.all(x == pt_sim['d_class_mopt_epochs'][y]):
        tmp_class_mopt[y] = pt_sim['J_opt'][y]
    else:
        tmp_class_mopt[y] = pt_sim['J_class_mopt_epochs'][y]

    ## Making sure total discounted reward of OP is at least the total discounted reward of MP
    if np.all(x == pt_sim['d_mopt_epochs'][y]):
        tmp_mopt[y] = pt_sim['J_opt'][y]
    else:
        tmp_mopt[y] = pt_sim['J_mopt_epochs'][y]
tmp_dict = {'J_class_mopt_epochs': tmp_class_mopt, 'J_mopt_epochs': tmp_mopt}
pt_sim.update(tmp_dict); del tmp_dict, tmp_class_mopt, tmp_mopt

## Making sure that the total expected reward is OP >= CMP >= MP (to avoid numerical issues)
tmp_class_mopt, tmp_mopt = [[] for _ in range(2)]
for y, x in enumerate(pt_sim['J_opt']):

    ## Making sure total discounted reward of CMP is at most the total discounted reward of OP
    if x < pt_sim['J_class_mopt_epochs'][y]:
        tmp_class_mopt.append(x)
    else:
        tmp_class_mopt.append(pt_sim['J_class_mopt_epochs'][y])

    ## Making sure the total discounted reward of MP is at most the total discounted reward of CMP
    if tmp_class_mopt[y] < pt_sim['J_mopt_epochs'][y]:
        tmp_mopt.append(tmp_class_mopt[y])
    else:
        tmp_mopt.append(pt_sim['J_mopt_epochs'][y])
tmp_dict = {'J_mopt_epochs': tmp_mopt, 'J_class_mopt_epochs': tmp_class_mopt}
pt_sim.update(tmp_dict); del tmp_dict, tmp_mopt, tmp_class_mopt

# Evaluating number and type of medications recommended by each policy
polnames = ['Optimal', 'Class-Ordered Monotone', 'Monotone', 'Clinical Guidelines'] # names of policies
sub_ptdata1 = ptdata1.loc[ptdata1.id.isin(pt_sim['pt_id']), ['id', 'wt', 'bp_cat']].reset_index(drop=True) # selecting ids, weights, and BP groups of patients in original data that have all policies
trt_df = pd.DataFrame() # data frame to store matching percentages treatment results
mult = 1000 # multiplier for number of patients (sampling weigths are divided by mult)
for year in [0, 9]:
    ## Data frame of number of medications and drug combination
    tmp = [np.select([[z in x for z in [y[0, year] for y in pt_sim[pol]]] for x in A_class], meds)
            for pol in ['d_opt', 'd_class_mopt_epochs', 'd_mopt_epochs', 'd_aha']] # converting actions to number of medications
    tmp1 = pd.DataFrame(tmp, index=polnames).T
    tmp2 = pd.concat([sub_ptdata1, tmp1], axis=1)
    tmp3 = tmp2.melt(id_vars=['id', 'wt', 'bp_cat'], var_name='policy', value_name='meds')

    ## Data frame of drug types
    tmp4 = [[np.select([y[0, year] == x for x in range(len(drugs_concat))], drugs_concat) for y in pt_sim[x]]
            for x in ['d_opt', 'd_class_mopt_epochs', 'd_mopt_epochs', 'd_aha']] # converting actions to treatments
    tmp5 = pd.DataFrame(tmp4, index=polnames).T.applymap(str)
    tmp6 = pd.concat([sub_ptdata1, tmp5], axis=1)
    tmp7 = tmp6.melt(id_vars=['id', 'wt', 'bp_cat'], var_name='policy', value_name='drugs')

    ## Combining data frames
    tmp8 = pd.concat([tmp3, tmp7['drugs']], axis=1)
    del tmp, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7

    ## Expanding data according to (reduced) sampling weights
    tmp9 = sm.stats.DescrStatsW(tmp8, weights=tmp8.wt/mult) # weighted data
    tmp_trt = pd.DataFrame(tmp9.asrepeats()); del tmp9

    ## Modifying dataset
    tmp_trt.columns = tmp8.columns
    tmp_trt['year'] = year+1
    tmp_trt = tmp_trt.drop(['wt'], axis=1)

    ## Combining years of data
    trt_df = pd.concat([trt_df, tmp_trt], axis=0)
    del tmp8, tmp_trt

## Change data type of number of medications to numeric
trt_df['meds'] = trt_df['meds'].apply(pd.to_numeric)

## Sorting data frame
trt_df['bp_cat'] = trt_df['bp_cat'].astype('category') # converting BP categories to category
trt_df['bp_cat'] = trt_df.bp_cat.cat.set_categories(bp_cat_labels) # adding sorted categories
trt_df['policy'] = trt_df['policy'].astype('category') # converting policy to category
trt_df['policy'] = trt_df.policy.cat.set_categories(polnames) # adding sorted categories
trt_df = trt_df.sort_values(['bp_cat', 'policy']) # sorting dataframe based on selected columns

## Plotting number of medications by policy and BP category per year
os.chdir(fig_dir)
plot_trt_dist(trt_df)

## Tables of drug combinations
os.chdir(results_dir)
### Overall
count_drug_types = trt_df.groupby(['year', 'policy', 'meds', 'drugs']).size().reset_index()
count_drug_types.to_csv('Counts of Medication Types.csv', index=False)

### By BP category
count_drug_types_bpcat = trt_df.groupby(['year', 'bp_cat', 'policy', 'meds', 'drugs']).size().reset_index()
count_drug_types_bpcat.to_csv('Counts of Medication Types - BP Category.csv', index=False)

## Tables of frequency of drug type
os.chdir(results_dir)
drugs = alldrugs[:6]

### Overall
freq_df = pd.DataFrame() # data frame to store frequencies
for i, d in enumerate(drugs):
    trt_df[d] = trt_df.drugs.str.count(d) # counting occurrence of each medication type in drug combinations
    tmp_freq = trt_df.groupby(['year', 'policy'])[d].sum().reset_index() # adding up counts by categories
    if i==0:
        freq_df = pd.concat([freq_df, tmp_freq], axis=1) # combining drug types into a single data frame
    else:
        freq_df = pd.concat([freq_df, tmp_freq.loc[:, d]], axis=1) # combining drug types into a single data frame (excluding variables already merged)
freq_df = freq_df.melt(id_vars=['year', 'policy'], var_name='drug_type', value_name='freq')  # melting data frame
# freq_df.to_csv('Frequency of Medication Types by Medication Number.csv', index=False)

### By BP category
freq_df_bpcat = pd.DataFrame() # data frame to store frequencies
for i, d in enumerate(drugs):
    trt_df[d] = trt_df.drugs.str.count(d) # counting occurrence of each medication type in drug combinations
    tmp_freq = trt_df.groupby(['year', 'bp_cat', 'policy'])[d].sum().reset_index() # adding up counts by categories
    if i==0:
        freq_df_bpcat = pd.concat([freq_df_bpcat, tmp_freq], axis=1) # combining drug types into a single data frame
    else:
        freq_df_bpcat = pd.concat([freq_df_bpcat, tmp_freq.loc[:, d]], axis=1) # combining drug types into a single data frame (excluding variables already merged)
freq_df_bpcat = freq_df_bpcat.melt(id_vars=['year', 'bp_cat', 'policy'], var_name='drug_type', value_name='freq')  # melting data frame
# freq_df_bpcat.to_csv('Frequency of Medication Types by Medication Number - BP Category.csv', index=False)

## Plotting frequency of drug type per number of medication
os.chdir(fig_dir)
freq_df_bpcat = freq_df_bpcat[freq_df_bpcat.bp_cat!=bp_cat_labels[0]] # removing normal BP category
freq_df_bpcat['bp_cat'] = freq_df_bpcat.bp_cat.cat.set_categories(bp_cat_labels[1:], inplace=True) # adding sorted categories
freq_df_bpcat.freq = freq_df_bpcat.freq/1e03 # converting numbers in per million (easier to understand than thousands)
freq_drug(freq_df_bpcat)

# Evaluating QALYs saved and events averted per policy, compared to no treatment (in per_capita number of patients)
polnames = ['Optimal', 'Class-Ordered Monotone', 'Monotone'] # names of policies

## Data frames of expected QALYs saved at year 0 and healthy state
tmp = [[y[0, 0] - pt_sim['V_aha'][i][0, 0] for i, y in enumerate(pt_sim[x])] for x in ['V_opt', 'V_class_mopt_epochs', 'V_mopt_epochs']] # calculating expected number of events allowed by each suboptimal policy in healthy state at the first year of analysis
tmp1 = pd.DataFrame(tmp, index=polnames).T; del tmp # creating data frame from list of lists
qalys_df = pd.concat([ptdata1.loc[ptdata1.id.isin(pt_sim['pt_id']), ['id', 'wt', 'bp_cat']].reset_index(drop=True), tmp1], axis=1); del tmp1 # creating data frames with ids, weights, and BP categories
qalys_df = qalys_df.melt(id_vars=['id', 'wt', 'bp_cat'], var_name='policy', value_name='qalys') # melting data frame
qalys_df['qalys'] = qalys_df['qalys'].multiply(qalys_df['wt'], axis=0) # adjusting value functions with sampling weights

### Overall
tmp2 = qalys_df.groupby(['policy']).wt.sum() # adding up sampling weights
qalys_df_ovr = qalys_df.merge(tmp2, on=['policy']); del tmp2 # merging dataframes
qalys_df_ovr.rename(columns={'wt_x': 'wt', 'wt_y': 'sum_wt'}, inplace=True) # renaming weight column
qalys_df_ovr['qalys'] = qalys_df_ovr['qalys'].divide(qalys_df_ovr['sum_wt'], axis=0)*per_capita # calculating PI per_capita number of patients
qalys_df_ovr['policy'] = qalys_df_ovr['policy'].astype('category') # converting policy to category
qalys_df_ovr['policy'] = qalys_df_ovr.policy.cat.set_categories(polnames) # adding sorted categories
qalys_df_ovr = qalys_df_ovr.sort_values(['policy']) # sorting dataframe based on selected columns
print(qalys_df_ovr.groupby(['policy']).qalys.sum())

### By BP group
tmp2 = qalys_df.groupby(['bp_cat', 'policy']).wt.sum() # adding up sampling weights
qalys_df_bp = qalys_df.merge(tmp2, on=['bp_cat', 'policy']); del tmp2 # merging dataframes
qalys_df_bp.rename(columns={'wt_x': 'wt', 'wt_y': 'sum_wt'}, inplace=True) # renaming weight column
qalys_df_bp['qalys'] = qalys_df_bp['qalys'].divide(qalys_df_bp['sum_wt'], axis=0)*per_capita # calculating PI per_capita number of patients

#### Sorting and printing data frame
qalys_df_bp['bp_cat'] = qalys_df_bp['bp_cat'].astype('category') # converting BP categories to category
qalys_df_bp['bp_cat'] = qalys_df_bp.bp_cat.cat.set_categories(bp_cat_labels) # adding sorted categories
qalys_df_bp['policy'] = qalys_df_bp['policy'].astype('category') # converting policy to category
qalys_df_bp['policy'] = qalys_df_bp.policy.cat.set_categories(polnames) # adding sorted categories
qalys_df_bp = qalys_df_bp.sort_values(['bp_cat', 'policy']) # sorting dataframe based on selected columns
print(qalys_df_bp.groupby(['policy']).qalys.sum())
print(qalys_df_bp.groupby(['bp_cat', 'policy']).qalys.sum().round(0)) # expected QALYs saved at year 0 and healthy state (results for paper)

##### Plotting results
os.chdir(fig_dir)
qalys_df_bp = qalys_df_bp[qalys_df_bp.bp_cat!=bp_cat_labels[0]] # removing normal BP category
qalys_df_bp['bp_cat'] = qalys_df_bp.bp_cat.cat.set_categories(bp_cat_labels[1:]) # adding sorted categories
qalys_events(qalys_df_bp)

## Data frames of expected events averted at year 0 and healthy state
polnames = ['Optimal', 'Class-Ordered Monotone', 'Monotone'] # names of policies
tmp = [[pt_sim['e_aha'][i][0, 0] - y[0, 0] for i, y in enumerate(pt_sim[x])] for x in ['e_opt', 'e_class_mopt_epochs', 'e_mopt_epochs']] # calculating expected number of events allowed by each suboptimal policy in healthy state at the first year of analysis
tmp1 = pd.DataFrame(tmp, index=polnames).T; del tmp # creating data frame from list of lists
events_df = pd.concat([ptdata1.loc[ptdata1.id.isin(pt_sim['pt_id']), ['id', 'wt', 'bp_cat']].reset_index(drop=True), tmp1], axis=1); del tmp1 # creating data frames with ids, weights, and BP categories
events_df = events_df.melt(id_vars=['id', 'wt', 'bp_cat'], var_name='policy', value_name='events') # melting data frame
events_df['events'] = events_df['events'].multiply(events_df['wt'], axis=0) # adjusting value functions with sampling weights

### Overall
tmp2 = events_df.groupby(['policy']).wt.sum() # adding up sampling weights
events_df_ovr = events_df.merge(tmp2, on=['policy']); del tmp2 # merging dataframes
events_df_ovr.rename(columns={'wt_x': 'wt', 'wt_y': 'sum_wt'}, inplace=True) # renaming weight column
events_df_ovr['events'] = events_df_ovr['events'].divide(events_df_ovr['sum_wt'], axis=0)*per_capita # calculating PI per_capita number of patients
events_df_ovr['policy'] = events_df_ovr['policy'].astype('category') # converting policy to category
events_df_ovr['policy'] = events_df_ovr.policy.cat.set_categories(polnames) # adding sorted categories
events_df_ovr = events_df_ovr.sort_values(['policy']) # sorting dataframe based on selected columns
print(events_df_ovr.groupby(['policy']).events.sum()) # expected events averted at year 0 and healthy state (results for paper)

### By BP group
tmp2 = events_df.groupby(['bp_cat', 'policy']).wt.sum() # adding up sampling weights
events_df_bp = events_df.merge(tmp2, on=['bp_cat', 'policy']); del tmp2 # merging dataframes
events_df_bp.rename(columns={'wt_x': 'wt', 'wt_y': 'sum_wt'}, inplace=True) # renaming weight column
events_df_bp['events'] = events_df_bp['events'].divide(events_df_bp['sum_wt'], axis=0)*per_capita # calculating PI per_capita number of patients

#### Sorting and printing data frame
events_df_bp['bp_cat'] = events_df_bp['bp_cat'].astype('category') # converting BP categories to category
events_df_bp['bp_cat'] = events_df_bp.bp_cat.cat.set_categories(bp_cat_labels) # adding sorted categories
events_df_bp['policy'] = events_df_bp['policy'].astype('category') # converting policy to category
events_df_bp['policy'] = events_df_bp.policy.cat.set_categories(polnames) # adding sorted categories
events_df_bp = events_df_bp.sort_values(['bp_cat', 'policy']) # sorting dataframe based on selected columns
print(events_df_bp.groupby(['bp_cat', 'policy']).events.sum().round(0)) # expected events averted at year 0 and healthy state (results for paper)

##### Plotting results
os.chdir(fig_dir)
events_df_bp = events_df_bp[events_df_bp.bp_cat!=bp_cat_labels[0]] # removing normal BP category
events_df_bp['bp_cat'] = events_df_bp.bp_cat.cat.set_categories(bp_cat_labels[1:]) # adding sorted categories
qalys_events(events_df_bp, events=True)

# Making plots of expected total price of interpretability (per BP group)
## Data frame of expected total price of interpretability
polnames = ['Class-Ordered Monotone', 'Monotone', 'Clinical Guidelines'] # names of policies
tmp = [[pt_sim['J_opt'][i] - y for i, y in enumerate(pt_sim[x])] for x in ['J_class_mopt_epochs', 'J_mopt_epochs', 'J_aha']] # calculating price of interpretability per policy and removing optimal policy value functions
tmp1 = pd.DataFrame(tmp).T; tmp1.columns = polnames # creating data frame from list of lists
pi_df = pd.concat([ptdata1.loc[ptdata1.id.isin(pt_sim['pt_id']), ['id', 'wt', 'bp_cat', 'sbp', 'map', 'risk', 'risk10']].reset_index(drop=True), tmp1], axis=1) # creating data frames with ids, weights, and BP categories
os.chdir(results_dir); pi_df.to_csv('Price of Interpretability per Policy.csv', index=False) # saving data frame as CSV for hypothesis tesing on R
del tmp, tmp1

## Melting data frame and ordering results for plot
pi_df = pi_df.melt(id_vars=['id', 'wt', 'bp_cat', 'sbp', 'map', 'risk', 'risk10'], var_name='policy', value_name='pi')
pi_df['bp_cat'] = pi_df['bp_cat'].astype('category') # converting BP categories to category
pi_df['bp_cat'] = pi_df.bp_cat.cat.set_categories(bp_cat_labels) # adding sorted categories
pi_df['policy'] = pi_df['policy'].astype('category') # converting policy to category
pi_df['policy'] = pi_df.policy.cat.set_categories(polnames) # adding sorted categories
pi_df = pi_df.sort_values(['bp_cat', 'policy']) # sorting dataframe based on selected columns

## Making plots and tables
os.chdir(fig_dir)

### Distribution of PI
pi_df_dist = pi_df[(pi_df.policy!=polnames[2])] # removing clinical guidelines
pi_df_dist['policy'] = pi_df_dist.policy.cat.set_categories(polnames[:-1]) # adding sorted categories (without clinical guidelines)
price_interpret_dist(pi_df_dist)

### Adjusting PI with sampling weights
pi_df['pi'] = pi_df['pi'].multiply(pi_df['wt'], axis=0) # adjusting value functions with sampling weights

### Overall PI per policy
tmp = pi_df.groupby(['policy']).wt.sum() # adding up sampling weights
pi_ovr = pi_df.merge(tmp, on=['policy']); del tmp # merging dataframes
pi_ovr.rename(columns={'wt_x': 'wt', 'wt_y': 'sum_wt'}, inplace=True) # renaming weight column
pi_ovr['pi'] = pi_ovr['pi'].divide(pi_ovr['sum_wt'], axis=0)*per_capita # calculating PI per_capita number of patients
print(pi_ovr.groupby(['policy']).pi.sum()) # table of PI per policy (for paper)

#### PI per capita (by BP category and policy)
tmp = pi_df.groupby(['bp_cat', 'policy']).wt.sum() # adding up sampling weights
pi_df = pi_df.merge(tmp, on=['bp_cat', 'policy']); del tmp # merging dataframes
pi_df.rename(columns={'wt_x': 'wt', 'wt_y': 'sum_wt'}, inplace=True) # renaming weight column
pi_df['pi'] = pi_df['pi'].divide(pi_df['sum_wt'], axis=0)*per_capita # calculating PI per_capita number of patients
print(pi_df.groupby(['bp_cat', 'policy']).pi.sum()) # table of PI per BP category and policy (for paper)

### Removing outliers by BP group (trimming values above 99 percentile) - done to reduce error bars in plot
tmp = pd.DataFrame(pi_df.groupby(['bp_cat', 'policy']).pi.describe(percentiles=[0.99]).round(6)['99%']); tmp.columns = ['trim']
pi_df = pi_df.merge(tmp, left_on=['bp_cat', 'policy'], right_index=True); del tmp
pi_df = pi_df[(pi_df['pi'] <= pi_df['trim'])]

#### Plotting results
pi_df = pi_df[pi_df.bp_cat!=bp_cat_labels[0]] # removing normal BP category
pi_df['bp_cat'] = pi_df.bp_cat.cat.set_categories(bp_cat_labels[1:]) # adding sorted categories
pi_df = pi_df[(pi_df.policy!=polnames[2])] # removing clinical guidelines
pi_df['policy'] = pi_df.policy.cat.set_categories(polnames[:-1]) # adding sorted categories
price_interpret(pi_df) # plotting results

# Making plot of pairwise differences between CMP and MP
## Data frame of expected total price of interpretability
tmp = [[pt_sim['J_class_mopt_epochs'][i] - y for i, y in enumerate(pt_sim[x])] for x in ['J_mopt_epochs']] # calculating price of interpretability per policy and removing optimal policy value functions
tmp1 = pd.DataFrame(tmp).T; tmp1.columns = ['pi'] # creating data frame from list of lists
diff_df = pd.concat([ptdata1.loc[ptdata1.id.isin(pt_sim['pt_id']), ['wt', 'bp_cat', 'sbp', 'map', 'risk', 'risk10']].reset_index(drop=True), tmp1], axis=1) # creating data frames with ids, weights, and BP categories
diff_df['bp_cat'] = diff_df['bp_cat'].astype('category') # converting BP categories to category
diff_df['bp_cat'] = diff_df.bp_cat.cat.set_categories(bp_cat_labels) # adding sorted categories
diff_df = diff_df.sort_values(['bp_cat']) # sorting dataframe based on selected columns
del tmp, tmp1

## Making plot of distribution of pairwise differences
os.chdir(fig_dir)
price_interpret_dist(diff_df, pairwise=True)

# ----------------------
# Sensitivity analyses
# ----------------------

# Notes: Scenario 0 is the base case. The meaning of scenarios 1 and 2 were exchanged in the paper. Scenario 3 was excluded.

# Listing files in sensitivity analysis directory
_, _, filenames = next(os.walk(sens_dir))

# Creating table of price of interpretabilty, events averted, and number of medications
## Loading results and identifying patients with incomplete policies
pt_sens_sim, not_nan_index, pop_count, exc_rec, exc_pop = [[] for i in range(5)] # lists to store combined results
single_drugs = drugs_concat[:6] # list of single drugs
os.chdir(sens_dir)
for sc, sc_name in enumerate(filenames):

    # Loading results
    with open(sc_name, 'rb') as f:
        pt_sens_sim.append(pk.load(f))

    # Excluding NaN values (patients for which the MIP was not able to find a solution in 1 hour)
    ## Indexes of patients with incomplete policies
    if sc == 0: # monotone policies on the states only are solely considered in the base case scenario
        nan_index = reduce(np.union1d, (np.where([np.isnan(x).any() for x in pt_sens_sim[sc]['V_mopt_epochs']])[0],
                                        np.where([np.isnan(x).any() for x in pt_sens_sim[sc]['V_class_mopt_epochs']])[0],
                                        ))
    else:
        nan_index = reduce(np.union1d, (np.where([np.isnan(x).any() for x in pt_sens_sim[sc]['V_mopt_epochs']])[0],
                                        np.where([np.isnan(x).any() for x in pt_sens_sim[sc]['V_class_mopt_epochs']])[0]
                                        ))

    ## Indexes of patients with MIP solutions in all policies
    not_nan_index.append(np.delete(np.arange(len(pt_sens_sim[sc]['pt_id'])), nan_index))#; del nan_index

    ## Counting number of people with policies in each scenario
    ptresults = {k: [v[i] for i in not_nan_index[sc]] for k, v in pt_sens_sim[sc].items()}
    sub_ptdata1 = ptdata1.loc[ptdata1.id.isin(ptresults['pt_id']), ['id', 'wt']].reset_index(
        drop=True)  # selecting ids and weights of patients in original data that have all policies
    pop_count.append(np.round(sub_ptdata1.wt.sum()/1e06, 2)); del ptresults, sub_ptdata1 # total population with all policies in scenario sc
    exc_pop.append(np.round(66.50-pop_count[sc], 2)) # population excluded in scenario sc
    exc_rec.append(len(nan_index)) # records excluded in scenario sc
    # pd.DataFrame({'Records': exc_rec, 'Population': exc_pop}) # data frame of exclusions

# Keeping only patients with all policies
not_nan_index_int = reduce(np.intersect1d, not_nan_index) # intersection of all list of indexes without NAs
pt_sens_sim = [{k: [v[i] for i in not_nan_index_int] for k, v in ptresults.items()} for ptresults in pt_sens_sim] # removing cases with incomplete policies in any scenario
sub_ptdata1 = ptdata1.loc[ptdata1.id.isin(pt_sens_sim[0]['pt_id']), ['id', 'wt']].reset_index(
    drop=True)  # selecting ids and weights of patients in original data that have all policies
tot_pop_count = np.round(sub_ptdata1.wt.sum()/1e06, 2) # counting the total population with all policies accross all scenarios
tot_exc_pop = np.round(66.50-tot_pop_count, 2); del tot_pop_count # calculating the total population exluded from any scenario
tot_exc_rec = ptdata1.shape[0] - len(not_nan_index_int) # calculating the number of records excluded from any scenario

# Calculating the quantities of interests at each scenario
polnames = ['CMP', 'MP', 'CG'] # names of policies # 'Class-Ordered Monotone', 'Monotone', 'Clinical Guidelines'
healthy = [0, 0, 5, 5, 0, 0, 0, 0, 0] # identification of health states accounting for dead states (they were removed after the simulation)
sens_df = pd.DataFrame() # empty dataframe to store results
for rs, ptresults in enumerate(pt_sens_sim):

    ## Making sure that equal polcicies do not result in different total discounted reward (to avoid numerical issues)
    tmp_mopt, tmp_class_mopt = [[] for _ in range(2)]
    for y, x in enumerate(ptresults['d_opt']):

        ## Making sure the total discounted reward of CMP is at least the total discounted reward of MP
        if np.all(ptresults['d_class_mopt_epochs'] == ptresults['d_mopt_epochs'][y]):
            tmp_mopt.append(ptresults['J_class_mopt_epochs'][y])
            tmp_class_mopt.append(ptresults['J_class_mopt_epochs'][y])
        else:
            tmp_mopt.append(ptresults['J_mopt_epochs'][y])
            tmp_class_mopt.append(ptresults['J_class_mopt_epochs'][y])

        ## Making sure total discounted reward of OP is at least the total discounted reward of CMP
        if np.all(x == ptresults['d_class_mopt_epochs'][y]):
            tmp_class_mopt[y] = ptresults['J_opt'][y]
        else:
            tmp_class_mopt[y] = ptresults['J_class_mopt_epochs'][y]

        ## Making sure total discounted reward of OP is at least the total discounted reward of MP
        if np.all(x == ptresults['d_mopt_epochs'][y]):
            tmp_mopt[y] = ptresults['J_opt'][y]
        else:
            tmp_mopt[y] = ptresults['J_mopt_epochs'][y]
    tmp_dict = {'J_class_mopt_epochs': tmp_class_mopt, 'J_mopt_epochs': tmp_mopt}
    ptresults.update(tmp_dict); del tmp_dict, tmp_class_mopt, tmp_mopt

    ## Making sure that the total expected reward is OP >= CMP >= MP (to avoid numerical issues)
    tmp_class_mopt, tmp_mopt = [[] for _ in range(2)]
    for y, x in enumerate(ptresults['J_opt']):

        ## Making sure total discounted reward of CMP is at most the total discounted reward of OP
        if x < ptresults['J_class_mopt_epochs'][y]:
            tmp_class_mopt.append(x)
        else:
            tmp_class_mopt.append(ptresults['J_class_mopt_epochs'][y])

        ## Making sure the total discounted reward of MP is at most the total discounted reward of OP
        if tmp_class_mopt[y] < ptresults['J_mopt_epochs'][y]:
            tmp_mopt.append(tmp_class_mopt[y])
        else:
            tmp_mopt.append(ptresults['J_mopt_epochs'][y])

    tmp_dict = {'J_mopt_epochs': tmp_mopt, 'J_class_mopt_epochs': tmp_class_mopt}
    ptresults.update(tmp_dict); del tmp_dict, tmp_mopt, tmp_class_mopt

    ## Data frame of expected total price of interpretability
    tmp = [[ptresults['J_opt'][i] - y for i, y in enumerate(ptresults[x])] for x in ['J_class_mopt_epochs', 'J_mopt_epochs', 'J_aha']] # calculating price of interpretability per policy and removing optimal policy value functions
    tmp1 = pd.DataFrame(tmp, index=polnames).T # creating data frame from list of lists
    tmp1[polnames] = tmp1[polnames].multiply(ptdata1[ptdata1.id.isin(ptresults['pt_id'])].wt.to_numpy(), axis=0) # adjusting value functions with sampling weights
    tmp_pi = tmp1.sum().to_frame().T # adding up over each policy
    tmp_pi = tmp_pi.divide(ptdata1[ptdata1.id.isin(ptresults['pt_id'])].wt.sum(), axis=0)*per_capita # calculating results per capita
    del tmp, tmp1

    ## Data frame of total ASCVD events allowed
    tmp = [[y[healthy[rs], 0] - ptresults['e_opt'][i][healthy[rs], 0] for i, y in enumerate(ptresults[x])] for x in ['e_class_mopt_epochs', 'e_mopt_epochs', 'e_aha']] # calculating expected number of events allowed by each suboptimal policy in healthy state at the first year of analysis
    tmp1 = pd.DataFrame(tmp, index=polnames).T # creating data frame from list of lists
    tmp1[polnames] = tmp1[polnames].multiply(ptdata1[ptdata1.id.isin(ptresults['pt_id'])].wt.to_numpy(), axis=0) # adjusting value functions with sampling weights
    tmp_e = tmp1.sum().to_frame().T # adding up over each policy
    tmp_e = tmp_e.divide(ptdata1[ptdata1.id.isin(ptresults['pt_id'])].wt.sum(), axis=0)*per_capita  # calculating results per capita
    del tmp, tmp1

    ## Data frames of average number of medications
    ### Initial year
    tmp0 = [np.select([[z in x for z in [y[healthy[rs], 0] for y in ptresults[pol]]] for x in A_class], meds)
            for pol in ['d_opt', 'd_class_mopt_epochs', 'd_mopt_epochs', 'd_aha']] # converting actions to number of medications
    tmp = pd.DataFrame(tmp0, index=['OP']+polnames).T
    tmp_meds0 = (tmp.multiply(sub_ptdata1.wt, axis=0).sum()/sub_ptdata1.wt.sum()*per_capita/1e03).to_frame().T; del tmp

    ### Middle year
    tmp5 = [np.select([[z in x for z in [y[healthy[rs], 5] for y in ptresults[pol]]] for x in A_class], meds)
            for pol in ['d_opt', 'd_class_mopt_epochs', 'd_mopt_epochs', 'd_aha']] # converting actions to number of medications
    tmp = pd.DataFrame(tmp5, index=['OP']+polnames).T
    tmp_meds5 = (tmp.multiply(sub_ptdata1.wt, axis=0).sum()/sub_ptdata1.wt.sum()*per_capita/1e03).to_frame().T; del tmp

    ### Last year
    tmp9 = [np.select([[z in x for z in [y[healthy[rs], 9] for y in ptresults[pol]]] for x in A_class], meds)
             for pol in ['d_opt', 'd_class_mopt_epochs', 'd_mopt_epochs', 'd_aha']] # converting actions to number of medications
    tmp = pd.DataFrame(tmp9, index=['OP']+polnames).T
    tmp_meds9 = (tmp.multiply(sub_ptdata1.wt, axis=0).sum()/sub_ptdata1.wt.sum()*per_capita/1e03).to_frame().T; del tmp
    del tmp0, tmp5, tmp9

    ## Merging data frames
    rs_df = pd.concat([tmp_pi, tmp_e, tmp_meds0, tmp_meds5, tmp_meds9
                       ], axis=1)

    ## Combining scenarios
    sens_df = pd.concat([sens_df, rs_df], axis=0)
    del rs_df, sub_ptdata1
del pt_sens_sim

# Exporting results
os.chdir(results_dir)
sens_df.to_csv('Sensitivity Analysis Results.csv', index=False)

# ------------------------
# Patient-level analysis
# ------------------------

# Loading results (scenario 0 is the base case)
os.chdir(sens_dir)
with open('Sensitivity analysis results for scenario 0 using 4590 patients with 1 hour time limit and 0.001 absolute MIP gap.pkl',
          'rb') as f:
    pt_sim = pk.load(f)

# Excluding NaN values (patients that the MIP was not able to find a solution in 1 hour)
# Indexes of patients with incomplete policies
nan_index = reduce(np.union1d, (np.where([np.isnan(x).any() for x in pt_sim['V_class_mopt_epochs']])[0],
                                np.where([np.isnan(x).any() for x in pt_sim['V_mopt_epochs']])[0]))

## Indexes of patients with MIP solutions in all policies
not_nan_index = np.delete(np.arange(len(pt_sim['V_opt'])), nan_index)

## Keeping only patients with all policies
pt_sim = {k: [v[i] for i in not_nan_index] for k, v in pt_sim.items()}

## Making sure that equal polcicies do not result in different total discounted reward (to avoid numerical issues)
tmp_mopt, tmp_class_mopt = [[] for _ in range(2)]
for y, x in enumerate(pt_sim['d_opt']):

    ## Making sure the total discounted reward of CMP is at least the total discounted reward of MP
    if np.all(pt_sim['d_class_mopt_epochs'] == pt_sim['d_mopt_epochs'][y]):
        tmp_mopt.append(pt_sim['J_class_mopt_epochs'][y])
        tmp_class_mopt.append(pt_sim['J_class_mopt_epochs'][y])
    else:
        tmp_mopt.append(pt_sim['J_mopt_epochs'][y])
        tmp_class_mopt.append(pt_sim['J_class_mopt_epochs'][y])

    ## Making sure total discounted reward of OP is at least the total discounted reward of CMP
    if np.all(x == pt_sim['d_class_mopt_epochs'][y]):
        tmp_class_mopt[y] = pt_sim['J_opt'][y]
    else:
        tmp_class_mopt[y] = pt_sim['J_class_mopt_epochs'][y]

    ## Making sure total discounted reward of OP is at least the total discounted reward of MP
    if np.all(x == pt_sim['d_mopt_epochs'][y]):
        tmp_mopt[y] = pt_sim['J_opt'][y]
    else:
        tmp_mopt[y] = pt_sim['J_mopt_epochs'][y]
tmp_dict = {'J_class_mopt_epochs': tmp_class_mopt, 'J_mopt_epochs': tmp_mopt}
pt_sim.update(tmp_dict); del tmp_dict, tmp_class_mopt, tmp_mopt

## Making sure that the total expected reward is OP >= CMP >= MP (to avoid numerical issues)
tmp_class_mopt, tmp_mopt = [[] for _ in range(2)]
for y, x in enumerate(pt_sim['J_opt']):

    ## Making sure total discounted reward of CMP is at most the total discounted reward of OP
    if x < pt_sim['J_class_mopt_epochs'][y]:
        tmp_class_mopt.append(x)
    else:
        tmp_class_mopt.append(pt_sim['J_class_mopt_epochs'][y])

    ## Making sure the total discounted reward of MP is at most the total discounted reward of CMP
    if tmp_class_mopt[y] < pt_sim['J_mopt_epochs'][y]:
        tmp_mopt.append(tmp_class_mopt[y])
    else:
        tmp_mopt.append(pt_sim['J_mopt_epochs'][y])
tmp_dict = {'J_mopt_epochs': tmp_mopt, 'J_class_mopt_epochs': tmp_class_mopt}
pt_sim.update(tmp_dict); del tmp_dict, tmp_mopt, tmp_class_mopt

# Plotting parameters
s = range(6); t = 9 # indicators for alive states and year 9

# Plotting policies for selected patients
dist_id_select = [539, 472, 1116, 1660] # ids of patients used in patient-level results
dist_ind_select = np.searchsorted(pt_sim['pt_id'], dist_id_select)

## Creating dataframe of policies
policy_df = pd.DataFrame()
for i in dist_ind_select:
    # Combining results into a single data frame
    tmp = pd.concat([

        # For single year plots by state
        pd.Series(np.repeat(pt_sim['pt_id'][i], len(alive)), name='profile'),
        pd.Series(np.arange(len(alive)), name='state_id'),
        pd.Series(['Healthy', 'CHD', 'H-CHD', 'Stroke', 'H-Stroke', 'H-Both'], name='state'),

        # Policies
        pd.Series(pt_sim['d_opt'][i][s, t], name='OP'),
        pd.Series(pt_sim['d_mopt_epochs'][i][s, t], name='MP'),
        pd.Series(pt_sim['d_class_mopt_epochs'][i][s, t], name='CMP')#,
        # pd.Series(pt_sim['d_aha'][i][s, t], name='AHA')
    ],
        axis=1)
    policy_df = pd.concat([policy_df, tmp], axis=0, ignore_index=True)
pt_sim = {k: [v[i] for i in dist_ind_select] for k, v in pt_sim.items()}

# Melting dataframe
policy_df = pd.melt(policy_df, id_vars=['profile', 'state_id', 'state'], var_name='policy', value_name='action')

## Identifying treatment from actions
policy_df['trt'] = np.select([policy_df.action == x for x in range(len(drugs_concat))], drugs_concat)

## Identifying acion classes in terms of number of medications
policy_df['meds'] = np.select([[policy_df.action[y] in A_class[x] for y in range(policy_df.shape[0])] for x in range(len(A_class))], meds)

## Incorporating jitter to number of medications
policy_df['action_order'] = np.select([policy_df.action==x for x in action_order], range(len(drugs_concat)))
policy_df['meds_jitt'] = np.select([policy_df.action_order==x for x in range(len(drugs_concat))], meds_jitt)

## Plotting policies
os.chdir(fig_dir)
plot_policies_state(policy_df)

# Calculating PI in patient profiles
polnames = ['CMP', 'MP']
tmp = [[pt_sim['J_opt'][i] - y for i, y in enumerate(pt_sim[x])] for x in ['J_class_mopt_epochs', 'J_mopt_epochs']] # calculating price of interpretability per policy and removing optimal policy value functions
tmp1 = pd.DataFrame(tmp, index=polnames).T # creating data frame from list of lists
