# =====================================
# Calculating transition probabilities
# =====================================

# Loading modules
import numpy as np
from post_treatment_risk import new_risk
from sbp_reductions_drugtype import sbp_reductions
from dbp_reductions_drugtype import dbp_reductions

# Transition probabilities' calculation
def TP(periodrisk, chddeath, strokedeath, alldeath, riskslope, pretrtsbp, pretrtdbp, sbpmin, dbpmin,
       sbpmax, dbpmax, alldrugs):

    """"
    Calculating probability of health states transitions given ASCVD risk, fatality likelihoods,
    and probability of mortality due to non-ASCVD events
    
    Inputs:
        periodrisk: 1-year risk of CHD and stroke
        chddeath: likelihood of death given a CHD event
        strokedeath: likelihood of death given a stroke event 
        alldeath: likelihood of death due to non-ASCVD events
        riskslope: relative risk estimates of CHD and stroke events
        pretrtsbp: pre-treatment SBP
        pretrtdbp: pre-treatment DBP
        sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
        dbpmin: minimum DBP allowed (clinical constraint)
        alldrugs: treatment options being considered (196 trts: no treatment plus 1 to 5 drugs from 5 different types at standard dosage)
    
    Outputs:
        ptrans: transition probabilities
        feasible: indicator of whether the treatment is clinically feasible at each state and year
    """""

    # Extracting parameters
    numhealth = periodrisk.shape[0]  # number of states
    years = periodrisk.shape[1]  # number of non-stationary stages
    events = periodrisk.shape[2]  # number of events
    numtrt = len(alldrugs)  # number of treatment choices

    # Storing feasibility indicators
    feasible = np.empty((numhealth, years, numtrt)); feasible[:] = np.nan  # indicators of whether the treatment is clinically feasible at each state and year

    # Storing risk and TP calculations
    risk = np.empty((numhealth, years, events, numtrt)); risk[:] = np.nan  # stores post-treatment risks
    ptrans = np.zeros((numhealth, numhealth, years, numtrt))  # state transition probabilities--default of 0, to reduce coding/computations

    # Storing BP reductions
    sbpreduc = np.empty((numhealth, years, numtrt)); sbpreduc[:] = np.nan  # stores SBP reductions
    dbpreduc = np.empty((numhealth, years, numtrt)); dbpreduc[:] = np.nan  # stores DBP reductions

    for t in range(years):  # each non-stationary stage (year)
        for h in range(numhealth):  # each health state
            for j in range(numtrt):  # each treatment
                if j == 0:  # the do nothing treatment
                    sbpreduc[h, t, j] = 0; dbpreduc[h, t, j] = 0  # no reduction when taking 0 drugs
                    if pretrtsbp[h, t] > sbpmax or pretrtdbp[h, t] > dbpmax:
                        feasible[h, t, j] = 0  # must give treatment
                    else:
                        feasible[h, t, j] = 1  # do nothing is always feasible
                else: # prescibe >0 drugs
                    sbpreduc[h, t, j] = sbp_reductions(j, pretrtsbp[h, t], alldrugs)
                    dbpreduc[h, t, j] = dbp_reductions(j, pretrtdbp[h, t], alldrugs)
                    newsbp = pretrtsbp[h, t] - sbpreduc[h, t, j]
                    newdbp = pretrtdbp[h, t] - dbpreduc[h, t, j]
                    if (newsbp < sbpmin or pretrtsbp[h, t] < 0) or (newdbp < dbpmin or dbpreduc[h, t, j] < 0):  # violates minimum allowable SBP/DBP constraint (or reductions are innacurate [i.e., negative]) and the do nothing option is feasible
                        feasible[h, t, j] = 0
                    else:  # does not violate min sbp/dbp
                        feasible[h, t, j] = 1

                for k in range(events):  # each event type
                    # Calculating post-treatment risks
                    risk[h, t, k, j] = new_risk(sbpreduc[h, t, j], riskslope.iloc[t, :], periodrisk[h, t, k], k)

                # Health state transition probabilities: allows for both CHD and stroke in same period
                # Let Dead state dominate the transition to all others
                if h == 6 or h == 7 or h == 8 or h == 9:  # Dead
                    ptrans[h, 9, t, j] = 1  # must stay dead  [and go to the overall dead state]
                else:  # alternate denotes the state that is default state if neither a CHD even stroke, nor death [from CHD, stroke or other] occurs
                    if h == 3:  # History of CHD and Stroke
                        alternate = 3
                    elif h == 4 or h == 1:  # CHD Event or History of CHD
                        alternate = 1
                    elif h == 5 or h == 2:  # Stroke or History of Stroke
                        alternate = 2
                    else:  # Healthy
                        alternate = 0

                    quits = 0
                    while quits == 0:  # compute transition probabilities, using a "break" command if you've exceeded 1 (this never happens!)
                        ptrans[h, 8, t, j] = min(1, strokedeath.iloc[t] * risk[h, t, 1, j])  # likelihood of death from stroke
                        cumulprob = ptrans[h, 8, t, j]

                        ptrans[h, 7, t, j] = min(1, chddeath.iloc[t] * risk[h, t, 0, j])  # likelihood of death from CHD event
                        if cumulprob + ptrans[h, 7, t, j] >= 1:  # check for invalid probabilities
                            ptrans[h, 7, t, j] = 1 - cumulprob
                            break  # all other probabilities should be left as 0 [as initialized before loop]
                        cumulprob += ptrans[h, 7, t, j]

                        ptrans[h, 6, t, j] = min(1, alldeath.iloc[t])  # likelihood of death from non CVD cause
                        if cumulprob + ptrans[h, 6, t, j] >= 1:  # check for invalid probabilities
                            ptrans[h, 6, t, j] = 1 - cumulprob
                            break  # all other probabilities should be left as 0 [as initialized before loop]
                        cumulprob += ptrans[h, 6, t, j]

                        ptrans[h, 5, t, j] = min(1, (1 - strokedeath.iloc[t]) * risk[h, t, 1, j])  # likelihood of having stroke and surviving
                        if cumulprob + ptrans[h, 5, t, j] >= 1:  # check for invalid probabilities
                            ptrans[h, 5, t, j] = 1 - cumulprob
                            break  # all other probabilities should be left as 0 [as initialized before loop]
                        cumulprob += ptrans[h, 5, t, j]

                        ptrans[h, 4, t, j] = min(1, (1 - chddeath.iloc[t]) * risk[h, t, 0, j])  # likelihood of having CHD and surviving
                        if cumulprob + ptrans[h, 4, t, j] >= 1:  # check for invalid probabilities
                            ptrans[h, 4, t, j] = 1 - cumulprob
                            break  # all other probabilities should be left as 0 [as initialized before loop]
                        cumulprob += ptrans[h, 4, t, j]

                        ptrans[h, alternate, t, j] = 1 - cumulprob  # otherwise, you go to the alternate state
                        break  # computed all probabilities, now quit

            # Making sure that no treatment is feasible if nothing else is (useful for when one BP reading is very high and the other is very low - see patient 4)
            if feasible[h, t, :].max() == 0:
                feasible[h, t, 0] = 1

    return ptrans, feasible
