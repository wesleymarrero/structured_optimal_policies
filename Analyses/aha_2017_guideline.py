# ==========================================================================
# 2017 ACC/AHA's Guideline for Management of High Blood Pressure in Adults
# ==========================================================================

# Loading modules
import numpy as np
from post_treatment_risk import new_risk
from sbp_reductions_drugtype import sbp_reductions_generic
from dbp_reductions_drugtype import dbp_reductions_generic

# Function to obtain policy according to the AHA's guideline
def aha_guideline(pretrtrisk, pretrtsbp, pretrtdbp, targetrisk, targetsbp, targetdbp, sbpmin, dbpmin,
                  riskslope, numtrt, healthy):

    """"
    Inputs:
        pretrtrisk: pre-treatment 1-year risk of CHD and stroke
        pretrtsbp: pre-treatment SBP
        pretrtdbp: pre-treatment DBP
        targetrisk: target risk recommended by the guidelines
        targetsbp: target SBP recommended by the guidelines
        targetdbp: target DBP recommended by the guidelines
        sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
        dbpmin: minimum DBP allowed (clinical constraint)
        riskslope: relative risk estimates of CHD and stroke events
        numtrt: maximum number of drugs combined
        healthy: indicator that the patient has not had an ASCVD event

    Outputs:
        policy: number of medications recommended by the guidelines
    """""

    # Extracting parameters
    numhealth = pretrtrisk.shape[0]  # number of states
    years = pretrtrisk.shape[0]

    # Arrays to store results (initializing with no treatmnet)
    policy = np.empty((numhealth, years)); policy[:] = np.nan

    # Determining action per stage
    for t in range(years):
        for h in range(numhealth):  # each health state

            # Identifying patient's past treatment
            if t == min(range(years)):
                past_trt = 0  # start with no treatment
            else:
                past_trt = policy[h, t-1]  # evaluate last patient's treatment first

            # Calculating post-treatment risk and BP with past treatment
            sbpreduc = sbp_reductions_generic(past_trt, pretrtsbp[h, t])
            dbpreduc = dbp_reductions_generic(past_trt, pretrtdbp[h, t])

            post_trt_risk = new_risk(sbpreduc, riskslope.iloc[t, :], pretrtrisk[h, t, 0], 0) +\
                            new_risk(sbpreduc, riskslope.iloc[t, :], pretrtrisk[h, t, 1], 1)
            post_trt_sbp = pretrtsbp[h, t] - sbpreduc
            post_trt_dbp = pretrtdbp[h, t] - dbpreduc

            # Making sure that BP is not on target without increasing treatment
            if ((post_trt_risk >= targetrisk or h != healthy) and (post_trt_sbp >= 130 or post_trt_dbp >= 80))\
                    or ((post_trt_sbp >= 140 or post_trt_dbp >= 90) and (post_trt_sbp < targetsbp+20 and post_trt_dbp < targetdbp+10)): # High risk (or history of ASCVD) with stage 1 hypertension or stage 2 hypertension with BP within 20/10 mm Hg of target

                # Simulating 1-month evaluations within each year
                month = 1  # initial month
                while month <= 12 and (post_trt_sbp >= targetsbp or post_trt_dbp >= targetdbp):  # BP not on target with current medication within the same year

                    # Attempting to increase treatment
                    if (past_trt + 1) > numtrt:
                        new_trt = past_trt # cannot give more than 5 medications
                    else:
                        new_trt = past_trt + 1 # increase medication intensity

                    # Calculating post-treatment BP with new potential treatment
                    sbpreduc = sbp_reductions_generic(new_trt, pretrtsbp[h, t])
                    post_trt_sbp = pretrtsbp[h, t] - sbpreduc
                    dbpreduc = dbp_reductions_generic(new_trt, pretrtdbp[h, t])
                    post_trt_dbp = pretrtdbp[h, t] - dbpreduc

                    # Evaluating the feasibility of new treatment
                    if (post_trt_sbp < sbpmin or sbpreduc < 0) or (post_trt_dbp < dbpmin or dbpreduc < 0):
                        policy[h, t] = past_trt # new treatment is not feasible
                    else:
                        policy[h, t] = new_trt  # new treatment is feasible

                    past_trt = policy[h, t] # next month's evaluation
                    month += 1 # next month's evaluation
            elif post_trt_sbp >= targetsbp+20 or post_trt_dbp >= targetdbp+10: # Stage 2 hypertension and 20/10 mm Hg above target
                # Simulating 1-month evaluations within each year
                month = 1  # initial month
                while month <= 12 and (post_trt_sbp >= targetsbp or post_trt_dbp >= targetdbp):  # BP not on target with current medication within the same year

                    # Attempting to increase treatment
                    if (past_trt + 1) > numtrt:
                        new_trt = past_trt # cannot give more than 5 medications
                    elif past_trt < 2:
                        new_trt = 2 # patients should be treated with at least two agents (unless not feasible)
                    else:
                        new_trt = past_trt + 1 # increase medication intensity

                    # Calculating post-treatment BP with new potential treatment
                    sbpreduc = sbp_reductions_generic(new_trt, pretrtsbp[h, t])
                    post_trt_sbp = pretrtsbp[h, t] - sbpreduc
                    dbpreduc = dbp_reductions_generic(new_trt, pretrtdbp[h, t])
                    post_trt_dbp = pretrtdbp[h, t] - dbpreduc

                    # Evaluating the feasibility of new treatment
                    if (post_trt_sbp < sbpmin or sbpreduc < 0) or (post_trt_dbp < dbpmin or dbpreduc < 0): # new treatment is not feasible
                        # Considering less intensive treatment (useful for patients with a very high BP reading and the other not high enough to make 2 agents feasible - see patient 7)
                        alt_trt = new_trt - 1

                        # Calculating post-treatment BP with new potential treatment
                        sbpreduc = sbp_reductions_generic(alt_trt, pretrtsbp[h, t])
                        post_trt_sbp = pretrtsbp[h, t]-sbpreduc
                        dbpreduc = dbp_reductions_generic(alt_trt, pretrtdbp[h, t])
                        post_trt_dbp = pretrtdbp[h, t]-dbpreduc

                        # Evaluating the feasibility of alternative treatment
                        if (post_trt_sbp < sbpmin or sbpreduc < 0) or (post_trt_dbp < dbpmin or dbpreduc < 0):  # alternative treatment is not feasible either
                            policy[h, t] = past_trt
                        else: # alternative treatment is feasible
                            policy[h, t] = alt_trt
                    else: # new treatment is feasible
                        policy[h, t] = new_trt

                    past_trt = policy[h, t]  # next month's evaluation
                    month += 1 # next month's evaluation
            else: # BP already on target keeping past year's treatment
                policy[h, t] = past_trt # keep current treatment

    return policy
