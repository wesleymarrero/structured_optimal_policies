# =======================================================
# Patient simulation - hypertension treatment case study
# =======================================================

# Loading modules
import numpy as np  # array operations
from termcolor import colored # colored warnings
from ascvd_risk import arisk  # risk calculations
from transition_probabilities import TP  # transition probability calculations
from optimal_monotone_mdp import lp_mdp_dual, mip_mdp_dual_epochs, mip_mdp_dual_classes_epochs, fixed_lp_dual, notrt  # MDPs and no treatment policy
from aha_2017_guideline import aha_guideline # ACC/AHA's 2017 guidelines

# Patient simulation function
def patient_sim(pt_id, patientdata, numhealth, healthy, dead, years, events, stroke_hist, ascvd_hist, event_states,
                lifedata, mortality_rates, chddeathdata, strokedeathdata, alldeathdata, riskslopedata, sbpmin, dbpmin,
                sbpmax, dbpmax, alldrugs, trtharm, QoL, QoLterm, alpha, gamma, state_order, S_class, action_order, A_class,
                action_class_meds, targetrisk, targetsbp, targetdbp, numeds):

    """"
    This function generates risk estimates and transition probabilities
    to determine treatment policies per patient. It is a continuation of hypertension_treatment_monotone_mdp.py; 
    it was made a function to parallelize operations.  
    """""

    # Assume that the patient has the same pre-treatment SBP/DBP no matter the health condition
    pretrtsbp = np.ones([numhealth, years])*np.array(patientdata.sbp)
    pretrtdbp = np.ones([numhealth, years])*np.array(patientdata.dbp)

    # Storing risk calculations
    ascvdrisk1 = np.empty((numhealth, years, events))  # 1-y CHD and stroke risk (for transition probabilities)
    periodrisk1 = np.empty((numhealth, years, events))  # 1-y risk after scaling

    ascvdrisk10 = np.empty((numhealth, years, events))  # 10-y CHD and stroke risk (for AHA's guidelines)
    periodrisk10 = np.empty((numhealth, years, events))  # 10-y risk after scaling

    for h in range(numhealth):  # each state (in order of rewards)
        for t in range(years):  # each age

            # Changing scaling factor based on age
            if patientdata.age.iloc[t] >= 60:
                ascvd_hist_sim = ascvd_hist
                ascvd_hist_sim[stroke_hist, 1] = 2
            else:
                ascvd_hist_sim = ascvd_hist

            for k in range(events):  # each event type

                # 1-year ASCVD risk calculation (for transition probabilities)
                ascvdrisk1[h, t, k] = arisk(k, patientdata.sex.iloc[t], patientdata.race.iloc[t], patientdata.age.iloc[t],
                                            patientdata.sbp.iloc[t], patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                            patientdata.hdl.iloc[t], patientdata.diab.iloc[t], 0, 1)

                # 10-year ASCVD risk calculation (for AHA's guidelines)
                ascvdrisk10[h, t, k] = arisk(k, patientdata.sex.iloc[t], patientdata.race.iloc[t], patientdata.age.iloc[t],
                                             patientdata.sbp.iloc[t], patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                             patientdata.hdl.iloc[t], patientdata.diab.iloc[t], 0, 10)

                if ascvd_hist_sim[h, k] > 1:
                    # Scaling odds of 1-year risks
                    periododds = ascvdrisk1[h, t, k]/(1-ascvdrisk1[h, t, k])
                    periododds = ascvd_hist_sim[h, k]*periododds
                    periodrisk1[h, t, k] = periododds/(1+periododds)

                    # Scaling odds of 10-year risks
                    periododds = ascvdrisk10[h, t, k]/(1-ascvdrisk10[h, t, k])
                    periododds = ascvd_hist_sim[h, k]*periododds
                    periodrisk10[h, t, k] = periododds/(1+periododds)

                elif ascvd_hist_sim[h, k] == 0:  # set risk to 0
                    periodrisk1[h, t, k] = 0
                    periodrisk10[h, t, k] = 0
                else:  # no scale
                    periodrisk1[h, t, k] = ascvdrisk1[h, t, k]
                    periodrisk10[h, t, k] = ascvdrisk10[h, t, k]

    # life expectancy and death likelihood data index
    if patientdata.sex.iloc[0] == 0:  # male
        sexcol = 1  # column in deathdata corresponding to male
    else:
        sexcol = 2  # column in deathdata corresponding to female

    # Death rates
    chddeath = chddeathdata.iloc[list(np.where([i in list(patientdata.age) for i in list(chddeathdata.iloc[:, 0])])[0]), sexcol]
    strokedeath = strokedeathdata.iloc[list(np.where([i in list(patientdata.age) for i in list(strokedeathdata.iloc[:, 0])])[0]), sexcol]
    alldeath = alldeathdata.iloc[list(np.where([i in list(patientdata.age) for i in list(alldeathdata.iloc[:, 0])])[0]), sexcol]

    # Risk slopes (for BP reductions)
    riskslope = riskslopedata.iloc[list(np.where([i in list(patientdata.age) for i in list(riskslopedata.iloc[:, 0])])[0]), 1:3].reset_index(drop=True, inplace=False)

    # Calculating transition probabilities
    P, feas = TP(periodrisk1, chddeath, strokedeath, alldeath, riskslope, pretrtsbp, pretrtdbp, sbpmin, dbpmin, sbpmax, dbpmax, alldrugs)

    # Sorting transition probabilities and feasibility indicators accroding to state ordering
    P = P[state_order, :, :, :]; P = P[:, state_order, :, :]; feas = feas[state_order, :, :]

    # Sorting transition probabilities and feasibility indicators accroding to action ordering
    P = P[:, :, :, action_order]; feas = feas[:, :, action_order]

    # Extracting list of infeasible actions per state and decision epoch
    infeasible = []  # stores index of infeasible actions
    feasible = []  # stores index of feasible actions
    for s in range(feas.shape[0]):
        tmp = []; tmp1 = []
        for t in range(feas.shape[1]):
            tmp.append(list(np.where(feas[s, t, :] == 0)[0]))
            tmp1.append(list(np.where(feas[s, t, :] == 1)[0]))
        infeasible.append(tmp); feasible.append(tmp1); del tmp, tmp1
    del feas

    # Calculating expected rewards
    r = np.empty((numhealth, years, len(alldrugs))); r[:] = np.nan  # stores rewards
    for t in range(years):
        # QoL weights by age
        qol = None
        if 40 <= patientdata.age.iloc[t] <= 44:
            qol = QoL.get("40-44")
        elif 45 <= patientdata.age.iloc[t] <= 54:
            qol = QoL.get("45-54")
        elif 55 <= patientdata.age.iloc[t] <= 64:
            qol = QoL.get("55-64")
        elif 65 <= patientdata.age.iloc[t] <= 74:
            qol = QoL.get("65-74")
        elif 75 <= patientdata.age.iloc[t] <= 84:
            qol = QoL.get("75-84")
        qol = np.array(qol)[state_order]  # Ordering rewards

        # Subtracting treatment disutility
        harmsort = np.array(trtharm)[action_order] # sorting disutilities according to action order
        for a in range(len(alldrugs)):
            r[:, t, a] = [max(0, rw-harmsort[a]) for rw in qol]  # bounding rewards below by zero (assuming there is nothing worse than death)

    # Terminal conditions
    ## Healthy life expectancy
    healthy_lifexp = lifedata.iloc[np.where(patientdata.age.iloc[max(range(years))] == lifedata.iloc[:, 0])[0][0], sexcol]

    ## Mortality rates by gender
    if patientdata.sex.iloc[0] == 0:  # Male mortality rates
        SMR = mortality_rates.get("Males <2 CHD events")
    else:  # Female mortality rates
        SMR = mortality_rates.get("Females <2 CHD events")

    ## Calculating terminal rewards (terminal QALYs by age)
    rterm = None
    if 40 <= patientdata.age.iloc[max(range(years))] <= 44:
        rterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("40-44"))]
    elif 45 <= patientdata.age.iloc[max(range(years))] <= 54:
        rterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("45-54"))]
    elif 55 <= patientdata.age.iloc[max(range(years))] <= 64:
        rterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("55-64"))]
    elif 65 <= patientdata.age.iloc[max(range(years))] <= 74:
        rterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("65-74"))]
    elif 75 <= patientdata.age.iloc[max(range(years))] <= 84:
        rterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("75-84"))]
    rterm = np.array(rterm)[state_order] # Ordering terminal rewards

    # Ordering state distribution
    alpha = alpha[state_order, :] # initial state distribution for MIPs
    alpha1 = np.ones(alpha.shape) # initial state distribution for LPs

    # Determining optimal policies (using dual formulation)
    V_opt, d_opt, occup, J_opt, e_opt = lp_mdp_dual(P, r, rterm, alpha1, gamma, infeasible, event_states)
    d_opt[dead, :] = 0  # treating only on alive states

    ## Correcting total expected discounted reward with actual initial state distribution
    J_opt = np.dot(alpha.flatten(), V_opt.flatten())

    # Generating monotone policy from optimal solution (enforcing monotonicity by increasing actions) to warm start MIPs
    ws_mopt = np.empty((numhealth, years, len(alldrugs))); ws_mopt[:] = np.nan
    for h in range(numhealth):
        for t in range(years):
            for a in range(len(alldrugs)):
                ws_mopt[h, t, a] = np.where(d_opt[h, t] == a, 1, 0)
            if h > 0:
                if np.argmax(ws_mopt[h, t, :]) < np.argmax(ws_mopt[(h-1), t, :]):
                    ws_mopt[h, t, int(np.argmax(ws_mopt[h, t, :]))] = 0
                    ws_mopt[h, t, int(np.argmax(ws_mopt[(h-1), t, :]))] = 1

    # Determining monotone policies in states and decision epochs (using dual formulation)
    V_mopt_epochs, d_mopt_epochs, J_mopt_epochs, e_mopt_epochs = mip_mdp_dual_epochs(P, r, rterm, alpha, gamma, infeasible, event_states, J_opt, ws_mopt)
    d_mopt_epochs[dead, :] = 0  # treating only on alive states

    # Determining class-ordered monotone policies in states and decision epochs (using dual formulation)
    V_class_mopt_epochs, d_class_mopt_epochs, J_class_mopt_epochs, e_class_mopt_epochs = mip_mdp_dual_classes_epochs(P, r, rterm, alpha, gamma, infeasible, event_states, S_class, A_class, J_opt, ws_mopt)
    d_class_mopt_epochs[dead, :] = 0  # treating only on alive states

    # Determining policy based on the 2017 AHA's guidelines
    pi_aha = aha_guideline(periodrisk10, pretrtsbp, pretrtdbp, targetrisk, targetsbp, targetdbp, sbpmin, dbpmin,
                           riskslope, numeds, healthy)

    ## Making sure clinical guidelines are feasible
    feas_meds_list = [[np.unique(np.select([[y in action_class_meds[x] for y in fst] for x in range(len(action_class_meds))], np.arange(numeds+1))).tolist()
                        for fst in fs] for fs in feasible] # feasible number of medications
    for h in range(numhealth):
        for t in range(years):
            if pi_aha[h, t] not in feas_meds_list[h][t]: # checking feasibility of policy
                pi_aha[h, t] = np.where((pi_aha[h, t]+1) in feas_meds_list[h][t], (pi_aha[h, t]+1), (pi_aha[h, t]-1)).astype(int)  # checking if adding/subtracting 1 medication is feasible
                if pi_aha[h, t] not in feas_meds_list[h][t]: # if neither is feasible, returning the largest number of medications feasible (with warning)
                    print(colored("Warning: Feasibility conditions not met for clinical guidelines in patient " + str(pt_id) + " on year " + str(t+1), "blue"))
                    pi_aha[h, t] = max(feas_meds_list[h][t])

    # Finding drug classes that maximize QALYs based on AHA's policy (used as a constraint)
    ## Note: action class meds need to be re-arranged to match P and r in the scenario where the actions are ordered according to the DBP reductions (scenario 5)
    ## This is not needed for the sensitivity analyses (the policy remains constant across all scenarios).
    ## The policies following the clinical guidelines in this scenario are not valid (or necessary).
    V_aha, d_aha, J_aha, e_aha = fixed_lp_dual(P, r, rterm, alpha1, gamma, action_class_meds, pi_aha.astype(int), event_states)
    d_aha[dead, :] = 0  # treating only on alive states

    ## Correcting total expected discounted reward with actual initial state distribution
    J_aha = np.dot(alpha.flatten(), V_aha.flatten())

    # Evaluating no treatment policy
    V_notrt, e_notrt = notrt(P, r, rterm, gamma, event_states)

    # Changing policies back to original order (to match alldrugs list - it also allows for comparison across sensitivity scenarios)
    ## Making sure that all LPs and MIPs have been solved to optimality (otherwise the results of the patient will be excluded when reporting results)
    if ~np.isnan(np.stack((d_opt, d_class_mopt_epochs, d_mopt_epochs, d_aha))).any():
        d_opt = np.array(action_order)[d_opt.flatten().astype(int)].reshape(d_opt.shape)
        d_class_mopt_epochs = np.array(action_order)[d_class_mopt_epochs.flatten().astype(int)].reshape(d_class_mopt_epochs.shape)
        d_mopt_epochs = np.array(action_order)[d_mopt_epochs.flatten().astype(int)].reshape(d_mopt_epochs.shape)
        d_aha = np.array(action_order)[d_aha.flatten().astype(int)].reshape(d_aha.shape)

    print("Patient " + str(pt_id) + " Done")

    return (pt_id, V_notrt, e_notrt,
            V_opt, d_opt, occup, J_opt, e_opt,
            V_mopt_epochs, d_mopt_epochs, J_mopt_epochs, e_mopt_epochs,
            V_class_mopt_epochs, d_class_mopt_epochs, J_class_mopt_epochs, e_class_mopt_epochs,
            V_aha, d_aha, J_aha, e_aha
            )
