# =========================================================
# Calculating relative risk reductions based on risk slopes
# =========================================================

def new_risk(sbpreduc, riskslope, pretrtrisk, event):

    """"
    Calculating post-treatment risk for each event type

    Inputs:
        riskslope: relative risk estimates of CHD and stroke events
        pretrtrisk: pre-treatment 1-year risk of CHD and stroke
        sbpredc: SBP reductions from treatment
        event: 0=CHD, 1=stroke

    Outputs:
        risk: post-treatment 1-year risk of CHD and stroke
    """""

    RR = (list(riskslope)[event])**(sbpreduc/20)
    risk = RR*pretrtrisk

    return risk
