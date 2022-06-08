# ================================
# Finite horizon policy evaluation
# ================================

# Loading modules
import numpy as np  # array operations

# Finite horizon policy evaluation function
def evaluate_pi(pi, P, r, rterm, gamma):

    """"
        Evaluating policy in finite horizon MDP
        
        Inputs:
            pi: policy to be evaluated
            P: transition probabilities of the MDP
            r: rewards of the MDP
            rterm: terminal rewards
            gamma: discount factor
        
        Outputs: 
            V_pi: value functions following policy pi
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs

    #Array to store value function (including terminal rewards)
    V_pi = np.empty((S, T+1)); V_pi[:] = np.nan

    #Terminal value functions
    V_pi[:, T] = rterm

    #Policy evaluation
    for t in reversed(range(T)): 
        for s in range(S):
            V_pi[s, t] = r[s, t, int(pi[s, t])] + gamma*np.dot(P[s, :, t, int(pi[s, t])], V_pi[:, t+1])

    return V_pi

# Finite horizon policy evaluation in terms of events function
def evaluate_events(pi, P, event_states):

    """"
        Evaluating policy in finite horizon MDP

        Inputs:
            pi: policy to be evaluated
            P: transition probabilities of the MDP
            event_states: indicators of whether a state is associated with an ASCVD event

        Outputs: 
            E_pi: expected number of ASCVD events following policy pi
    """""

    # Extrating parameters
    S = P.shape[0] # number of states
    T = P.shape[1] # number of decision epochs

    # Array to store value function
    E_pi = np.empty((S, T+1)); E_pi[:] = np.nan # stores expected number of events following the policy being evaluated

    # Terminal event indicators
    E_pi[:, T] = event_states

    # Policy evaluation
    for t in reversed(range(T)):
        for s in range(S):
            E_pi[s, t] = event_states[s] + np.dot(P[s, :, t, int(pi[s, t])], E_pi[:, t+1])

    return E_pi

