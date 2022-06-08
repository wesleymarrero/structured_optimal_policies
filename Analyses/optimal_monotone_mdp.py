# ==============================================================
# Standard LP-MDP, monotone MDP, and Class-ordered monotone MDP
# ==============================================================

# Loading modules
import numpy as np  # array operations
from policy_evaluation import evaluate_pi, evaluate_events # policy evaluation
from gurobipy import * # solver for LPs and MIPs

# Function to solve an infinite horizon MDP using the primal formulation of an LP
def lp_mdp(P, r, rterm, alpha, gamma, feasible, event_states):
    # Infinite horizon MDP using linear programming

    """
    Inputs:
    P is an S x S x T x A array of transition probabilities
    r is an S x T x A array of rewards
    rterm is an S array of terminal rewards 
    alpha is an S x T array of initial transition probabilities
    gamma is the discount factor
    """""

    """
    Outputs:
        d is the decision rule
        v is the value
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Generating tuples of states per decision epoch (including terminal condition)
    states_per_epoch = []
    for s in states:
        for t in epochs:
            states_per_epoch.append((s, t))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    V = m.addVars(states_per_epoch, lb=-GRB.INFINITY)

    # Declaring model objective
    m.setObjective(quicksum(alpha[s, t]*V[s, t] for s in states for t in dec_epochs) +
                   quicksum(alpha[s, T]*V[s, T] for s in states), GRB.MINIMIZE)

    # Adding constraints
    const1 = m.addConstrs((V[s, t] >= r[s, t, a]+gamma*quicksum(P[s, ss, t, a]*V[ss, t+1] for ss in states)
                           for s in states for t in dec_epochs for a in feasible[s][t]))
    const2 = m.addConstrs((V[s, T] >= rterm[s] for s in states))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Optimizing model
    m.optimize()

    # Storing optimal value of objective function
    J_opt = m.objVal

    # Extracting value functions and decision rule
    v = np.empty((S, T+1)); v[:] = np.nan
    d = np.empty((S, T)); d[:] = np.nan
    for t in epochs:
        for s in states:
            v[s, t] = V[s, t].X
            if t < max(epochs):
                for a in feasible[s][t]:
                    if const1[(s, t, a)].Slack == 0 and np.isnan(d[s, t]):
                        d[s, t] = a

    # Extracting occupancy measures (from dual LP) - assumes that if an action is infeasible it has occupancy 0
    occup = np.zeros((S, T+1, A))
    for t in epochs:
        for s in states:
            if t < max(epochs):
                for a in feasible[s][t]:
                    occup[s, t, a] = const1[(s, t, a)].Pi
            else:
                occup[s, t, 0] = const2[s].Pi
                occup[s, t, 1:] = 0

    # Calculating expected number of events following policy
    events = evaluate_events(d, P, event_states)

    return v, d, occup, J_opt, events

# Function to solve an infinite horizon MDP using the dual formulation of an LP
def lp_mdp_dual(P, r, rterm, alpha, gamma, infeasible, event_states):
    # Infinite horizon MDP using linear programming

    """
    Inputs:
    P is an S x S x T x A array of transition probabilities
    r is an S x T x A array of rewards
    rterm is an S array of terminal rewards 
    alpha is an S x T array of initial transition probabilities
    gamma is the discount factor
    infeasible is a list of nested lists of the index of "clinically infeasible" actions per state and decision epoch
    """""

    """
    Outputs:
        d is the decision rule
        v is the value
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    y = m.addVars(state_action_pairs)
    yterm = m.addVars(states)

    # Declaring model objective
    m.setObjective(quicksum(r[s, t, a]*y[s, t, a] for s in states for t in dec_epochs for a in actions) +
                   quicksum(rterm[s]*yterm[s] for s in states), GRB.MAXIMIZE)

    # Adding constraints
    const1 = m.addConstrs((quicksum(y[s, 0, a] for a in actions) == alpha[s, 0] for s in states))
    const2 = m.addConstrs((quicksum(y[s, t, a] for a in actions)) -
                          gamma*quicksum(P[ss, s, t-1, aa]*y[ss, t-1, aa]
                                         for ss in states for aa in actions) == alpha[s, t]
                          for s in states for t in dec_epochs[1:])
    const3 = m.addConstrs((yterm[s]-gamma*quicksum(P[ss, s, T-1, aa]*y[ss, T-1, aa]
                                                   for ss in states for aa in actions) == alpha[s, T]
                           for s in states))

    ## Constraint to ensure the policy is feasible
    m.addConstrs((quicksum(y[s, t, ia] for ia in infeasible[s][t]) == 0 for s in states for t in dec_epochs))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Optimizing model
    m.optimize()

    # Storing optimal value of objective function
    J_opt = m.objVal

    # Extracting occupancy measures and decision rule
    d = np.empty((S, T)); d[:] = np.nan
    occup = np.empty((S, T+1, A)); occup[:] = np.nan
    for t in epochs:
        for s in states:
            if t < max(epochs):
                for a in actions:
                    occup[s, t, a] = y[s, t, a].X
                d[s, t] = np.argmax(occup[s, t, :])
            else:
                occup[s, t, 0] = yterm[s].X
                occup[s, t, 1:] = 0

    # Extracting value functions (from primal LP)
    v = np.empty((S, T+1)); v[:] = np.nan
    for t in epochs:
        for s in states:
            if t == 0:
                v[s, t] = const1[s].Pi
            elif t == max(epochs):
                v[s, t] = const3[s].Pi
            else:
                v[s, t] = const2[(s, t)].Pi

    # Calculating expected number of events following policy
    events = evaluate_events(d, P, event_states)

    return v, d, occup, J_opt, events

# Function to solve an infinite horizon MDP with monotonic constraints on the states using the primal formulation of an MIP
def mip_mdp(P, r, rterm, alpha, gamma, M, infeasible, event_states, J_opt, warm):
    """Inputs:
    X is the set of states
    A is the set of actions
    r is the reward function
    P is the set of transition probability matrices
    alpha is the initial state distribution
    gamma is the discount factor
    M is used for big-M constraints
    infeasible is a list of nested lists of the index of "clinically infeasible" actions per state and decision epoch
    V_opt is the optimal value function (from LP)
    warm is the start value for the MIP
    """""

    """Outputs:
    V_mopt is the monotone optimal value function
    d_mopt is the monotone optimal decision rule
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Generating tuples of states per decision epoch (including terminal condition)
    states_per_epoch = []
    for s in states:
        for t in epochs:
            states_per_epoch.append((s, t))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    V = m.addVars(states_per_epoch, lb=-GRB.INFINITY)
    d = m.addVars(state_action_pairs, vtype=GRB.BINARY)

    # Declaring model objective
    m.setObjective(quicksum(alpha[s, t]*V[s, t] for s in states for t in dec_epochs) +
                   quicksum(alpha[s, T]*V[s, T] for s in states), GRB.MAXIMIZE)

    # Warm starting the MIP
    for s in states:
        for t in dec_epochs:
            for a in actions:
                d[s, t, a].start = warm[s, t, a]

    # Adding constraints
    m.addConstrs((V[s, t] <= r[s, t, a]+gamma*quicksum(P[s, ss, t, a]*V[ss, t+1] for ss in states) +
                  M[s, t, a]*(1-d[s, t, a]) for s in states for t in dec_epochs for a in actions))
    m.addConstrs((V[s, T] <= rterm[s] for s in states))
    m.addConstrs((quicksum(d[s, t, a] for a in actions) == 1 for s in states for t in dec_epochs))
    m.addConstrs((d[s, t, a] <= quicksum(d[s+1, t, aa] for aa in [aa for aa in actions if aa >= a])
                  for s in [s for s in states if s < max(states)] for t in dec_epochs for a in actions))

    ## Constraint to ensure the policy is feasible
    m.addConstrs((quicksum(d[s, t, ia] for ia in infeasible[s][t]) == 0 for s in states for t in dec_epochs))

    ## Additional constraints to aid solving the MIPs
    m.addConstr(quicksum(alpha[s, t]*V[s, t] for s in states for t in dec_epochs) +
                quicksum(alpha[s, T]*V[s, T] for s in states), GRB.LESS_EQUAL, J_opt, "")

    for s in states:
        for t in dec_epochs:
            m.addSOS(GRB.SOS_TYPE1, [d[s, t, a] for a in actions], list(range(1, A+1)))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Storing only 1 MIP solution
    m.setParam('PoolSolutions', 1)

    # Changing the focus of the MIP solver #use 2 to focus on proving optimality #use 3 to focus on the bound
    m.setParam('MIPFocus', 3)

    # Changing the tolerance level of the MIP
    m.setParam('MIPGapAbs', 0.001) # 0.019 for within 1 week of perfect health #0.0025 for within 1 day of perfect health #0.01 for 1% of a year of perfect health

    # Optimizing model
    m.optimize()

    # Extracting objective value, optimal value function, and optimal policy
    d_mopt = np.empty((S, T)); d_mopt[:] = np.nan
    V_mopt = np.empty((S, T+1)); V_mopt[:] = np.nan
    if m.Status == 2: # Model was solved to optimality
        # Storing optimal value of objective function
        J_mopt = m.objVal

        # Extracting optimal value function and optimal policy
        for t in epochs:
            for s in states:
                V_mopt[s, t] = V[s, t].X
                if t < max(epochs):
                    for a in actions:
                        if d[s, t, a].X == 1:
                            d_mopt[s, t] = a

        # Calculating expected number of events following policy
        e_mopt = evaluate_events(d_mopt, P, event_states)
    else: # Display warning message and do not store results (model was not solve to optimality
        # print("Monotone MDP in states was not solved to optimality. Status code: ", m.Status)
        # if hasattr(m, 'ObjVal'):
        #     print("Final absolute MIP gap value: ", float(m.MIPGap)*abs(m.ObjVal))
        # else:
        #     print("Final absolute MIP gap value: ", "NA")
        J_mopt = np.nan # Indicator that the MIP was not solved to optimality
        e_mopt = np.empty((S, T+1)); e_mopt[:] = np.nan  # Indicator that the MIP was not solved to optimality

    return V_mopt, d_mopt, J_mopt, e_mopt

# Function to solve an infinite horizon MDP with monotonic constraints on the states using the dual formulation of an MIP
def mip_mdp_dual(P, r, rterm, alpha, gamma, infeasible, event_states, J_opt, warm):
    """
    Inputs:
    P is an S x S x T x A array of transition probabilities
    r is an S x T x A array of rewards
    rterm is an S array of terminal rewards 
    alpha is an S x T array of initial transition probabilities
    gamma is the discount factor
    infeasible is a list of nested lists of the index of "clinically infeasible" actions per state and decision epoch
    """""

    """
    Outputs:
        d is the decision rule
        v is the value
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    x = m.addVars(state_action_pairs)
    pi = m.addVars(state_action_pairs, vtype=GRB.BINARY)
    xterm = m.addVars(states)
    piterm = m.addVars(states, vtype=GRB.BINARY)

    # Warm starting the MIP
    for s in states:
        for t in dec_epochs:
            for a in actions:
                pi[s, t, a].start = warm[s, t, a]

    # Declaring model objective
    m.setObjective(quicksum(r[s, t, a]*x[s, t, a] for s in states for t in dec_epochs for a in actions) +
                   quicksum(rterm[s]*xterm[s] for s in states), GRB.MAXIMIZE)

    # Adding constraints
    m.addConstrs((quicksum(x[s, 0, a] for a in actions) == alpha[s, 0] for s in states))
    m.addConstrs((quicksum(x[s, t, a] for a in actions)) -
                  gamma*quicksum(P[ss, s, t-1, aa]*x[ss, t-1, aa]
                                 for ss in states for aa in actions) == alpha[s, t]
                  for s in states for t in dec_epochs[1:])
    m.addConstrs((xterm[s]-gamma*quicksum(P[ss, s, T-1, aa]*x[ss, T-1, aa]
                                           for ss in states for aa in actions) == alpha[s, T]
                   for s in states))
    m.addConstrs((quicksum(pi[s, t, a] for a in actions) == 1 for s in states for t in dec_epochs))
    m.addConstrs((x[s, t, a] <= pi[s, t, a] for s in states for t in dec_epochs for a in actions))
    m.addConstrs((xterm[s] <= piterm[s] for s in states))
    m.addConstrs((pi[s, t, a] <= quicksum(pi[s+1, t, aa] for aa in [aa for aa in actions if aa >= a])
                  for s in [s for s in states if s < max(states)] for t in dec_epochs for a in actions))

    ## Constraint to ensure the policy is feasible
    m.addConstrs((quicksum(x[s, t, ia] for ia in infeasible[s][t]) == 0 for s in states for t in dec_epochs))
    
    ## Additional constraints to aid solving the MIPs
    m.addConstr(quicksum(r[s, t, a]*x[s, t, a] for s in states for t in dec_epochs for a in actions) +
                quicksum(rterm[s]*xterm[s] for s in states), GRB.LESS_EQUAL, J_opt, "")

    for s in states:
        for t in dec_epochs:
            m.addSOS(GRB.SOS_TYPE1, [pi[s, t, a] for a in actions], list(range(1, A+1)))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Storing only 1 MIP solution
    m.setParam('PoolSolutions', 1)

    # Changing the focus of the MIP solver #use 2 to focus on proving optimality #use 3 to focus on the bound
    m.setParam('MIPFocus', 3)

    # Changing the tolerance level of the MIP
    m.setParam('MIPGapAbs', 0.001) # 0.019 for within 1 week of perfect health #0.0025 for within 1 day of perfect health #0.01 for 1% of a year of perfect health

    # Not pre-solving the model to avoid numerical issues?
    m.setParam('Presolve', 0)

    # Optimizing model
    m.optimize()

    # Extracting objective value and optimal policy
    d_mopt = np.empty((S, T)); d_mopt[:] = np.nan
    if m.Status == 2: # Model was solved to optimality
        # Storing optimal value of objective function
        J_mopt = m.objVal

        # Extracting decision rule
        for t in epochs:
            for s in states:
                if t < max(epochs):
                    for a in actions:
                        if np.round(pi[s, t, a].X) > 0:
                            d_mopt[s, t] = a

        # Evaluating policy
        V_mopt = evaluate_pi(d_mopt, P, r, rterm, gamma)

        # Calculating expected number of events following policy
        e_mopt = evaluate_events(d_mopt, P, event_states)

    else: # Display warning message and do not store results (model was not solve to optimality
        # print("Monotone MDP in states was not solved to optimality. Status code: ", m.Status)
        # if hasattr(m, 'ObjVal'):
        #     print("Final absolute MIP gap value: ", float(m.MIPGap)*abs(m.ObjVal))
        # else:
        #     print("Final absolute MIP gap value: ", "NA")
        J_mopt = np.nan  # Indicator that the MIP was not solved to optimality
        V_mopt = np.empty((S, T+1)); V_mopt[:] = np.nan # Indicator that the MIP was not solved to optimality
        e_mopt = np.empty((S, T+1)); e_mopt[:] = np.nan # Indicator that the MIP was not solved to optimality

    return V_mopt, d_mopt, J_mopt, e_mopt

# Function to solve an infinite horizon MDP with monotonic constraints on the states and decision epochs using the dual formulation of an MIP
def mip_mdp_dual_epochs(P, r, rterm, alpha, gamma, infeasible, event_states, J_opt, warm):
    """
    Inputs:
    P is an S x S x T x A array of transition probabilities
    r is an S x T x A array of rewards
    rterm is an S array of terminal rewards 
    alpha is an S x T array of initial transition probabilities
    gamma is the discount factor
    infeasible is a list of nested lists of the index of "clinically infeasible" actions per state and decision epoch
    """""

    """
    Outputs:
        d is the decision rule
        v is the value
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    x = m.addVars(state_action_pairs)
    pi = m.addVars(state_action_pairs, vtype=GRB.BINARY)
    xterm = m.addVars(states)
    piterm = m.addVars(states, vtype=GRB.BINARY)

    # Warm starting the MIP
    for s in states:
        for t in dec_epochs:
            for a in actions:
                pi[s, t, a].start = warm[s, t, a]

    # Declaring model objective
    m.setObjective(quicksum(r[s, t, a]*x[s, t, a] for s in states for t in dec_epochs for a in actions)+
                   quicksum(rterm[s]*xterm[s] for s in states), GRB.MAXIMIZE)

    # Adding constraints
    m.addConstrs((quicksum(x[s, 0, a] for a in actions) == alpha[s, 0] for s in states))
    m.addConstrs((quicksum(x[s, t, a] for a in actions))-
                 gamma*quicksum(P[ss, s, t-1, aa]*x[ss, t-1, aa]
                                for ss in states for aa in actions) == alpha[s, t]
                 for s in states for t in dec_epochs[1:])
    m.addConstrs((xterm[s]-gamma*quicksum(P[ss, s, T-1, aa]*x[ss, T-1, aa]
                                          for ss in states for aa in actions) == alpha[s, T]
                  for s in states))
    m.addConstrs((quicksum(pi[s, t, a] for a in actions) == 1 for s in states for t in dec_epochs))
    m.addConstrs((x[s, t, a] <= pi[s, t, a] for s in states for t in dec_epochs for a in actions))
    m.addConstrs((xterm[s] <= piterm[s] for s in states))
    m.addConstrs((pi[s, t, a] <= quicksum(pi[s+1, t, aa] for aa in [aa for aa in actions if aa >= a])
                  for s in [s for s in states if s < max(states)] for t in dec_epochs for a in actions))

    ## Constraint to ensure the policy is feasible
    m.addConstrs((quicksum(x[s, t, ia] for ia in infeasible[s][t]) == 0 for s in states for t in dec_epochs))

    ## Additional constraints to guarantee monotonicity over decision epochs
    m.addConstrs((pi[s, t, a] <= quicksum(pi[s, t+1, aa] for aa in [aa for aa in actions if aa >= a])
                  for s in states for t in dec_epochs[:-1] for a in actions))

    ## Additional constraints to aid solving the MIPs
    m.addConstr(quicksum(r[s, t, a]*x[s, t, a] for s in states for t in dec_epochs for a in actions)+
                quicksum(rterm[s]*xterm[s] for s in states), GRB.LESS_EQUAL, J_opt, "")

    for s in states:
        for t in dec_epochs:
            m.addSOS(GRB.SOS_TYPE1, [pi[s, t, a] for a in actions], list(range(1, A+1)))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Storing only 1 MIP solution
    m.setParam('PoolSolutions', 1)

    # Changing the focus of the MIP solver #use 2 to focus on proving optimality #use 3 to focus on the bound
    m.setParam('MIPFocus', 3)

    # Changing the tolerance level of the MIP
    m.setParam('MIPGapAbs', 0.001) # 0.019 for within 1 week of perfect health #0.0025 for within 1 day of perfect health #0.01 for 1% of a year of perfect health

    # Not pre-solving the model to avoid numerical issues?
    m.setParam('Presolve', 0)

    # Optimizing model
    m.optimize()

    # Extracting objective value and optimal policy
    d_mopt = np.empty((S, T)); d_mopt[:] = np.nan
    if m.Status == 2:  # Model was solved to optimality
        # Storing optimal value of objective function
        J_mopt = m.objVal

        # Extracting decision rule
        for t in epochs:
            for s in states:
                if t < max(epochs):
                    for a in actions:
                        if np.round(pi[s, t, a].X) > 0:
                            d_mopt[s, t] = a

        # Evaluating policy
        V_mopt = evaluate_pi(d_mopt, P, r, rterm, gamma)

        # Calculating expected number of events following policy
        e_mopt = evaluate_events(d_mopt, P, event_states)

    else: # Display warning message and do not store results (model was not solved to optimality)
        # print("Monotone MDP in states and decision epochs was not solved to optimality. Status code: ", m.Status)
        # if hasattr(m, 'ObjVal'):
        #     print("Final absolute MIP gap value: ", float(m.MIPGap)*abs(m.ObjVal))
        # else:
        #     print("Final absolute MIP gap value: ", "NA")
        J_mopt = np.nan  # Indicator that the MIP was not solved to optimality
        V_mopt = np.empty((S, T+1)); V_mopt[:] = np.nan  # Indicator that the MIP was not solved to optimality
        e_mopt = np.empty((S, T+1)); e_mopt[:] = np.nan  # Indicator that the MIP was not solved to optimality

    return V_mopt, d_mopt, J_mopt, e_mopt

# Function to solve an infinite horizon MDP with class-ordered monotonic constraints on the states using the primal formulation of an MIP
def mip_mdp_classes(P, r, rterm, alpha, gamma, M, infeasible, event_states, S_class, A_class, J_opt, warm):

    """Inputs:
    X is the set of states
    A is the set of actions
    r is the reward function
    P is the set of transition probability matrices
    alpha is the initial state distribution
    gamma is the discount factor
    M is used for big-M constraints
    infeasible is a list of nested lists of the index of "clinically infeasible" actions per state and decision epoch
    """""

    """Outputs:
    V_opt is the optimal value function
    d_opt is the optimal decision rule
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Generating tuples of states per decision epoch (including terminal condition)
    states_per_epoch = []
    for s in states:
        for t in epochs:
            states_per_epoch.append((s, t))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    V = m.addVars(states_per_epoch, lb=-GRB.INFINITY)
    d = m.addVars(state_action_pairs, vtype=GRB.BINARY)

    # Warm starting the MIP
    for s in states:
        for t in dec_epochs:
            for a in actions:
                d[s, t, a].start = warm[s, t, a]

    # Declaring model objective
    m.setObjective(quicksum(alpha[s, t]*V[s, t] for s in states for t in dec_epochs) +
                   quicksum(alpha[s, T]*V[s, T] for s in states), GRB.MAXIMIZE)

    # Adding constraints
    m.addConstrs((V[s, t] <= r[s, t, a]+gamma*quicksum(P[s, ss, t, a]*V[ss, t+1] for ss in states)+
                  M[s, t, a]*(1-d[s, t, a]) for s in states for t in dec_epochs for a in actions))
    m.addConstrs((V[s, T] <= rterm[s] for s in states))
    m.addConstrs((quicksum(d[s, t, a] for a in actions) == 1 for s in states for t in dec_epochs))

    state_pairs = []
    for i in range(len(S_class)-1):
        for s in S_class[i]:
            for ss in S_class[i+1]:
                state_pairs.append((s, ss))

        for ac in A_class:
            m.addConstrs((quicksum(d[s, t, a] for a in ac) <= quicksum(
                d[ss, t, aa] for aa in [aa for aa in actions if aa >= min(ac)])
                          for (s, ss) in state_pairs for t in dec_epochs))

    ## Constraint to ensure the policy is feasible
    m.addConstrs((quicksum(d[s, t, ia] for ia in infeasible[s][t]) == 0 for s in states for t in dec_epochs))

    ## Additional constraints to aid solving the MIPs
    m.addConstr(quicksum(alpha[s, t]*V[s, t] for s in states for t in dec_epochs)+
                quicksum(alpha[s, T]*V[s, T] for s in states), GRB.LESS_EQUAL, J_opt, "")

    for s in states:
        for t in dec_epochs:
            m.addSOS(GRB.SOS_TYPE1, [d[s, t, a] for a in actions], list(range(1, A+1)))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Storing only 1 MIP solution
    m.setParam('PoolSolutions', 1)

    # Changing the focus of the MIP solver #use 2 to focus on proving optimality #use 3 to focus on the bound
    m.setParam('MIPFocus', 3)

    # Optimizing model
    m.optimize()

    # Extracting objective value, optimal value function, and optimal policy
    d_class_mopt = np.empty((S, T)); d_class_mopt[:] = np.nan
    V_class_mopt = np.empty((S, T+1)); V_class_mopt[:] = np.nan
    if m.Status == 2: # Model was solve to optimality
        # Storing optimal value of objective function
        J_class_mopt = m.objVal

        # Extracting optimal value function and optimal policy
        for t in epochs:
            for s in states:
                V_class_mopt[s, t] = V[s, t].X
                if t < max(epochs):
                    for a in actions:
                        if d[s, t, a].X == 1:
                            d_class_mopt[s, t] = a

        # Calculating expected number of events following policy
        e_class_mopt = evaluate_events(d_class_mopt, P, event_states)

    else: # Display warning message and do not store results (model was not solved to optimality)
        # print("Class-ordered monotone MDP in states was not solved to optimality. Status code: ", m.Status)
        # if hasattr(m, 'ObjVal'):
        #     print("Final absolute MIP gap value: ", float(m.MIPGap)*abs(m.ObjVal))
        # else:
        #     print("Final absolute MIP gap value: ", "NA")
        J_class_mopt = np.nan  # Indicator that the MIP was not solved to optimality
        e_class_mopt = np.empty((S, T+1)); e_class_mopt[:] = np.nan  # Indicator that the MIP was not solved to optimality

    return V_class_mopt, d_class_mopt, J_class_mopt, e_class_mopt

# Function to solve an infinite horizon MDP with class-ordered monotonic constraints on the states using the dual formulation of an MIP
def mip_mdp_dual_classes(P, r, rterm, alpha, gamma, infeasible, event_states, S_class, A_class, J_opt, warm):
    """
    Inputs:
    P is an S x S x T x A array of transition probabilities
    r is an S x T x A array of rewards
    rterm is an S array of terminal rewards 
    alpha is an S x T array of initial transition probabilities
    gamma is the discount factor
    infeasible is a list of nested lists of the index of "clinically infeasible" actions per state and decision epoch
    """""

    """
    Outputs:
        d is the decision rule
        v is the value
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    x = m.addVars(state_action_pairs)
    pi = m.addVars(state_action_pairs, vtype=GRB.BINARY)
    xterm = m.addVars(states)
    piterm = m.addVars(states, vtype=GRB.BINARY)

    # Warm starting the MIP
    for s in states:
        for t in dec_epochs:
            for a in actions:
                pi[s, t, a].start = warm[s, t, a]

    # Declaring model objective
    m.setObjective(quicksum(r[s, t, a]*x[s, t, a] for s in states for t in dec_epochs for a in actions)+
                   quicksum(rterm[s]*xterm[s] for s in states), GRB.MAXIMIZE)

    # Adding constraints
    m.addConstrs((quicksum(x[s, 0, a] for a in actions) == alpha[s, 0] for s in states))
    m.addConstrs((quicksum(x[s, t, a] for a in actions))-
                 gamma*quicksum(P[ss, s, t-1, aa]*x[ss, t-1, aa]
                                for ss in states for aa in actions) == alpha[s, t]
                 for s in states for t in dec_epochs[1:])
    m.addConstrs((xterm[s]-gamma*quicksum(P[ss, s, T-1, aa]*x[ss, T-1, aa]
                                          for ss in states for aa in actions) == alpha[s, T]
                  for s in states))
    m.addConstrs((quicksum(pi[s, t, a] for a in actions) == 1 for s in states for t in dec_epochs))
    m.addConstrs((x[s, t, a] <= pi[s, t, a] for s in states for t in dec_epochs for a in actions))
    m.addConstrs((xterm[s] <= piterm[s] for s in states))

    state_pairs = []
    for i in range(len(S_class)-1):
        for s in S_class[i]:
            for ss in S_class[i+1]:
                state_pairs.append((s, ss))

        for ac in A_class:
            m.addConstrs((quicksum(pi[s, t, a] for a in ac) <=
                          quicksum(pi[ss, t, aa] for aa in [aa for aa in actions if aa >= min(ac)])
                          for (s, ss) in state_pairs for t in dec_epochs))

    ## Constraint to ensure the policy is feasible
    m.addConstrs((quicksum(x[s, t, ia] for ia in infeasible[s][t]) == 0 for s in states for t in dec_epochs))

    ## Additional constraints to aid solving the MIPs
    m.addConstr(quicksum(r[s, t, a]*x[s, t, a] for s in states for t in dec_epochs for a in actions)+
                quicksum(rterm[s]*xterm[s] for s in states), GRB.LESS_EQUAL, J_opt, "")

    for s in states:
        for t in dec_epochs:
            m.addSOS(GRB.SOS_TYPE1, [pi[s, t, a] for a in actions], list(range(1, A+1)))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Storing only 1 MIP solution
    m.setParam('PoolSolutions', 1)

    # Changing the focus of the MIP solver #use 2 to focus on proving optimality #use 3 to focus on the bound
    m.setParam('MIPFocus', 3)

    # Changing the tolerance level of the MIP
    m.setParam('MIPGapAbs', 0.001) # 0.019 for within 1 week of perfect health #0.0025 for within 1 day of perfect health #0.01 for 1% of a year of perfect health

    # Not pre-solving the model to avoid numerical issues?
    m.setParam('Presolve', 0)

    # Optimizing model
    m.optimize()

    # Extracting objective value and optimal policy
    d_class_mopt = np.empty((S, T)); d_class_mopt[:] = np.nan
    if m.Status == 2:  # Model was solved to optimality
        # Storing optimal value of objective function
        J_class_mopt = m.objVal

        # Extracting decision rule
        for t in epochs:
            for s in states:
                if t < max(epochs):
                    for a in actions:
                        if np.around(pi[s, t, a].X) > 0:
                            d_class_mopt[s, t] = a

        V_class_mopt = evaluate_pi(d_class_mopt, P, r, rterm, gamma)

        # Calculating expected number of events following policy
        e_class_mopt = evaluate_events(d_class_mopt, P, event_states)

    else: # Display warning message and do not store results (model was not solved to optimality)
        # print("Class-ordered monotone MDP in states was not solved to optimality. Status code: ", m.Status)
        # if hasattr(m, 'ObjVal'):
        #     print("Final absolute MIP gap value: ", float(m.MIPGap)*abs(m.ObjVal))
        # else:
        #     print("Final absolute MIP gap value: ", "NA")
        J_class_mopt = np.nan  # Indicator that the MIP was not solved to optimality
        V_class_mopt = np.empty((S, T+1)); V_class_mopt[:] = np.nan  # Indicator that the MIP was not solved to optimality
        e_class_mopt = np.empty((S, T+1)); e_class_mopt[:] = np.nan  # Indicator that the MIP was not solved to optimality

    return V_class_mopt, d_class_mopt, J_class_mopt, e_class_mopt

# Function to solve an infinite horizon MDP with class-ordered monotonic constraints on the states and decision epochs using the dual formulation of an MIP
def mip_mdp_dual_classes_epochs(P, r, rterm, alpha, gamma, infeasible, event_states, S_class, A_class, J_opt, warm):
    """
    Inputs:
    P is an S x S x T x A array of transition probabilities
    r is an S x T x A array of rewards
    rterm is an S array of terminal rewards 
    alpha is an S x T array of initial transition probabilities
    gamma is the discount factor
    infeasible is a list of nested lists of the index of "clinically infeasible" actions per state and decision epoch
    """""

    """
    Outputs:
        d is the decision rule
        v is the value
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    x = m.addVars(state_action_pairs)
    pi = m.addVars(state_action_pairs, vtype=GRB.BINARY)
    xterm = m.addVars(states)
    piterm = m.addVars(states, vtype=GRB.BINARY)

    # Warm starting the MIP
    for s in states:
        for t in dec_epochs:
            for a in actions:
                pi[s, t, a].start = warm[s, t, a]

    # Declaring model objective
    m.setObjective(quicksum(r[s, t, a]*x[s, t, a] for s in states for t in dec_epochs for a in actions)+
                   quicksum(rterm[s]*xterm[s] for s in states), GRB.MAXIMIZE)

    # Adding constraints
    m.addConstrs((quicksum(x[s, 0, a] for a in actions) == alpha[s, 0] for s in states))
    m.addConstrs((quicksum(x[s, t, a] for a in actions))-
                 gamma*quicksum(P[ss, s, t-1, aa]*x[ss, t-1, aa]
                                for ss in states for aa in actions) == alpha[s, t]
                 for s in states for t in dec_epochs[1:])
    m.addConstrs((xterm[s]-gamma*quicksum(P[ss, s, T-1, aa]*x[ss, T-1, aa]
                                          for ss in states for aa in actions) == alpha[s, T]
                  for s in states))
    m.addConstrs((quicksum(pi[s, t, a] for a in actions) == 1 for s in states for t in dec_epochs))
    m.addConstrs((x[s, t, a] <= pi[s, t, a] for s in states for t in dec_epochs for a in actions))
    m.addConstrs((xterm[s] <= piterm[s] for s in states))

    state_pairs = []
    for i in range(len(S_class)-1):
        for s in S_class[i]:
            for ss in S_class[i+1]:
                state_pairs.append((s, ss))

        for ac in A_class:
            m.addConstrs((quicksum(pi[s, t, a] for a in ac) <=
                          quicksum(pi[ss, t, aa] for aa in [aa for aa in actions if aa >= min(ac)])
                          for (s, ss) in state_pairs for t in dec_epochs))

    ## Constraint to ensure the policy is feasible
    m.addConstrs((quicksum(x[s, t, ia] for ia in infeasible[s][t]) == 0 for s in states for t in dec_epochs))

    ## Additional constraints to guarantee monotonicity in decision epochs
    for ac in A_class:
        m.addConstrs((quicksum(pi[s, t, a] for a in ac) <=
                      quicksum(pi[s, t+1, aa] for aa in [aa for aa in actions if aa >= min(ac)])
                      for s in states for t in dec_epochs[:-1]))

    ## Additional constraints to aid solving the MIPs
    m.addConstr(quicksum(r[s, t, a]*x[s, t, a] for s in states for t in dec_epochs for a in actions)+
                quicksum(rterm[s]*xterm[s] for s in states), GRB.LESS_EQUAL, J_opt, "")

    for s in states:
        for t in dec_epochs:
            m.addSOS(GRB.SOS_TYPE1, [pi[s, t, a] for a in actions], list(range(1, A+1)))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Storing only 1 MIP solution
    m.setParam('PoolSolutions', 1)

    # Changing the focus of the MIP solver #use 2 to focus on proving optimality #use 3 to focus on the bound
    m.setParam('MIPFocus', 3)

    # Changing the tolerance level of the MIP
    m.setParam('MIPGapAbs', 0.001) # 0.019 for within 1 week of perfect health #0.0025 for within 1 day of perfect health #0.01 for 1% of a year of perfect health

    # Not pre-solving the model to avoid numerical issues?
    m.setParam('Presolve', 0)

    # Optimizing model
    m.optimize()

    # Extracting objective value and optimal policy
    d_class_mopt = np.empty((S, T)); d_class_mopt[:] = np.nan
    if m.Status == 2:  # Model was solved to optimality
        # Storing optimal value of objective function
        J_class_mopt = m.objVal

        # Extracting decision rule
        for t in epochs:
            for s in states:
                if t < max(epochs):
                    for a in actions:
                        if np.round(pi[s, t, a].X) > 0:
                            d_class_mopt[s, t] = a

        # Evaluating policy
        V_class_mopt = evaluate_pi(d_class_mopt, P, r, rterm, gamma)

        # Calculating expected number of events following policy
        e_class_mopt = evaluate_events(d_class_mopt, P, event_states)

    else: # Display warning message and do not store results (model was not solved to optimality)
        # print("Class-ordered monotone MDP in states and decision epochs was not solved to optimality. Status code: ", m.Status)
        # if hasattr(m, 'ObjVal'):
        #     print("Final absolute MIP gap value: ", float(m.MIPGap)*abs(m.ObjVal))
        # else:
        #     print("Final absolute MIP gap value: ", "NA")
        J_class_mopt = np.nan  # Indicator that the MIP was not solved to optimality
        V_class_mopt = np.empty((S, T+1)); V_class_mopt[:] = np.nan  # Indicator that the MIP was not solved to optimality
        e_class_mopt = np.empty((S, T+1)); e_class_mopt[:] = np.nan  # Indicator that the MIP was not solved to optimality

    return V_class_mopt, d_class_mopt, J_class_mopt, e_class_mopt

# Function to solve an infinite horizon MDP using the dual formulation of an LP (with constraints based on AHA's guidelines or risk-based policies)
def fixed_lp_dual(P, r, rterm, alpha, gamma, action_class_meds, pi_fixed, event_states):

    """
    Inputs:
    pi_fixed is a fixed policy determined using the 2017 AHA hypertension guidelines or a risk threshold
    P is an S x S x T x A array of transition probabilities
    r is an S x T x A array of rewards
    rterm is an S array of terminal rewards 
    alpha is an S x T array of initial transition probabilities
    gamma is the discount factor
    infeasible is a list of nested lists of the index of "clinically infeasible" actions per state and decision epoch
    """""

    """
    Outputs:
        d is the decision rule
        v is the value
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs
    A = P.shape[3]  # number of actions

    # Creating lists of states and actions
    states = list(np.arange(0, S, 1))
    dec_epochs = list(np.arange(0, T, 1))
    epochs = list(np.arange(0, T+1, 1))
    actions = list(np.arange(0, A, 1))

    # Generating list of tuples of state-action pairs
    state_action_pairs = []
    for s in states:
        for t in dec_epochs:
            for a in actions:
                state_action_pairs.append((s, t, a))

    # Creating Gurobi model object
    m = Model()

    # Adding decision variables to model
    y = m.addVars(state_action_pairs)
    yterm = m.addVars(states)

    # Declaring model objective
    m.setObjective(quicksum(r[s, t, a]*y[s, t, a] for s in states for t in dec_epochs for a in actions) +
                   quicksum(rterm[s]*yterm[s] for s in states), GRB.MAXIMIZE)

    # Adding constraints
    const1 = m.addConstrs((quicksum(y[s, 0, a] for a in actions) == alpha[s, 0] for s in states))
    const2 = m.addConstrs((quicksum(y[s, t, a] for a in actions)) -
                          gamma*quicksum(P[ss, s, t-1, aa]*y[ss, t-1, aa]
                                         for ss in states for aa in actions) == alpha[s, t]
                          for s in states for t in dec_epochs[1:])
    const3 = m.addConstrs((yterm[s]-gamma*quicksum(P[ss, s, T-1, aa]*y[ss, T-1, aa]
                                                   for ss in states for aa in actions) == alpha[s, T]
                           for s in states))

    ## Constraint to ensure the policy satisfies AHA's guidelines or risk-based policy (including feasibility if possible) - each action class contains a different number of medications
    m.addConstrs((quicksum(y[s, t, na] for na in [item for sublist in [x for x in action_class_meds if x != action_class_meds[pi_fixed[s, t].astype(int)]]
                                                  for item in sublist]) == 0 for s in states for t in dec_epochs))

    # Processing model specifications
    m.update()

    # Surpressing output
    m.setParam('OutputFlag', False)

    # Setting time limit to 1 hour
    m.setParam('TimeLimit', 3600)

    # Optimizing model
    m.optimize()

    # Storing optimal value of objective function
    J_opt = m.objVal

    # Extracting occupancy measures and decision rule
    d = np.empty((S, T)); d[:] = np.nan
    occup = np.empty((S, T+1, A)); occup[:] = np.nan
    for t in epochs:
        for s in states:
            if t < max(epochs):
                for a in actions:
                    occup[s, t, a] = y[s, t, a].X
                d[s, t] = np.argmax(occup[s, t, :])
            else:
                occup[s, t, 0] = yterm[s].X
                occup[s, t, 1:] = 0

    # Extracting value functions (from primal LP)
    v = np.empty((S, T+1)); v[:] = np.nan
    for t in epochs:
        for s in states:
            if t == 0:
                v[s, t] = const1[s].Pi
            elif t == max(epochs):
                v[s, t] = const3[s].Pi
            else:
                v[s, t] = const2[(s, t)].Pi

    # Calculating expected number of events following policy
    events = evaluate_events(d, P, event_states)

    return v, d, J_opt, events

# Function to evaluate the no treatment policy
def notrt(P, r, rterm, gamma, event_states):

    """"
        Evaluating performance of no treatment in finite horizon MDP

        Inputs:
            P: transition probabilities of the MDP
            r: rewards of the MDP
            rterm: terminal rewards
            gamma: discount factor
            event_states: indicators of whether a state is associated with an ASCVD event

        Outputs: 
            V_pi: value functions associated with no treatment
            events: expected number of ASCVD events following policy pi
    """""

    # Extrating parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of decision epochs

    # Matrix representing "no treatment"
    d_notrt = np.zeros((S, T), dtype=int)

    # Evaluating the no treatment policy
    V_notrt = evaluate_pi(d_notrt, P, r, rterm, gamma)

    # Calculating expected number of events following policy
    events = evaluate_events(d_notrt, P, event_states)

    return V_notrt, events
