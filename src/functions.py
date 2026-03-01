import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import networkx as nx


def network_auralization(G, m=0.9, l=50):
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    # adjacency matrix for given graph G
    A = nx.to_numpy_array(G, nodelist=nodes)

    # initialize transition matrix and with normalize columns for getting
    # transition probabilites for each node (power it may apply to its neighbors)
    col_sum = A.sum(axis=0) + 1e-32 # add small value to avoid division by zero
    P = (A / col_sum).T

    # initialize S matrix with initial condition (1 for all nodes at time 0)
    S = np.zeros((l, n))
    S[0] = np.ones(n)

    # initialize delta S matrix to zeros for initail flow between nodes 
    DELTA_S = np.zeros((n, n))

    # simulation of wave propagation over l time steps
    for t in range(1, l):
        # create diagonal matrix from St-1
        diag_S = np.diag(S[t-1])

        # update delta_S based on how much power each node applies to its neighbors
        DELTA_S = diag_S @ P + m * DELTA_S # include previous flow momentum

        # calculate inflow and outflow for each node
        inflow = DELTA_S.sum(axis=0)
        outflow = DELTA_S.sum(axis=1)

        # update S based on inflow and outflow
        S[t] = S[t-1] + inflow - outflow

    # remove DC component
    S = S - S.mean(axis=0)

    return S, nodes



def network_auralization_edges(G, m=0.9, l=50):
    nodes = list(G.nodes())
    edges = list(G.edges())
    n = len(nodes)
    m_edges = len(edges)

    # adjacency matrix for given graph G
    A = nx.to_numpy_array(G, nodelist=nodes)

    # initialize transition matrix and with normalize columns for getting
    # transition probabilites for each node (power it may apply to its neighbors)
    col_sum = A.sum(axis=0) + 1e-32 # add small value to avoid division by zero
    P = (A / col_sum).T

    # initialize S matrix with initial condition (1 for all nodes at time 0)
    S = np.zeros((l, n))
    S[0] = np.ones(n)

    # initialize delta_S matrix to zeros for initail flow between nodes 
    DELTA_S = np.zeros((n, n))

    # this is for edge-based flow, we need to track flow on edges, we create row and colum indices for edges
    # each edge (u, v) gets its own index / number, so we can track flows for it
    row_idx = np.array([u - 1 for u, v in edges])
    col_idx = np.array([v - 1 for u, v in edges])

    # Ft matrix represents flow on edges over time (like St)
    F = np.zeros((l, m_edges))

    # simulation of wave propagation over l time steps
    for t in range(1, l):
        # create diagonal matrix from St-1
        diag_S = np.diag(S[t-1])

        # update delta_S based on how much power each node applies to its neighbors
        DELTA_S = diag_S @ P + m * DELTA_S # include previous flow momentum

        # calculate inflow and outflow for each node
        inflow = DELTA_S.sum(axis=0)
        outflow = DELTA_S.sum(axis=1)

        # update S based on inflow and outflow
        S[t] = S[t-1] + inflow - outflow

    # calculate flow on edges from S, represent flow on edge (u, v) as S[u] + S[v] (flow on both nodes)
    F = S[:, row_idx] + S[:, col_idx]

    # remove DC component for edge flows
    F = F - F.mean(axis=0)

    return F, edges


# simulate power flow on network, AC or DC
def run_power_flow(network, use_dc=False):
    pp.runpp(network) if not use_dc else pp.rundcpp(network)


# main function for simulating network outage on edges and compute delta_P matrix
def network_outage_impact(network, use_dc=False, use_abs=True):
    # define number of lines and transfers in given network
    NUM_LINES = len(network.line)
    NUM_TRAFOS = len(network.trafo)
    NUM_BRANCHES = NUM_LINES + NUM_TRAFOS

    # define our delta_P matrix that represents each line impact
    # on the other lines (power outage)
    DELTA_P = np.zeros((NUM_BRANCHES, NUM_BRANCHES))

    # compute network power baseline with all lines connected
    run_power_flow(network, use_dc)
    baseline_lines = network.res_line['p_from_mw'].values
    baseline_trafos = network.res_trafo['p_hv_mw'].values
    baseline = np.concatenate([baseline_lines, baseline_trafos])

    # iterate over each line and simulate its outage, compute new power flow on network
    for i, line in enumerate(network.line.index):
        # remove current line from our network
        network.line.at[line, 'in_service'] = False

        try:
            # run full power flow on our network without currrnt line and compute new flow
            run_power_flow(network, use_dc)
            flow_lines = network.res_line['p_from_mw'].values
            flow_trafos = network.res_trafo['p_hv_mw'].values
            total_flow = np.concatenate([flow_lines, flow_trafos])

            # compute the delta between new flow and baseline, use absolute if given
            DELTA_P[i] = total_flow - baseline if not use_abs else np.abs(total_flow - baseline)

        # if exeption occured it means power flow didnt converge, means network is unstable or got disconnected, importent case to handle
        except pp.LoadflowNotConverged:
            # assign large value to delta to indicate significant impact for this line
            max_delta = np.nanmax(DELTA_P)
            DELTA_P[i] = max_delta + 0.1 * max_delta

        # add current line back to out network for later iterations
        network.line.at[line, 'in_service'] = True

    # iterate over each transformer and simulate its outage, compute new power flow on network
    for i, trafo in enumerate(network.trafo.index):
        # run full power flow on our network without currrnt transformer and compute new flow
        network.trafo.at[trafo, 'in_service'] = False

        try:
            # run full power flow on our network without currrnt transformer and compute new flow
            run_power_flow(network, use_dc)
            flow_lines = network.res_line['p_from_mw'].values
            flow_trafos = network.res_trafo['p_hv_mw'].values
            total_flow = np.concatenate([flow_lines, flow_trafos])

            # compute the delta between new flow and baseline, use absolute if given
            DELTA_P[NUM_LINES + i] = total_flow - baseline if not use_abs else np.abs(total_flow - baseline)

        # if exeption occured it means power flow didnt converge, means network is unstable or got disconnected, importent case to handle
        except pp.LoadflowNotConverged:
            # assign large value to delta to indicate significant impact for this transformer
            max_delta = np.nanmax(DELTA_P)
            DELTA_P[NUM_LINES + i] = max_delta + 0.1 * max_delta

        # add current transformer back to out network for later iterations
        network.trafo.at[trafo, 'in_service'] = True
    
    return DELTA_P, baseline