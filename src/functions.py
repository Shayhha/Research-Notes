import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import networkx as nx


# represents network auralization algorithm for nodes in numpy
def network_auralization(G, nodes, sig=None, m=0.9, l=100):
    n = len(nodes)

    # adjacency matrix for given graph G
    A = nx.to_numpy_array(G, nodelist=nodes, weight='weight')

    # initialize transition matrix and with normalize columns for getting
    # transition probabilites for each node (power it may apply to its neighbors)
    col_sum = A.sum(axis=0) + 1e-32 # add small value to avoid division by zero
    P = (A / col_sum).T

    # initialize S matrix with initial condition
    S = np.zeros((l, n))

    # initialize delta S matrix to zeros for initail flow between nodes 
    DELTA_S = np.zeros((n, n))

    # if no signal provided, use default condition (1 for all nodes at time 0)
    if sig is None:
        sig = np.zeros(l)
        sig[0] = 1

    # simulation of wave propagation over l time steps
    for t in range(1, l):
        # add signal injection to St-1 before propagation
        S[t-1] += sig[(t-1) % len(sig)]

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

    return S


# represents network auralization algorithm for edges in numpy (custom)
def network_auralization_edges(G, nodes, edges, sig=None, m=0.9, l=100):
    n = len(nodes)
    m_edges = len(edges)
    node_index = {node: i for i, node in enumerate(nodes)}

    # adjacency matrix for given graph G
    A = nx.to_numpy_array(G, nodelist=nodes, weight='weight')

    # initialize transition matrix and with normalize columns for getting
    # transition probabilites for each node (power it may apply to its neighbors)
    col_sum = A.sum(axis=0) + 1e-32 # add small value to avoid division by zero
    P = (A / col_sum).T

    # initialize S matrix with initial condition
    S = np.zeros((l, n))

    # initialize delta_S matrix to zeros for initail flow between nodes 
    DELTA_S = np.zeros((n, n))

    #? this is for edge-based flow, we need to track flow on edges, we create row and colum indices for edges
    #? each edge (u, v) gets its own index / number, so we can track flows for it
    row_idx = np.array([node_index[u] for u, v in edges])
    col_idx = np.array([node_index[v] for u, v in edges])

    #? Ft matrix represents flow on edges over time (like St)
    F = np.zeros((l, m_edges))

    # if no signal provided, use default condition (1 for all nodes at time 0)
    if sig is None:
        sig = np.zeros(l)
        sig[0] = 1

    # simulation of wave propagation over l time steps
    for t in range(1, l):
        # add signal injection to St-1 before propagation
        S[t-1] += sig[(t-1) % len(sig)]

        # create diagonal matrix from St-1
        diag_S = np.diag(S[t-1])

        # update delta_S based on how much power each node applies to its neighbors
        DELTA_S = diag_S @ P + m * DELTA_S # include previous flow momentum

        # calculate inflow and outflow for each node
        inflow = DELTA_S.sum(axis=0)
        outflow = DELTA_S.sum(axis=1)

        # update S based on inflow and outflow
        S[t] = S[t-1] + inflow - outflow

    #? calculate flow on edges from S, represent flow on edge (u, v) as S[u] + S[v] (flow on both nodes)
    F = S[:, row_idx] + S[:, col_idx]

    #? remove DC component for edge flows
    F = F - F.mean(axis=0)

    return F


# function for building network graph including various power network features for buses and branches
def build_network_graph(network: pp.pandapowerNet, is_weighted=True):
    G = nx.MultiGraph()
    NUM_LINES = len(network.line)

    # represents nodes and edges lists, we maintain them for later easy access to features
    nodes, edges = [], []

    # add nodes to our network graph
    for bus in network.bus.index:
        vn_kv = network.bus.at[bus, 'vn_kv'] # nominal voltage
        p_load = network.load[network.load.bus == bus].p_mw.sum() if not network.load.empty else 0.0 # active load
        q_load = network.load[network.load.bus == bus].q_mvar.sum() if not network.load.empty else 0.0 # reactive load
        p_sgen = network.sgen[network.sgen.bus == bus].p_mw.sum() if not network.sgen.empty else 0.0 # active static generation
        q_sgen = network.sgen[network.sgen.bus == bus].q_mvar.sum() if not network.sgen.empty else 0.0 # reactive static generation
        p_gen = network.gen[network.gen.bus == bus].p_mw.sum() if not network.gen.empty else 0.0 # active conventional generation

        G.add_node(bus, vn_kv=vn_kv, p_load=p_load, q_load=q_load, p_sgen=p_sgen, q_sgen=q_sgen, p_gen=p_gen)
        nodes.append(bus)

    # add edges for lines to our network graph
    for idx, (pp_idx, line) in enumerate(network.line.iterrows()):
        r = line.r_ohm_per_km * line.length_km # resistance of line
        x = line.x_ohm_per_km * line.length_km # reactance of line
        z = np.sqrt(r**2 + x**2) # impedance magnitude |Z| = sqrt(R^2 + X^2)
        y_mag = 1 / max(z, 1e-6) # admittance magnitude |Y| = 1 / |Z|
        rx_ratio = r / max(x, 1e-6) # ratio between resistance & reactance

        G.add_edge(line.from_bus, line.to_bus, key=f'line_{idx}', index=idx, pp_index=pp_idx, type='line', r=r, x=x, z=z, y_mag=y_mag, rx_ratio=rx_ratio, weight=y_mag if is_weighted else 1.0)
        edges.append((line.from_bus, line.to_bus, f'line_{idx}'))

    # add edges for transformers to our network graph
    for idx, (pp_idx, trafo) in enumerate(network.trafo.iterrows()):
        r = trafo.vkr_percent / 100 * (trafo.vn_hv_kv ** 2) / trafo.sn_mva # resistance of line
        x = np.sqrt(max((trafo.vk_percent / 100)**2 - (trafo.vkr_percent / 100)**2, 0)) * (trafo.vn_hv_kv ** 2) / trafo.sn_mva # reactance of line
        z = np.sqrt(r**2 + x**2) # impedance magnitude
        y_mag = 1 / max(z, 1e-6) # admittance magnitude
        rx_ratio = r / (x + 1e-6) # ratio between resistance & reactance

        G.add_edge(trafo.hv_bus, trafo.lv_bus, key=f'trafo_{idx}', index=NUM_LINES + idx, pp_index=pp_idx, type='trafo', r=r, x=x, z=z, y_mag=y_mag, rx_ratio=rx_ratio, weight=y_mag if is_weighted else 1.0)
        edges.append((trafo.hv_bus, trafo.lv_bus, f'trafo_{idx}'))

    # we return new graph with nodes and edges lists
    return G, nodes, edges


# simulate power flow on network, AC or DC
def run_power_flow(network: pp.pandapowerNet, use_dc=False):
    pp.runpp(network) if not use_dc else pp.rundcpp(network)


# main function for simulating network outage on edges and compute delta_P matrix
def network_outage_impact(network: pp.pandapowerNet, G: nx.MultiGraph, edges, use_dc=False, use_abs=True):
    # define number of branches in given network
    NUM_BRANCHES = len(edges)

    # define our delta_P matrix that represents each line impact
    # on the other lines (power outage)
    DELTA_P = np.zeros((NUM_BRANCHES, NUM_BRANCHES))

    # compute network power baseline with all lines connected
    run_power_flow(network, use_dc)
    baseline_lines = network.res_line['p_from_mw'].values
    baseline_trafos = network.res_trafo['p_hv_mw'].values
    baseline = np.concatenate([baseline_lines, baseline_trafos])

    # iterate over each line/tarnfo and simulate its outage, compute new power flow on network
    for (u, v, k) in edges:
        # get edge indices and type for simulating outage
        edge_idx = G[u][v][k]['index']
        pp_idx = G[u][v][k]['pp_index']
        edge_type = G[u][v][k]['type']

        # remove current line/tranfo from our network
        if edge_type == 'line':
            network.line.at[pp_idx, 'in_service'] = False
        else:
            network.trafo.at[pp_idx, 'in_service'] = False

        try:
            # run full power flow on our network without currrnt line/tranfo and compute new flow
            run_power_flow(network, use_dc)
            flow_lines = network.res_line['p_from_mw'].values
            flow_trafos = network.res_trafo['p_hv_mw'].values
            total_flow = np.concatenate([flow_lines, flow_trafos])

            # compute the delta between new flow and baseline, use absolute if given
            DELTA_P[edge_idx] = total_flow - baseline if not use_abs else np.abs(total_flow - baseline)

        #! if exception occured it means power flow didnt converge, means network is unstable or got disconnected, importent case to handle
        except pp.LoadflowNotConverged:
            # assign large value to delta to indicate significant impact for this line/tranfo
            max_delta = np.nanmax(DELTA_P)
            DELTA_P[edge_idx] = max_delta + 0.1 * max_delta

        # add current line/tranfo back to out network for later iterations
        if edge_type == 'line':
            network.line.at[pp_idx, 'in_service'] = True
        else:
            network.trafo.at[pp_idx, 'in_service'] = True
    
    return DELTA_P, baseline


# helper fucntion for computing flow centrality on given network using our simulation
def compute_flow_centrality(network: pp.pandapowerNet, G: nx.MultiGraph, edges, use_dc=False, normalize=False):
    DELTA_P, _ = network_outage_impact(network, G, edges, use_dc=use_dc)
    impact = np.linalg.norm(DELTA_P, axis=1)

    # return normalized impact values using min-max normalization
    if normalize:
        impact = (impact - impact.min()) / (impact.max() - impact.min() + 1e-12)

    return impact