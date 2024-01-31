#Test comment
# import required packages

import numpy as np
import pandas as pd
from scipy import io
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import time as time
import igraph as igraph

#Define colormaps for matrix plotting functions
OFFDIAG_CMAP = mpl.colors.LinearSegmentedColormap.from_list("mycmap", ['0.1', '0.9'])
DIAG_CMAP = 'seismic'


#DW model with dynamic confidence implemented in igraph
def DW(graph, mu, delta, gamma, random_seed = None, 
       t_start = 0, confidence_start = None,
       tol = 0.02, Tmax = 10**8, Tmin = 0, T_acc = None, sigfigs = 3,
       check_convergence = True, return_confidence_changes = False,
       return_opinion_series = False, opinion_timestep = 1, 
       return_animation_matrix = False, matrix_timestep = 100, nodeorder = None):
    
    """
    Simulates the Deffaunt--Weisbuch model of opinion dynamics with dynamic confidence bounds.
    Calculates an "effective subgraph" consisting of edges with opinion difference within the confidence
    in order to sort the nodes into clusters and check convergence of the model.
    
    Parameters
    ------------
    graph : igraph.graph
        igraph Graph with 'opinion' node-level attribute and n nodes
    c : float
        Initial confidence radius. We set this to be the homogeneous confidence radius on all edges initially
        
        !!!!!!!!!!!!!!!!!!!!!!!change to node-based, make it a vector
        !!!!!!!!!!!!!!!!!!!!!!!neighbor for each node, only consider in
        
    c : float vector
        Initial confidence radius. Imported from the graph, node-based vector
        
        
    mu : float
        Learning rate, also known as cautiousness parameter (homogeneous)
    delta : float
        Confidence shrinking parameter. This determines how much the confidence radius between 
        two nodes shrinks when they have a negative interaction - i.e. interaction where their
        opinion difference is greater than their confidence radius
    gamma : float
        Confidence expanding parameter. This determines how much the confidence radius between
        two nodes grows when they have a positive interaction - i.e. interaction where thier
        opinion difference is less than their confidence radius, so their opinions move towards each other
    random_seed: int, optional
        random seed for interacting edge selection at each timestep of the model
    t_start: int, default = 0
        What timestep to start counting the model at. By default t = 0 and the model is started at the "beginning."
        To continue a simulation, set t_start to the time left off and set confidence_start
    confidence_start: np.array, optional
        The starting confidence values for the simulation. This is a numpy array with of size equal to the number of 
        edges in the graph and the ith entry should correspond to the confidence value of the ith edge from the igraph
        order of edges in the edge list. confidence_start should be used to continue a simulation where left off
    tol : float, default = 0.02
        The tolerence/convergence critera. All opinion clusters must have difference between 
        maximum and minimum opinion less that tol for the simulation to be converged
    Tmax : int, default = 10**8
        Maximum number of timesteps for the simulation.
    Tmin : int, default = 0
        Minimum number of timesteps before we start checking convergence. Could be useful if redoing an experiement.
    T_acc : int, default = None
        Accuracy in convergence time.
        This is the number of timesteps we advance the model before checking for convergence again.
        The default value is to use an adaptive T_acc based on the sigfigs parameter
    sigfigs : int, default = 3
        The number of significant digits of accuracy for T. The accuracy in the number of decimal
        places for ln(T) is sigfigs - 1. This paramter controls the adaptive change of T_acc, and
        therefore the tradeoff between runtime and accuracy in T
    check_convergence : bool, default = True
        Specify if we should check for convergence using our stopping criteria of opinion clusters in the effective subgraph
        needing to have diameter < tol. If check_convergence is False, then the simuluation will run until Tmax.
        
    Other Parameters
    ----------------
    return_confidence_changes : bool, optional
        If True, returns a Pandas DataFrame documenting the confidence radius change at each timestep.
        The columns of the DataFrame are 't' (int : the time), 'node1' (int : first node in the interacting edge), 
        'node2' (int : second node in the interacting edge), 'confidence' (float : new confidence radius for the edge), 
        'increase' (bool : True if the confidence increased)
    return_opinion_series : bool, optional
        If True, returns matrix containing each node's opinion at each time
    series_timestep: int, default = 1
        If return_opinion_series is true, the opinion values will be added to the time series
        every series_timesteps steps
    return_animation_matrix : bool, optional
        If True returns an array of adjacency matrices for visualization
    matrix_timestep : int, default = 10
        If return_animation_matrix is True, a matrix will be generated every matrix_timestep steps
    nodeorder : int array, optional
        If return_animation_matrix is True, nodeorder, a list of values 0 to n-1 specifies the row/col
        order of returned visualization matrices
    
    Returns
    ----------------
    output : dictionary
        Dictionary of simulation outputs of {T: int, T_changed: int, T_acc: int, n_clusters: int, 
        clusters: list of lists, random_seed: int, final_opinions: list of float, total_change: list of float,
        n_updates: list of int, local_respectiveness: list of float, confidence: list of float}
        If return_opinion_series is True, also includes {opinion_series: numpy.array}
        If return_matrix is True, also includes {visualization_matrix: list}
        
        <T> int : number of timesteps to converge
        <T_changed> int : number of timesteps in which opinions changed
        <T_acc> int : accuracy of the timesteps, the convergence time will be wihtin T +/- T_acc
        <bailout> bool: True if bailout time was reached before final convergence.
                        Opinion cluster membership may still be reported if initial convergence check passed.
        <n_clusters> int : number of opinion clusters
        <clusters> list of list: cluster membership lists by int node ID
        <avg_opinoin_diff> float : the sum of the magnitudes of the opinion differences divided by the number of edges
        <random_seed> float : random seed used for random selection of nodes at each timestep  
        <final_opinions> list of float : the final opinions of each node
        <total_change> list of float: total absolute distance each node changed opinion
        <n_updates> list of int: total number of times each node updated its opinion
        <local_receptiveness> list of float: the fraction of neighbors each node is still willing to interact with
        <confidence> list of float: final confidence values for each edge. The edges are order numerically per G.es in igraph
        <opinion_series> numpy.array : n x T matrix of each node's opinion at each time
        <matrix_list> list of numpy.array: list of visualization matrices generated by get_visualization_matrix    
    """
    
    #Make a deep copy of the graph so we don't change the original
    G = graph.copy()
    n = G.vcount() #get number of nodes/verticies in the network

    ## Get the edge information for the graph
    # Store the start and end nodes for each edge (igraph will return edge_start < edge_end for each edge)
    # and the magnitude of the opinion difference and confidence radius on that edge as numpy arrays
    # Edge start and end
    edge_start = np.array([e.tuple[0] for e in G.es], dtype=np.int64)
    edge_end = np.array([e.tuple[1] for e in G.es], dtype=np.int64)
    n_edges = len(edge_start)

    # Magnitude of the opinion difference across each edge
    opinion_diff = np.array(G.vs[edge_start.tolist()]['opinion']) - np.array(G.vs[edge_end.tolist()]['opinion'])
    opinion_diff = np.abs(opinion_diff)

    
    
    # Store initial confidence bounds
    confidence_bounds = []
    for node in G.nodes:
        # Check if the 'feature_name' attribute exists for the node
        if 'conf_bound' in G.nodes[node]:
            confidence_bound = G.nodes[node]['conf_bound']
            confidence_bounds.append(confidence_bound)
        
        # Initialize the confidence on each edge as the provided initial value c
        if confidence_start is not None:
            confidence = confidence_start
        else:
            confidence = confidence_bound

            
    
    #If no T_acc was provided, we try and make it adaptive to the provided sig-figs
    #Otherwise, we use the provided T_acc value
    adaptive_T_acc = False
    if T_acc == None:
        adaptive_T_acc = True
        T_acc = 0 #initialize T_acc to 0, we want the exact number of time steps to start

    # Set random seed if given, otherwise generate, use and store a new seed
    if random_seed == None:
        random_seed = random.randrange(sys.maxsize)
    random.seed(a=random_seed)

    #If we are continuing from another simulation, get to the right time point drawing random numbers with this seed
    for i in range(t_start):
        number = random.uniform(0,1)
    
    # Create a dataframe to store the confidence changes if return_confidence_changes is True
    if return_confidence_changes:
        confidence_df = pd.DataFrame(columns = ['t', 'node1', 'node2', 'confidence', 'increase'])

    # Store initial opinions if return_opinion_series is True
    if return_opinion_series:
        opinion_series = G.vs['opinion']

    # Store visualization matrices if return_animation_matrix is True
    if return_animation_matrix:
        M = get_visualization_matrix(G, c, nodeorder = nodeorder)
        matrix_list = [M]

    # Initialize a list of total opinion change and number of update times for each node
    total_change = [0] * n
    n_updates = [0] * n
    
    #Initalize there being no opinion clusters
    clusters = []

    # Keep track of how many time steps actual resulted in an opinion change
    T_changed = 0 #This is the number of times the opinions changed and the confidence increased

    # Keep track of if we passed the initial convergence checks and final convergence criteria
    # First check = all nodes are <= tol or > d in opinion with their neighbors
    # Second check = nodes are in distinct opinion clusters separated by at least d
    # Final convergence = all clusters (already set in second check) have diameter < tol
    first_check_passed = False
    converged = False

    # Track if opinions changed at all and number of timesteps since last convergence check
    # We only check for convergence if it has been T_acc timesteps since the last check
    # and the opinions have changed at least once in those timesteps
    step_counter = 0
    opinions_changed = False
    
    ### RUN THE DEFFAUNT-WEISBUCH MODEL UNTIL TMAX AT MOST
    for t in range(t_start + 1, Tmax+1):
        
        # Update timestep counter for convergence checking
        step_counter = step_counter + 1

        ### RANDOMLY GET NEXT NODES TO INTERACT
        ## We do this by picking an edge uniformly at random 
        ## (this is different from picking a first node and then a second connected node)
        index = random.uniform(0,1) * n_edges
        index = math.floor(index)
        if index == n_edges: #Make sure we didn't happen to pick 1 when uniformly drawing
            index = n_edges - 1
        node1, node2 = edge_start[index], edge_end[index] #Get the nodes on this edge

        ## HAVE NODES 1 AND 2 INTERACT ACCORDING TO THE DEFFAUNT-WEISBUCH MECHANISM
        ## If they are within their confidence, then update opinions and increase the confidence
        ## If they are outside their confidence, then decrease the confidence further

        opinion1 = G.vs[node1]['initial_opinion']
        opinion2 = G.vs[node2]['initial_opinion']

        # If they are within their confidence, then update opinions and increase the confidence
        if ( abs(opinion1 - opinion2) < confidence[index] ):
            #Update the node's opinions
            G.vs[node1]['initial_opinion'] = opinion1 + mu*(opinion2 - opinion1)
            G.vs[node2]['initial_opinion'] = opinion2 + mu*(opinion1 - opinion2)

            #Update the confidence and increase it
            confidence[index] = (1-gamma) * confidence[index] + gamma

            #Store the change in confidence in the dataframe if we are saving it
            if return_confidence_changes:
                row = pd.DataFrame(columns = ['t', 'node1', 'node2', 'confidence', 'increase'])
                row.loc[0] = [t, node1, node2, confidence[index], True]
                confidence_df = confidence_df.append(row, ignore_index=True)

            #Update our variables for the timesteps with opinion changes, and total opinion change
            T_changed = T_changed + 1 #represents times opinion changed and confidence increased
            change = mu*abs(opinion2 - opinion1)
            total_change[node1] = total_change[node1] + change
            total_change[node2] = total_change[node2] + change
            n_updates[node1] = n_updates[node1] + 1
            n_updates[node2] = n_updates[node2] + 1
            opinions_changed = True

            #Update the edge values of the opinion differences with the two nodes
            index = np.argwhere((edge_start == node1) | (edge_start == node2) | (edge_end == node1) | (edge_end == node2))
            index = index.flatten()
            start_node, end_node = edge_start[index].tolist(), edge_end[index].tolist()
            opinion_diff[index] = np.abs( np.array(G.vs[start_node]['opinion']) - np.array(G.vs[end_node]['opinion']) )

        ## If they are outside their confidence, then decrease the confidence further
        else:
            confidence[index] = delta * confidence[index]

            #Store the change in confidence in the dataframe if we are saving it
            if return_confidence_changes:
                row = pd.DataFrame(columns = ['t', 'node1', 'node2', 'confidence', 'increase'])
                row.loc[0] = [t, node1, node2, confidence[index], False]
                confidence_df = confidence_df.append(row, ignore_index=True)

        # Store the opinions at the end of this time step if needed
        if return_opinion_series and (t % opinion_timestep == 0):
            opinion_series = np.vstack((opinion_series, np.asarray(G.vs['opinion'])))

        # Generate and store the visualization matrix at the end of this time step if needed
        if return_animation_matrix and (t % matrix_timestep == 0):
            M = get_visualization_matrix(G, c, nodeorder = nodeorder)
            matrix_list.append(M)

        ## CHECK CONVERGENCE
        if check_convergence:
            # Proceed to checking convergence it has been T_acc time steps since the last convergence check
            # AND opinions have changed at least once in those timesteps
            if (opinions_changed == True) and (step_counter >= T_acc) and (t > Tmin):

                # Adapt and increase T_acc by a factor of 10 if the timesteps has increased by a factor of 10
                if adaptive_T_acc:
                    digits = math.floor(math.log10(t))
                    T_acc = 10**(digits - sigfigs)

                # Reset timestep counter and opinion change tracking
                step_counter = 0
                opinions_changed = False

                ### FIRST CONVERGENCE CHECK
                ## Each node must have neighbors that are either < tol away, or >= their confidence away
                index = np.argwhere((tol <= opinion_diff) & (opinion_diff < confidence))
                index = index.flatten() 

                ### SECOND CONVERGENCE CHECK (only if first check passes)
                ## Nodes must be in distinct clusters, and each cluster must have diameter < tol
                if len(index) == 0:

                    converged = True

                    ## Construct the effective subgraph, which only consists of edges with
                    ## magnitude of the opinion difference less than d, i.e. the edges of possible interactions

                    #Get the edges which have opinion difference < d and convert them to an edge list
                    index = np.argwhere(opinion_diff < confidence)
                    index = index.flatten()
                    sub_edge_start = edge_start[index]
                    sub_edge_end = edge_end[index]
                    edge_list = [(sub_edge_start[i], sub_edge_end[i]) for i in range(len(index))]

                    #Construct an effective subgraph subG consisting of these edges with possible interactions 
                    subG = igraph.Graph()
                    subG.add_vertices(n)
                    subG.add_edges(edge_list)
                    subG.vs['opinion'] = G.vs['opinion']

                    #Get the opinion clusters, the components of the subgraph
                    cluster_membership = subG.clusters(mode='strong').membership
                    n_clusters = max(cluster_membership)+1
                    clusters = []
                    for i in range(n_clusters):
                        cluster = np.nonzero(np.array(cluster_membership)==i)[0]
                        clusters.append(cluster.tolist())

                    ## Check if the constructed clusters all have diameter < tol:
                    for cluster in clusters:
                        opinions = G.vs[cluster]['opinion']
                        diameter = max(opinions) - min(opinions)
                        if diameter >= tol:
                            converged = False
                            break

            ## If we're converged, stop running the model
            if converged:
                break

    ## Now that we are done running the DW simulation, return results

    T = t #get the convergence time
    final_opinions = G.vs['opinion'] #store the final opinions

    # If we hit Tmax and were not converged, specify that we hit the bailout time
    bailout = False
    if T == Tmax and (not converged):     
        bailout = True
        #T = T + 9 

    n_clusters = len(clusters) # calculate the number of clusters
    
#     #Calculate the local agreement for each node, this is the fraction of neighbors with
#     #opinion on the same side of the mean
#     G.vs['sign'] = np.sign(final_opinions - np.mean(final_opinions))
#     local_agreement = [( np.sum(G.vs[G.neighbors(node)]['sign'] == G.vs[node]['sign']) / G.degree(node) ) for node in range(n)]
    
    #Calculate the local receptiveness for each node - this is the fraction of neighbors that are < d away in opinion
    #If we bailed out before getting the cluster membership, then calculate the effective subgraph to get the local receptiveness
    if n_clusters == 0:
        #Get the edges which have opinion difference < d and convert them to an edge list
        index = np.argwhere(opinion_diff < confidence)
        index = index.flatten()
        sub_edge_start = edge_start[index]
        sub_edge_end = edge_end[index]
        edge_list = [(sub_edge_start[i], sub_edge_end[i]) for i in range(len(index))]

        #Construct an effective subgraph subG consisting of these edges with possible interactions 
        subG = igraph.Graph()
        subG.add_vertices(n)
        subG.add_edges(edge_list)
        
        #If we weren't checking convergence, then get the final clusters anyways
        if not check_convergence:
            subG.vs['opinion'] = G.vs['opinion']

            #Get the opinion clusters, the components of the subgraph
            cluster_membership = subG.clusters(mode='strong').membership
            n_clusters = max(cluster_membership)+1
            clusters = []
            for i in range(n_clusters):
                cluster = np.nonzero(np.array(cluster_membership)==i)[0]
                clusters.append(cluster.tolist())
                
            ## Check if the constructed clusters all have diameter < tol:
            for cluster in clusters:
                opinions = G.vs[cluster]['opinion']
                diameter = max(opinions) - min(opinions)
                if diameter >= tol:
                    n_clusters = 0
        
    local_receptiveness = [ (subG.degree(node) / G.degree(node)) for node in range(n) if G.degree(node) > 0 ]
    
    #Dictionary to store the output values to return
    outputs = {}
    
    #Graph level information
    outputs['T'] = T
    outputs['T_changed'] = T_changed
    outputs['T_acc'] = T_acc
    outputs['bailout'] = bailout
    outputs['n_clusters'] = n_clusters
    outputs['clusters'] = clusters
    outputs['random_seed'] = random_seed
    outputs['avg_opinion_diff'] = np.sum(opinion_diff) / len(opinion_diff)

    #Node level information - size n (# of nodes) vectors
    outputs['total_change'] = total_change
    outputs['n_updates'] = n_updates
    outputs['final_opinions'] = final_opinions
    outputs['local_receptiveness'] = local_receptiveness
    
    #Edge level information
    outputs['confidence'] = confidence
    
    if return_opinion_series == True:
        outputs['opinion_series'] = opinion_series
    if return_animation_matrix == True:
        outputs['matrix_list'] = matrix_list
    if return_confidence_changes == True:
        outputs['confidence_df'] = confidence_df
        
    return outputs

## -----------------------------------------------------------------------------------

#HK model with dynamic confidence implemented in igraph
def HK(graph, 
       delta, gamma,
       t_start = 0, confidence_start = None,
       tol = 0.02, Tmax = 10**6, Tmin = 0, T_acc = None, sigfigs = 3,
       check_convergence = True, return_opinion_series = False, opinion_timestep = 1, 
       return_animation_matrix = False, matrix_timestep = 100, nodeorder = None, patient_zero = False, initial_percentage_infected = 0.01):
    
    """
    Simulates the Hegselmann-Krause model of opinion dynamics with dynamic confidence bounds.
    Calculates an "effective subgraph" consisting of edges with opinion difference within the confidence
    in order to sort the nodes into clusters and check convergence of the model.
    
    Parameters
    ------------
    graph : igraph.graph
        igraph Graph with 'opinion' node-level attribute and n nodes
    c : float/now vector from the graph, dont need to input
        Initial confidence radius. We set this to be the homogeneous confidence radius on all edges initially
        
        !!!change to node-based, make it a vector
        neighbor for each node
        
    delta : float
        Confidence shrinking parameter. This determines how much the confidence radius between 
        two nodes shrinks when they have a negative interaction - i.e. interaction where their
        opinion difference is greater than their confidence radius
        default to 1
        
    gamma : float
        Confidence expanding parameter. This determines how much the confidence radius between
        two nodes grows when they have a positive interaction - i.e. interaction where thier
        opinion difference is less than their confidence radius, so their opinions move towards each other
        default to 0
        
    random_seed: int, optional
        random seed for interacting edge selection at each timestep of the model
    t_start: int, default = 0
        What timestep to start counting the model at. By default t = 0 and the model is started at the "beginning."
        To continue a simulation, set t_start to the time left off and set confidence_start
    confidence_start: np.array, optional
        The starting confidence values for the simulation. This is a numpy array with of size equal to the number of 
        edges in the graph and the ith entry should correspond to the confidence value of the ith edge from the igraph
        order of edges in the edge list. confidence_start should be used to continue a simulation where left off
    tol : float, default = 0.02
        The tolerence/convergence critera. All opinion clusters must have difference between 
        maximum and minimum opinion less that tol for the simulation to be converged
    Tmax : int, default = 10**6
        Maximum number of timesteps for the simulation.
    Tmin : int, default = 0
        Minimum number of timesteps before we start checking convergence. Could be useful if redoing an experiement.
    T_acc : int, default = None
        Accuracy in convergence time.
        This is the number of timesteps we advance the model before checking for convergence again.
        The default value is to use an adaptive T_acc based on the sigfigs parameter
    sigfigs : int, default = 3
        The number of significant digits of accuracy for T. The accuracy in the number of decimal
        places for ln(T) is sigfigs - 1. This paramter controls the adaptive change of T_acc, and
        therefore the tradeoff between runtime and accuracy in T
    check_convergence : bool, default = True
        Specify if we should check for convergence using our stopping criteria of opinion clusters in the effective subgraph
        needing to have diameter < tol. If check_convergence is False, then the simuluation will run until Tmax.
        
    Other Parameters
    ----------------
    return_opinion_series : bool, optional
        If True, returns matrix containing each node's opinion at each time
    series_timestep: int, default = 1
        If return_opinion_series is true, the opinion values will be added to the time series
        every series_timesteps steps
    return_animation_matrix : bool, optional
        If True returns an array of adjacency matrices for visualization
    matrix_timestep : int, default = 10
        If return_animation_matrix is True, a matrix will be generated every matrix_timestep steps
    nodeorder : int array, optional
        If return_animation_matrix is True, nodeorder, a list of values 0 to n-1 specifies the row/col
        order of returned visualization matrices
    patient_zero : bool, default = False
        determines whether we want to have a patient zero
    #initial_percentage_infected : float, default = 0.01
        #determines the percentage of nodes that are initially infected
    
    Returns
    ----------------
    output : dictionary
        Dictionary of simulation outputs of {T: int, T_acc: int, n_clusters: int, 
        clusters: list of lists, random_seed: int, final_opinions: list of float, total_change: list of float,
        local_respectiveness: list of float, confidence: list of float}
        If return_opinion_series is True, also includes {opinion_series: numpy.array}
        If return_matrix is True, also includes {visualization_matrix: list}
        
        <T> int : number of timesteps to converge
        <T_acc> int : accuracy of the timesteps, the convergence time will be wihtin T +/- T_acc
        <bailout> bool: True if bailout time was reached before final convergence.
                        Opinion cluster membership may still be reported if initial convergence check passed.
        <n_clusters> int : number of opinion clusters
        <clusters> list of list: cluster membership lists by int node ID
        <avg_opinoin_diff> float : the sum of the magnitudes of the opinion differences divided by the number of edges
        <random_seed> float : random seed used for random selection of nodes at each timestep  
        <final_opinions> list of float : the final opinions of each node
        <total_change> list of float: total absolute distance each node changed opinion
        <local_receptiveness> list of float: the fraction of neighbors each node is still willing to interact with
        <confidence> list of float: final confidence values for each edge. The edges are order numerically per G.es in igraph
        <opinion_series> numpy.array : n x T matrix of each node's opinion at each time
        <matrix_list> list of numpy.array: list of visualization matrices generated by get_visualization_matrix    
    """
    
    #Make a deep copy of the graph so we don't change the original
    G = graph.copy()
    n = G.vcount() #get number of nodes/verticies in the network

    ## Get the edge information for the graph
    # Store the start and end nodes for each edge (igraph will return edge_start < edge_end for each edge)
    # and the magnitude of the opinion difference and confidence radius on that edge as numpy arrays
    # Edge start and end
    edge_start = np.array([e.tuple[0] for e in G.es], dtype=np.int64)
    edge_end = np.array([e.tuple[1] for e in G.es], dtype=np.int64)
    n_edges = len(edge_start)
    
    # create n times 1 arrays
    #infect_time = -1*np.ones(n)
    #recover_time = -1*np.ones(n)
    
    # how many nodes to infect? 
    #if patient_zero:
        #k = 1
    #else:
        #k = math.ceil(n*initial_percentage_infected)
    #initial_infected_nodes = random.sample(range(n),k)
    
    #select nodes to be infected at time 0
    #infect_time[initial_infected_nodes]=0

    # Initialize the confidence on each edge based on the initial confidence bound of each node
    if confidence_start is not None:
        confidence = confidence_start
    else:
        #confidence = [G.nodes[node]['conf_bound'] for node in G.nodes()]
        confidence = [node['conf_bound'] for node in G.vs]

    #If no T_acc was provided, we try and make it adaptive to the provided sig-figs
    #Otherwise, we use the provided T_acc value
    adaptive_T_acc = False
    if T_acc == None:
        adaptive_T_acc = True
        T_acc = 0 #initialize T_acc to 0, we want the exact number of time steps to start

    # Store initial opinions if return_opinion_series is True
    if return_opinion_series:
        opinion_series = G.vs['opinion']

    # Store visualization matrices if return_animation_matrix is True
    if return_animation_matrix:
        M = get_visualization_matrix(G, c, nodeorder = nodeorder)
        matrix_list = [M]

    # Initialize a list of total opinion change
    total_change = [0] * n
    
    #Initalize there being no opinion clusters
    clusters = []

    # Keep track of if we converged
    converged = False
    
    # Track the number of timesteps since last convergence check
    # We only check for convergence if it has been T_acc timesteps since the last check
    step_counter = 0
    
    ### RUN THE HEGSELMANN-KRAUSE MODEL UNTIL TMAX AT MOST
    #We start at t_start = 0 by default since we check for convergence first at each time then do the opinion/confidence update
    for t in range(t_start, Tmax+1):
        
        # Adapt and increase T_acc by a factor of 10 if the timesteps has increased by a factor of 10
        if adaptive_T_acc and t > 0:
            digits = math.floor(math.log10(t))
            T_acc = 10**(digits - sigfigs)
        
        #Calculate the magnitude of the opinion difference across each edge
        #opinion_diff = np.array(G.vs[edge_start.tolist()]['opinion']) - np.array(G.vs[edge_end.tolist()]['opinion'])
        #opinion_diff = np.abs(opinion_diff)
        
        #Adjust opinion difference
        opinion_diff = np.zeros(len(edge_start))
        for i, (start, end) in enumerate(zip(edge_start, edge_end)):
            opinion_diff[i] = abs(G.vs[start]['opinion'] - G.vs[end]['opinion'])

        
        #Get the edges which have opinion difference < d, i.e. there is a positive interaction
        #pos_index = np.argwhere(opinion_diff < confidence)
        #pos_index = pos_index.flatten()
        
        pos_index = [i for i, diff in enumerate(opinion_diff) if diff < confidence[edge_start[i]]]
        neg_index = [i for i, diff in enumerate(opinion_diff) if diff >= confidence[edge_start[i]]]
        
        sub_edge_start = edge_start[pos_index]
        sub_edge_end = edge_end[pos_index]
        pos_edges = [(sub_edge_start[i], sub_edge_end[i]) for i in range(len(pos_index))]
        
        #Get the index of edges which have opinion difference >= d, i.e. there is a negative interaction
        #neg_index = np.argwhere(opinion_diff >= confidence)
        #neg_index = neg_index.flatten()
        
        #Construct an effective subgraph subG consisting of these edges with possible interactions 
        subG = igraph.Graph()
        subG.add_vertices(n)
        subG.add_edges(pos_edges)
        subG.vs['opinion'] = G.vs['opinion']
        
        #Check for convergence
        #TODO: add in convergence conditions for disease layer
        if check_convergence:
            # Proceed to checking convergence it has been T_acc time steps since the last convergence check and we're past the minimum check time
            if (step_counter >= T_acc) and (t > Tmin):
                
                step_counter = 0 #reset the time step counter
                converged = True
            
                #Get the opinion clusters, the components of the subgraph
                cluster_membership = subG.clusters(mode='strong').membership
                n_clusters = max(cluster_membership)+1
                clusters = []
                for i in range(n_clusters):
                    cluster = np.nonzero(np.array(cluster_membership)==i)[0]
                    clusters.append(cluster.tolist())

                ## Check if the constructed clusters all have diameter < tol:
                for cluster in clusters:
                    opinions = G.vs[cluster]['opinion']
                    diameter = max(opinions) - min(opinions)
                    if diameter >= tol:
                        converged = False
                        break
                        
        ## If we're converged, stop running the model
        if converged:
            break
        
        #Update the opinions of each node using the HK update mechanism
        #We will do this one node at a time
        
        original_opinions = np.array(G.vs['opinion']) #get all the current opinions to do updates with
        for node in range(n):
            old_opinion = original_opinions[node]
            
            #Get its neighbors in the effective subgraph. Include itself and calculate the opinions to average
            index = subG.neighbors(node) 
            index.append(node)
            opinions = original_opinions[index]
            
            new_opinion = sum(opinions) / len(opinions) #Get the new opinion of the node
            G.vs[node]['opinion'] = new_opinion #set the new opinion
            total_change[node] = total_change[node] + abs(old_opinion - new_opinion) #update the total opinion change for the node
            
            ##TODO: implement state changes here
        
        #Update the confidence values for all the edges
        #confidence[pos_index] = (1-gamma) * confidence[pos_index] + gamma #increases in confidence
        #confidence[neg_index] = delta * confidence[neg_index] #decreases in confidence

        for index in pos_index:
            start_node = edge_start[index]
            confidence[start_node] = (1 - gamma) * confidence[start_node] + gamma

        for index in neg_index:
            start_node = edge_start[index]
            confidence[start_node] = delta * confidence[start_node]

        step_counter += 1 #increase the step counter for how many time steps since we last check for convergence

    ## Now that we are done running the HK simulation, return results

    T = t #get the convergence time
    final_opinions = G.vs['opinion'] #store the final opinions

    # If we hit Tmax and were not converged, specify that we hit the bailout time
    # Add 9 to T just to distinguish that actual convergernce takes > Tmax time
    bailout = False
    if T == Tmax and (not converged):     
        bailout = True

    n_clusters = len(clusters) # calculate the number of clusters
    
    #Calculate the local receptiveness for each node - this is the fraction of neighbors that are < d away in opinion
    #If we bailed out before getting the cluster membership, then calculate the effective subgraph to get the local receptiveness
    if n_clusters == 0:
        #Get the edges which have opinion difference < d and convert them to an edge list
        index = np.argwhere(opinion_diff < confidence)
        index = index.flatten()
        sub_edge_start = edge_start[index]
        sub_edge_end = edge_end[index]
        edge_list = [(sub_edge_start[i], sub_edge_end[i]) for i in range(len(index))]

        #Construct an effective subgraph subG consisting of these edges with possible interactions 
        subG = igraph.Graph()
        subG.add_vertices(n)
        subG.add_edges(edge_list)
        
        #If we weren't checking convergence, then get the final clusters anyways
        if not check_convergence:
            subG.vs['opinion'] = G.vs['opinion']

            #Get the opinion clusters, the components of the subgraph
            cluster_membership = subG.clusters(mode='strong').membership
            n_clusters = max(cluster_membership)+1
            clusters = []
            for i in range(n_clusters):
                cluster = np.nonzero(np.array(cluster_membership)==i)[0]
                clusters.append(cluster.tolist())
                
            ## Check if the constructed clusters all have diameter < tol:
            for cluster in clusters:
                opinions = G.vs[cluster]['opinion']
                diameter = max(opinions) - min(opinions)
                if diameter >= tol:
                    n_clusters = 0
        
    local_receptiveness = [ (subG.degree(node) / G.degree(node)) for node in range(n) if G.degree(node) > 0 ]
    
    #Dictionary to store the output values to return
    outputs = {}
    
    #Graph level information
    outputs['T'] = T
    outputs['T_acc'] = T_acc
    outputs['bailout'] = bailout
    outputs['n_clusters'] = n_clusters
    outputs['clusters'] = clusters
    outputs['avg_opinion_diff'] = np.sum(opinion_diff) / len(opinion_diff)

    #Node level information - size n (# of nodes) vectors
    outputs['total_change'] = total_change
    outputs['final_opinions'] = final_opinions
    outputs['local_receptiveness'] = local_receptiveness
    
    #Edge level information
    outputs['confidence'] = confidence
    
    if return_opinion_series == True:
        outputs['opinion_series'] = opinion_series
    if return_animation_matrix == True:
        outputs['matrix_list'] = matrix_list
        
    return outputs


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

#Function to plot opinion_series from DW simulation
def opinion_plot(G, opinion_series, uniform = False, fig=None, ax=None, savefile = None, title = None, fontsizes = {'small': 10, 'medium': 10, 'large':12}):
    
    """
    Function to plot the opinion trajectories after DW simulation.
    
    Parameters
    ----------
    G : igraph.graph
        igraph object with 'opinion' and 'weight' node-level attributes and n nodes
    opinion_series : np.array
        Time series output of DW function. T x n matrix where T is number of timesteps and n is number of nodes
        ij-th entry is the opinion at time i of node j
    uniform : bool, default = False
        Boolean value indicating if DW simulation weights are uniform or not. If not uniform, then trajectory color for each node
        will be correlated with the weight/activation probability. If uniform, then rainbow colors will be used
    savefile : string, optional
        File name to save plot
    
    Other Parameters
    ----------------
    fig, ax: matplotlib fig and ax, optional
        If provided (i.e. subplots of a larger plot), then plot will be generated on them
    title : string, optional
        Title for figure
    fontsizes : dictionary
        Dictionary with keys 'small', 'medium', 'large' of fontsizes for plot
    
    Returns
    ----------------
    Shows (and saves if savefile is specified) plot of opinion_series opinion trajectories of each node
        
    """
    n = G.vcount() #get number of nodes/verticies in the network
    
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots()
    
    #Calculate activation probability of each node as the first of an interaction pair
    weights = G.vs['weight']
    weights = np.array(weights)
    p_activate = weights / sum(weights)
    
    # Set node colors based on weight magnitude
    colormap = mpl.cm.get_cmap('rainbow', 100)
    norm = mpl.colors.Normalize(vmin=0, vmax = round_decimals_up(np.amax(p_activate), decimals=2) )

    # Plot the colored opinion_series, use rainbow/spaced colors if uniform weights
    if uniform: 
        for node in range(0,n):
            ax.plot(range(0, opinion_series.shape[0]), opinion_series[:,node])
    # Plot the colored opinion_series, use color based on weights if nonuniform weights
    else:
        for node in np.argsort(p_activate):
            color = norm(p_activate[node])
            ax.plot(range(0, opinion_series.shape[0]), opinion_series[:,node], color = colormap(color))

    ax.set_xlabel("t", fontsize=fontsizes['medium'])
    ax.set_ylabel("opinion", fontsize=fontsizes['medium'])
    ax.tick_params(axis='both', which='major', labelsize=fontsizes['small'])
    if title:
        ax.set_title(title, fontsize=fontsizes['large'])
    
    # Add color bar if weights are not uniform
    if not uniform: 
        #psm = ax.pcolormesh(data, cmap=colormap, rasterized=True, vmin=-4, vmax=4)
        norm = mpl.colors.Normalize(vmin=0, vmax = round_decimals_up(np.amax(p_activate), decimals=2) )
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
        cbar.ax.tick_params(labelsize=fontsizes['small']) 
    
    #save file if specified
    if savefile:
        plt.savefig(savefile, bbox_inches='tight', facecolor='white', transparent = False)




# Function to generate visualization matrix of effective subgraph given a graph with opinions
def get_visualization_matrix(graph, d, nodeorder = None):
    """Outputs visualization matrix for an igraph structure with opinion as a node-attribute
    
    Parameters
    ------------
    graph : igraph.Graph
        igraph Graph with 'opinion' node-level attribute and n nodes
    d : float
        Confidence radius. Nodes with absolute opinion difference > d are not connected
        in the effective subgraph
    nodeorder : int array, optional
        Array of the ordering of nodes in generated matrix. Default: nodes are ordered 0 to n-1
    
    Returns
    -------
    M : numpy.ndarray
        numpy matrix for visualization of size n x n where nodes are renumbered by nodeorder.
        The diagonal values are the opinions of nodes, so M_ii = opinion(i).
        The off diagonal values are the difference in opinion from the diagonal node, 
        so M_ij = opinion(j) - opinion(i) if i != j
    """
    
    G = graph #copy the graph
    
    n = G.vcount() #get number of nodes/verticies in the network
    
    #If no nodeorder is provided, order from 0 to n-1
    if nodeorder is None:
        nodeorder = list(range(n))
        
    #Add order as a node attribute in the graph
    #G[i]['index'] is the index of node i in nodeorder
    G.vs['index'] = np.argsort(np.array(nodeorder))
        
    # Initialize effective subgraph with reordered nodes
    sub_G = igraph.Graph()
    sub_G.add_vertices(n)
    sub_G.vs['opinion'] = G.vs[nodeorder]['opinion']
    
    # Add edges with "diff" attribute for difference of opinion
    # Only include edges with weight < d, the confidence radius
    for index in range(0,n):
        opinion = sub_G.vs[index]['opinion']
        
        # Get the corresponding not reordered node label in G
        node = nodeorder[index]
        
        # Get the neighborhood in the original/full graph structure
        nbhd = G.neighbors(node)
        
        #Convert the neighborhood to the reordered indices
        nbhd_idx = np.asarray(G.vs[nbhd]['index'])
        
        # For each index, we only set relevent difference in opinion for nodes with larger index
        # That way we don't repeat differences we've already checked
        nbhd_idx = nbhd_idx[np.where(nbhd_idx > index)]
        
        #For each of the remaining neighbors, see if the opinion difference is within d
        for neighbor in nbhd_idx:
            diff = abs(sub_G.vs[neighbor]['opinion'] - opinion)
            if diff < d:
                #if diff = 0, then assign it machine epsilon so it doesn't get masked when plotting
                if diff == 0:
                    diff = np.finfo(float).eps
                sub_G.add_edges([(index, neighbor)], {'diff': diff})
    
    # Get upper triangle of weighted adjacency matrix from the effective subgraph
    #M = sub_G.get_adjacency(type = 'GET_ADJACENCY_UPPER', attribute = 'diff')
    M = sub_G.get_adjacency(attribute = 'diff')
    M = np.asarray(M.data)
    M = np.triu(M, k=1)
    
    # Add in the diagonal as the opinions of the nodes
    diag = np.diag(sub_G.vs['opinion'])
    
    # Add in the lower triangular part, M^T = M (symmetric)
    M = M + np.transpose(M) + diag
    return M

def plot_matrix(matrix, d, savefile = None,
                row_labels = False, col_labels = False, tickspace = 1,
                show_values = False, decimal_places = 2, 
                figsize = (12,10), title = None, fontsizes = {'small': 10, 'medium': 12, 'large':20}):
    
    """
    Plots heatmap of visualization matrix of effective subgraph
    
    Parameters
    ------------
    matrix : numpy.ndarray
        Adjacency matrix of effective subgraph to visualize. Output of get_visualization_matrix.
    d : float
        Confidence radius, which sets the max value of colormap for off diagonal entries
    savefile : string, optional
        File name to save plot
    
    Other Parameters
    ----------------
    row_labels, col_labels : array_like, optional
        Row labels and column labels
    tickspace : int, default = 1
        The spacing/frequency of x and y axis row/col labels if using
    show_values : bool or string, default = False
        If True, print values of each nonzero matrix value in the heatmap
        If 'diagonal' then only print values on the diagonal
    decimal_places : int, default = 2
        Number of decimal places to print for matrix values if show_values = True
        Also the number of decimal places for the colorbar of off-diagonal entries
    figsize : tuple, default = (15,10)
        Tuple with two values giving figure dimensions in inches
    title : string, optional
        Title for plot
    fontsizes : dictionary
        Dictionary with keys 'small', 'medium', 'large' of fontsizes for plot
    
    Returns
    -------
    Shows (and saves if savefile is specified) plot of visualization matrix
    
    """
    
    # Create new plot
    fig, ax = plt.subplots(figsize = figsize, facecolor='white')
    
    # Define colormaps
    offdiag_cmap = OFFDIAG_CMAP
    diag_cmap = DIAG_CMAP
    
    # Don't plot zero values and generate mask to avoid them
    zero_mask = np.zeros_like(matrix)
    zero_mask[np.where(matrix == 0)] = True
    
    #Create two separate masks to plot the diagonal and off diagonal values differently
    diag = np.eye(*matrix.shape, dtype=bool)
    mask = zero_mask + diag
    mask[np.where(mask != 0)] = True
    
    #Set tick label spacing
    xticklabels, yticklabels = tickspace, tickspace
    
    # Create pandas dataframe with row/col labels if available
    if row_labels and col_labels:
        df = pd.DataFrame(matrix, index = row_labels, columns = col_labels)
    elif row_labels:
        df = pd.DataFrame(matrix, index = row_labels)
        yticklabels=False
    elif col_labels:
        df = pd.DataFrame(matrix, columns = col_labels)
        xticklabels=False
    else:
        df = pd.DataFrame(matrix)
        xticklabels, yticklabels = False, False
    
    #Plot heatmap
    if show_values:
        if show_values == True:
            #Plot off diagonal differences with values
            sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                        xticklabels=xticklabels, yticklabels=yticklabels,
                        linewidths=.25, square=True, cbar_kws = dict(pad=-0.03),
                        annot=True, fmt="0."+str(decimal_places)+"f", annot_kws={"size":fontsizes['small']})
        else:
            #Plot off diagonal differences without values
            sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                        xticklabels=xticklabels, yticklabels=yticklabels,
                        linewidths=.25, square=True, cbar_kws = dict(pad=-0.03)) 
        #Plot diagonal
        sns.heatmap(df, mask = ~diag, cmap = diag_cmap, vmin = 0, vmax = 1, 
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    linewidths=.25, square=True, 
                    annot=True, fmt="0."+str(decimal_places)+"f", annot_kws={"size":fontsizes['small']})
    else:
        #Plot off diagonal differences
        sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                    linewidths=.25, xticklabels=xticklabels, yticklabels=yticklabels,
                    ax=ax, square=True, cbar_kws = dict(pad=-0.03))
        #Plot diagonal
        sns.heatmap(df, mask = ~diag, cmap = diag_cmap, vmin = 0, vmax = 1, 
                    linewidths=.25, xticklabels=xticklabels, yticklabels=yticklabels,
                    ax=ax, square=True)
    
    # Label colorbars and set axis font size
    cbar_offdiag = ax.collections[0].colorbar
    cbar_offdiag.ax.set_title('diff')
    cbar_offdiag.ax.tick_params(labelsize=fontsizes['small'])
    
    cbar_diag = ax.collections[1].colorbar
    cbar_diag.ax.tick_params(labelsize=fontsizes['small'])
    cbar_diag.ax.locator_params(nbins=11)
    cbar_diag.ax.set_title('opinion')
    
    # Draw frame for heatmap
    linewidth = 5
    ax.axvline(x=0, color='k',linewidth=linewidth)
    ax.axvline(x=matrix.shape[0], color='k',linewidth=linewidth)
    ax.axhline(y=0, color='k',linewidth=linewidth)
    ax.axhline(y=matrix.shape[1], color='k',linewidth=linewidth)
    
    #ax.set_ylabel("Confidence Radius (d)", fontsize = fontsizes['medium'])
    #ax.set_xlabel("Learning rate (mu)", fontsize = fontsizes['medium'])
    ax.set_title(title, fontsize = fontsizes['large'], loc = 'left')
    
    #save file if specified
    if savefile:
        plt.savefig(savefile, bbox_inches='tight', facecolor='white')
    plt.show()
    return

def animate_matrix(matrix_list, d, matrix_timestep, T,
                   fps = 2, repeat = False,
                   savefile = False, figsize = (12,10),
                   row_labels = False, col_labels = False, tickspace = 1,
                   show_values = False, decimal_places = 2, 
                   fontsizes = {'small': 10, 'medium': 12, 'large':20}):
    
    """
    Animated heatmaps of visualization matrix of effective subgraph until DW model converges
    
    Parameters
    ------------
    matrix_list : list of numpy.ndarray
        List of adjacency matrices of effective subgraph to visualize/animate
        i.e. list of output['visualization_matrix'] of DW when return_animation_matrix = True
    d : float
        Confidence radius, which sets the max value of colormap for off diagonal entries
    matrix_timestep : int
        The timestep between generated visualization matrices.
        This properly sets the titles of the animation matrices.
    T : int
        Final timestep of simulation
        This properly sets the titles of the animation matrices
    fps : int, default = 2
        Frames per second for the animation
    repeat: bool, default = False
        Whether to repeat animation when played in IDE (i.e. in jupyter lab widgets)
    savefile : string, optional
        File name to save the animation to if present. Should be a .mp4 extension
    
    Other Parameters
    ----------------
    figsize : tuple, default = (12,10)
        Tuple with two values giving figure dimensions in inches
    row_labels, col_labels : array_like, optional
        Row labels and column labels for heatmap
    tickspace : int, default = 1
        The spacing/frequency of x and y axis row/col labels if using
    show_values : bool or string, default = False
        If True, print values of each nonzero matrix value in the heatmap
        If 'diagonal' then only print values on the diagonal
    decimal_places : int, default = 2
        Number of decimal places to print for matrix values if show_values = True
    fontsizes : dictionary
        Dictionary with keys 'small', 'medium', 'large' of fontsizes for plot
    
    Returns
    -------
    Shows (and saves if savefile is specified) animation of matrix_list
    """
    
    # Get number of frames = number of matrices
    n_frames = len(matrix_list)
    
    # Create new plot
    fig, ax = plt.subplots(figsize = figsize, facecolor = 'white')
    
    #Set tick label spacing
    xticklabels, yticklabels = tickspace, tickspace
    
    # Define colormaps
    offdiag_cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", ['0.1', '0.99'])
    diag_cmap = 'seismic'
    
    # Calculate millisecond delay interval from fps
    interval = 1000/fps
    
    def generate_plot(matrix, colorbar, title = False):
        """
        Generate seaborn heatmap plot of matrix with boolean colorbar for whether to generate/display a new colorbar
        """
        
        # Create pandas dataframe with row/col labels if available
        if row_labels and col_labels:
            df = pd.DataFrame(matrix, index = row_labels, columns = col_labels)
        elif row_labels:
            df = pd.DataFrame(matrix, index = row_labels)
            yticklabels=False
        elif col_labels:
            df = pd.DataFrame(matrix, columns = col_labels)
            xticklabels=False
        else:
            df = pd.DataFrame(matrix)
            xticklabels, yticklabels = False, False

        # Don't plot zero values and generate mask to avoid them
        zero_mask = np.zeros_like(matrix)
        zero_mask[np.where(matrix == 0)] = True

        #Create two separate masks to plot the diagonal and off diagonal values differently
        diag = np.eye(*matrix.shape, dtype=bool)
        mask = zero_mask + diag
        mask[np.where(mask != 0)] = True

        #Plot heatmap
        if show_values:
            if show_values == True:
                #Plot off diagonal differences with values
                sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                            xticklabels=xticklabels, yticklabels=yticklabels,
                            linewidths=.25, square=True, cbar_kws = dict(pad=-0.03),
                            annot=True, fmt="0."+str(decimal_places)+"f", annot_kws={"size":fontsizes['small']},
                           cbar = colorbar)
            else:
                #Plot off diagonal differences without values
                sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                            xticklabels=xticklabels, yticklabels=yticklabels,
                            linewidths=.25, square=True, cbar_kws = dict(pad=-0.03),
                           cbar = colorbar) 
            #Plot diagonal
            sns.heatmap(df, mask = ~diag, cmap = diag_cmap, vmin = 0, vmax = 1, 
                        xticklabels=xticklabels, yticklabels=yticklabels,
                        linewidths=.25, square=True, 
                        annot=True, fmt="0."+str(decimal_places)+"f", annot_kws={"size":fontsizes['small']},
                        cbar = colorbar)
        else:
            #Plot off diagonal differences
            sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                        linewidths=.25, xticklabels=xticklabels, yticklabels=yticklabels,
                        ax=ax, square=True, 
                        cbar = colorbar, cbar_kws = dict(pad=-0.05))
            #Plot diagonal
            sns.heatmap(df, mask = ~diag, cmap = diag_cmap, vmin = 0, vmax = 1, 
                        linewidths=.25, xticklabels=xticklabels, yticklabels=yticklabels,
                        ax=ax, square=True,
                        cbar = colorbar)
        
        # Label colorbars and set axis font size if we have a colorbar
        if colorbar:
            cbar_diag = ax.collections[0].colorbar
            cbar_diag.ax.set_title('diff')
            cbar_diag.ax.tick_params(labelsize=fontsizes['small'])
            cbar_offdiag = ax.collections[1].colorbar
            cbar_offdiag.ax.tick_params(labelsize=fontsizes['small'])
            cbar_offdiag.ax.set_title('opinion')
            #ax.tick_params(axis='both', labelsize=fontsizes['small'])

        # Draw frame for heatmap
        linewidth = 5
        ax.axvline(x=0, color='k',linewidth=linewidth)
        ax.axvline(x=matrix.shape[0], color='k',linewidth=linewidth)
        ax.axhline(y=0, color='k',linewidth=linewidth)
        ax.axhline(y=matrix.shape[1], color='k',linewidth=linewidth)
    
        if title:
            ax.set_title(title, fontsize = fontsizes['large'], loc = 'left')
    
    def init():
        """
        Initialize animation with initial opinion matrix and generate colorbar
        """
        plt.cla()
        matrix = matrix_list[0]
        title = 't = 0'
        generate_plot(matrix, colorbar = True, title = title)
        # return ax
        
    def animate(i):
        """
        Define animation instructions to plot the next matrix for the ith frame
        """
        plt.cla()
        #select the matrix
        matrix = matrix_list[i]
        
        # Calculate t for the title
        if i == len(matrix_list)-1:
            t = T
        else:
            t = i*matrix_timestep 
        title = 't = ' + str(t)
        
        generate_plot(matrix, colorbar = False, title = title)
        # return ax
    
    # Generate and return animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = n_frames, interval = interval, repeat = repeat, cache_frame_data = False, blit=False)
    
    if savefile:
        print('saving file')
        videowriter = animation.FFMpegWriter(fps=fps) 
        anim.save(savefile, writer=videowriter)
        print('done saving file')
    
    return anim

def plot_heatmap(matrix, cs, mus, vmin = None, vmax = None, savefile = None, 
                 show_values = False, labels = None, decimal_places = 0, 
                 tickspace = 1, title = None, figsize = (10,10), 
                 fontsizes = {'small': 12, 'medium': 14, 'large':16}):
    """
    Function to plot head map of DW simulation results from varying confidence radius d and learning rate mu 
    
    Parameters
    ------------
    matrix : numpy.ndarray
        Matrix of heatmap values where entry ij represents the value obtained
        from the ith confidence radius (d) and jth learning rate (mu)
    cs : list of float
        List of initial confidence radii, labels the rows of the matrix
        Note that labels go from top row to bottom row, 
        so the list ds should be in descending order
    mus : list of float
        List of learning rates, labels the columnss of the matrix
        Note that labels go from left to right, 
        so the list mus should be in ascending order
    vmin : float, optional
        Sets minimum value for colorbar and coloring scheme if provided. 
    vmax : float, optional
        Sets maximum value for colorbar and coloring scheme if provided. 
    savefile : string, optional
        File name to save the animation to if present. Should be a .mp4 extension
    
    Other Parameters
    ----------------
    show_values : bool, default = False
        If True, show values of each matrix entry
    labels : numpy.ndarray, optional
        Array of the same size as matrix containing custom labels to display
        This over-rides anything set by decimal_places and ignores show_values
    decimal_places : int, default = 0
        Number of decimal places to print for matrix values if show_values = True
        and labels = False
    tickspace : int, default = 1
        The spacing/frequency of x and y axis row/col labels if using
    title : string, optional
        Title to display on plot
    figsize : tuple, default = (10,10)
        Tuple with two values giving figure dimensions (width, height) in inches
    fontsizes : dictionary
        Dictionary with keys 'small', 'medium', 'large' of fontsizes for plot
    
    Returns
    -------
    Shows (and saves if savefile is specified) the plotted heatmap
    """

    #Create plot
    fig, ax = plt.subplots(figsize = figsize)
    
    #Set tick label spacing
    xticklabels, yticklabels = tickspace, tickspace
    
    #Set min and max for colorbar
    # Keep using vmin if given, otherwise take it from matrix values
    if not vmin:
        # If we have -inf values, i.e. if a value is ln(0), then set to 0 so it will plot
        if (np.amin(matrix) == - np.inf):
            matrix[matrix == - np.inf] = 0
            vmin = 0
        else:
            vmin = round_decimals_down(np.nanmin(matrix), decimal_places)
    
    # Keep using vmax if given, otherwise take it from matrix values
    if not vmax:
        vmax = round_decimals_up(np.nanmax(matrix), decimal_places)
    
    #Convert matrix to dataframe for axis labels
    df = pd.DataFrame(matrix, index = cs, columns = mus)
    
    #Plot heat map with or without entry values depending on show_values
    if isinstance(labels, np.ndarray) or isinstance(labels, list):
        ax = sns.heatmap(df, ax = ax, cmap = "rainbow", 
                    vmin = vmin, vmax = vmax, 
                    annot = labels, fmt = "", annot_kws={"size":fontsizes['small']})
    
    elif show_values:
        ax = sns.heatmap(df, cmap = "rainbow", vmin = vmin, vmax = vmax, annot=True,
                         fmt="0."+str(decimal_places)+"f", annot_kws={"size":fontsizes['small']})
    
    else:
        ax = sns.heatmap(df, cmap = "rainbow", vmin = vmin, vmax = vmax)
    
    #Set colorbar and tick label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsizes['small'])
    ax.tick_params(axis='both', labelsize=fontsizes['small'])
    
    #Set axis labels, title and fontsize
    ax.set_ylabel("Initial Confidence Radius (c)", fontsize = fontsizes['medium'])
    ax.set_xlabel("Learning Rate (mu)", fontsize = fontsizes['medium'])
    ax.set_title(title, fontsize = fontsizes['large'])
    
    #save file if specified
    if savefile:
        plt.savefig(savefile, bbox_inches='tight', facecolor='white')
    plt.show()
    return
    


## -----------------------------------------------------------------------------

#Function to round up
def round_decimals_up(number:float, decimals:int=1):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

#Function to round down
def round_decimals_down(number:float, decimals:int=1):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor


#Define function to print average +/1 standard deviation with proper precision
def print_avg_with_std(avg, std):
    #If we have a numpy array, we want to return an array again, but need to fill in entries elementwise
    if isinstance(avg, np.ndarray):
        
        # if avg.shape != std.shape:
        #     raise ValueError('Avg and Std numpy arrays must be of the same shape')
        
        original_shape = avg.shape #save the original array shape
        
        #Flatten everything for convinient indexing, we will reshape at the end
        avg = avg.reshape((avg.size))
        std = std.reshape((std.size))
        results = np.empty(avg.shape, dtype=object)
        
        #Get the number of decimal places to round to
        decimals = np.where(std > 1e-15, int(0) - np.floor(np.log10(std)).astype(int), int(1))
        # decimals = int(1) - np.floor(np.log10(std)).astype(int)
        
        for i in range(avg.size):
            if decimals[i] <= 3:
                truncated_std = round(std[i], decimals[i])
                truncated_avg = round(avg[i], decimals[i])

                # results[i] = str(truncated_avg) + u"\u00B1" + str(truncated_std)
                try:
                    results[i] = ( "{:.{decimals}f}".format(truncated_avg, decimals = decimals[i])
                                   + u"\u00B1" + "{:.{decimals}f}".format(truncated_std, decimals = decimals[i]) )
                except:
                    results[i] = ( "{:.0f}".format(truncated_avg)
                                   + u"\u00B1" + "{:.0f}".format(truncated_std) )
            else:
                avg_exponent = np.floor(np.log10(abs(avg[i]))).astype(int)
                avg_significand = avg[i] * 10**(-avg_exponent)
                std_exponent = np.floor(np.log10(abs(std[i]))).astype(int)
                std_significand = std[i] * 10**(-std_exponent)
                
                results[i] = ("{:.1f}".format(avg_significand) + "e{:g}".format(avg_exponent) + u"\u00B1"
                              "{:.1f}".format(std_significand) + "e{:g}".format(std_exponent) )
                
                # results[i] = ("{:.1f}".format(avg_significand) + r"$\times 10^{{{:g}}}$".format(avg_exponent) + u"\u00B1"
                #               "{:.1f}".format(std_significand) + r"$\times 10^{{{:g}}}$".format(std_exponent) )
                
                #results[i] = ( "{:.1e}".format(avg[i]) + u"\u00B1" + "{:.1e}".format(std[i]))
                
        results = np.array(results).reshape(original_shape)
        return results
    
    # If we just have scalar values, just return the one print out
    else:
        decimals = int(0) - int(math.floor(math.log10(std)))
        
        if decimals[i] <= 3:
            truncated_std = round(std, decimals)
            truncated_avg = round(avg, decimals)

            return ( "{:.{decimals}f}".format(truncated_avg, decimals = decimals)
                               + u"\u00B1" + "{:.{decimals}f}".format(truncated_std, decimals = decimals) )
        else:
            return "{:.1e}".format(avg) + u"\u00B1" + "{:.1e}".format(std)
