"""
Created on 11/3/2021
Created by Grace Li

"""

# Import required packages
import numpy as np
import pandas as pd
from scipy import io
import sys
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import time as time
import igraph as igraph
import os.path
from os import getpid
import multiprocessing

# Make process "nicer" and lower priority
import psutil
psutil.Process().nice(1)# if on *ux

# Import our own DW module
import sys
sys.path.append('..') #look one directory above
import DynamicBC as dbc

# Class for running sets of DW experiments            
class HK_experiment:
    
    #Set class parameters
    tol = 1e-10 #1e-6 #Diameter required for convergence critera of opinion clusters
    Tmax = 10**6 #Bailout time for ending the simulation
    
    # Initialize class with graph_type and number of nodes n
    def __init__(self, graph_type, n, p=False):
        '''
        Initializes class to run Deffaunt-Weisbuch simulations for a particular graph type
        
        Parameters
        ----------
        graph_type : string
            String specifying the graph type. Currently the options are "complete" and "erdos-renyi"
        n : int
            Number of nodes in the graph(s) considered
        p : float, required only if graph_type == "erdos-renyi"
            If the graph_type is "erdos-renyi", then p is a required parameter. p is the edge probability
            in the G(n,p) Erdos-Renyi model.
        '''
        
        self.graph_type = graph_type
        self.n = n
        
        self.foldername = graph_type + str(n) #savefolder name for experiment
        
        #Check that a directory for this experiment exists, and if not, create it
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
            os.makedirs(self.foldername + '/matfiles')
            os.makedirs(self.foldername + '/txtfiles')
        
        if self.graph_type == "erdos-renyi":
            self.p = p

        
    def generate_seed_files(self):
        '''
        Generate and save random seed files for random graphs (if not complete), and initial opinions if they don't exist yet
        '''
        
        self.graph_seed_file = self.graph_type + str(self.n) + "/graph_seeds.csv"
        self.opinion_seed_file = self.graph_type + str(self.n) + "/opinion_seeds.csv"

        if self.graph_type == "complete":
                
            #There is only one opinion seed for a complete graph, so we generate and save it if it doesn't exist yet
            if not os.path.exists(self.opinion_seed_file):
                df = pd.DataFrame(columns = ['opinion_seed'])
                random.seed(a=None) #reset random by seeding it with the current time
                weight_seed = str(random.randrange(sys.maxsize))
                df.loc[0] = [weight_seed]
                df.to_csv(self.opinion_seed_file, index=False, header=True)
         
        elif self.graph_type == "erdos-renyi":
            
            #There is only graph seed per p value for an erdos-renyi graph, so we generate and save it if it doesn't exist yet ????
            if not os.path.exists(self.graph_seed_file):
                df = pd.DataFrame(columns = ['p', 'graph_seed'])
                df.to_csv(self.graph_seed_file, index=False, header=True)
            df = pd.read_csv(self.graph_seed_file)
            row = df[df['p'] == self.p]
            if len(row) == 0:
                random.seed(a=None) #reset random by seeding it with the current time
                graph_seed = str(random.randrange(sys.maxsize))
                row = pd.DataFrame(columns = ['p', 'graph_seed'])
                row.loc[0] = [self.p, graph_seed]
                df = df.append(row, ignore_index=True)
                df.to_csv(self.graph_seed_file, index=False, header=True)
            
            if not os.path.exists(self.opinion_seed_file):
                df = pd.DataFrame(columns = ['p', 'graph', 'opinion_seed'])
                df.to_csv(self.opinion_seed_file, index=False, header=True)
            
        return
    
    ## Function to Run DW model for this graph and weight/opinion seeds  
    def run_HK(self, params):
    
        '''
        Runs DW experiment and saves appropriate output files
        Takes in a dictionary params, containing
        "c" - the initial confidence radius,
        "delta" - the confidence shrinker parameter??????, "gamma" - the confidence expander parameter
        and "opinion_set" - an integer representing which opinion set to generate from 
        the random opinion seed to run the DW model on. 
        For a complete graph, we only need these parameters.
        For Erdos-Renyi graphs, the 5th parameter, "graph_number" needs to be specified,
        and it represents which randomly generated graph to consider.??????
        '''
        
        print('Process Number ', getpid())
        print('Params', params)

        ## Initial set up
        #Unpack parameters
        c = params["c"]
        delta, gamma = params["delta"], params["gamma"]
        opinion_set = params["opinion_set"]
        if self.graph_type == 'erdos-renyi':
            graph_number = params["graph_number"]

        ## Read the random seeds if they exist, and generate and store them if they don't exist yet
        lock.acquire()
        
        #Make sure the appropriate save folders for this delta-gamma combo exist, and if not, create them
        folder = '/delta' + str(delta) + '-gamma' + str(gamma)
        if not os.path.exists(self.foldername + "/matfiles" + folder):
            os.makedirs(self.foldername + '/matfiles' + folder)
        if not os.path.exists(self.foldername + "/txtfiles" + folder):
            os.makedirs(self.foldername + '/txtfiles' + folder)
        
        # Get the random graph seed if not a complete graph
        if self.graph_type == 'erdos-renyi':
            df = pd.read_csv(self.graph_seed_file)
            row = df[df['p'] == self.p]
            graph_seed = row['graph_seed'].values[0]
            graph_seed = int(graph_seed)
                
        # Get the random opinion set seed 
        df = pd.read_csv(self.opinion_seed_file)
        if self.graph_type == 'complete':
            opinion_seed = df['opinion_seed'].values[0]
            opinion_seed = int(opinion_seed)
        elif self.graph_type == 'erdos-renyi':
            row = df[df['p'] == self.p]
            row = row[row['graph'] == graph_number]
            if len(row) == 0:
                random.seed(a=None) #reset random by seeding it with the current time
                opinion_seed = random.randrange(sys.maxsize)
                row = pd.DataFrame(columns = ['p', 'graph', 'opinion_seed'])
                row.loc[0] = [self.p, graph_number, str(opinion_seed)]
                df = df.append(row, ignore_index=True)
                df.to_csv(self.opinion_seed_file, index=False, header=True)
            else:
                opinion_seed = row['opinion_seed'].values[0]
                opinion_seed = int(opinion_seed)
        lock.release()
        
        ## Specify the save file names for matfiles and txtfiles
        savename = ""
        if self.graph_type == 'erdos-renyi':
            savename = 'p' + str(self.p) + '/graph' + str(graph_number) + '/'
            savename = savename + 'p' + str(self.p) + '-graph' + str(graph_number) + "--"
        savename = savename + 'delta' + str(delta) + '-gamma' + str(gamma) + '--c' + str(c)
        txtfile = self.foldername + '/txtfiles/delta' + str(delta) + '-gamma' + str(gamma) + '/' + savename + '.txt'
        
        ## If the txtfile doesn't exist yet, create it and write the header with seed values to it
        lock.acquire()
        if not os.path.exists(txtfile):
            print(txtfile)
            with open(txtfile, 'w') as f:
                print('Experiment:', self.graph_type, ", n =", self.n, file=f, flush=True)
                if self.graph_type == 'erdos-renyi':
                    print("p =", self.p, ", graph_number = ", graph_number, file=f, flush=True)
                    print('graph_seed = ', graph_seed, file=f, flush=True)
                print('delta = ', delta, ' and gamma = ', gamma, file=f, flush = True)
                print('c = ', c, file=f, flush = True)
                print('opinion_seed = ', opinion_seed, file=f, flush = True)
        lock.release()
        
        # Generate graph
        if self.graph_type == "complete":
            G = igraph.Graph.Full(self.n)
        elif self.graph_type == "erdos-renyi":
            #Reinitialize the random seed and generate the corresponding graph number from that seed
            random_graph = np.random.default_rng(graph_seed)
            for i in range(graph_number + 1):
                seed = random_graph.integers(low=0, high=sys.maxsize)
                random.seed(a=seed)
                G = igraph.Graph.Erdos_Renyi(self.n, self.p)
            
        #Reinitialize the random seed and generate the corresponding opinion set from that seed
        random_opinion = np.random.default_rng(opinion_seed)
        for i in range(opinion_set + 1):
            init_opinions = random_opinion.uniform(0, 1, size=self.n)
        G.vs['opinion'] = init_opinions
        
        #Time the HK simulation for this weight + opinion set combo
        start_time = time.time()

        ## Run the DW model using the simulation seed
        # print('Process Number ', getpid(), 'starting HK') #deleteline
        outputs = dbc.HK(G, c, delta, gamma,
                        tol = self.tol, Tmax = self.Tmax)
        # print('Process Number ', getpid(), 'finished HK') #deleteline

        # Dump model outputs into file
        lock.acquire()
        with open(txtfile, 'a') as f:
            print("\n----- Opinion_set = %s -----" % opinion_set, file=f, flush=True)

            print("T = %s" % outputs['T'], file=f, flush=True)
            print("Min confidence = %.3f, and Max confidence = %.3f" % (min(outputs['confidence']), max(outputs['confidence'])), file=f, flush=True)
            print("Number of Clusters = %s" % outputs['n_clusters'], file=f, flush=True)

            print("Cluster Membership", file=f, flush=True)
            print(outputs['clusters'], file=f, flush=True)
            
            runtime = time.time() - start_time
            print('-- Runtime was %.0f seconds = %.3f hours--' % (runtime, runtime/3600) , file=f, flush=True)
        lock.release()

        ## Define dictionary to store simulation outputs for saving to a .mat file
        save_sim = {'c': c, 'delta': delta, 'gamma':gamma, 'opinion_set': opinion_set}
        
        #Include the graph-level results
        save_sim['T'] = outputs['T']       
        save_sim['T_acc'] = outputs['T_acc']
        save_sim['bailout'] = outputs['bailout']
        save_sim['avg_opinion_diff'] = outputs['avg_opinion_diff']
        
        #Include the cluster information
        clusters = outputs['clusters']
        save_sim['n_clusters'] = outputs['n_clusters']
        for i in range(outputs['n_clusters']):
            key = 'cluster' + str(i)
            save_sim[key] = clusters[i]
            #clusters can be extracted from matfile using list = clusteri.flatten().tolist()
            
        #Include the node-level results as size n arrays
        save_sim['init_opinions'] = init_opinions
        save_sim['final_opinions'] = outputs['final_opinions']
        save_sim['total_change'] = outputs['total_change']
        save_sim['local_receptiveness'] = outputs['local_receptiveness']
        
        #Edge level information
        save_sim['confidence'] = outputs['confidence']
        
        ## Save the simulation results to a matfile
        folder = '/delta' + str(delta) + '-gamma' + str(gamma) + '/'
        matfile = self.foldername + '/matfiles' + folder + savename + '-op' + str(opinion_set) +'.mat'
        io.savemat(matfile, save_sim)
        
    
def init(l):
    global lock
    lock = l
    

if __name__ == "__main__":

    ## EXPERIMENT PARAMETERS - CHANGE HERE
    graph_type = 'complete'
    n = 1000 #Complete graph size
    
    #Confidence shrinker
    deltas = [0.99, 0.5, 0.3, 0.7]
    deltas = [0.01, 0.1, 0.5, 0.9, 0.95, 0.99]

    #Confidence expander
    gammas = [0.01, 0.1, 0.3] 
    gammas = [0.1, 0.01, 0.05, 0.005, 0.001, 0.0005, 0.0001]

    delta_gammas = [(1.0, 0.0)]
    
    for delta in deltas:
        for gamma in gammas:
            delta_gammas.append((delta, gamma))
    
    #Initial confidence radius
    cs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.3, 0.4, 0.5]
    
    opinion_sets = list(range(0,10)) #Which opinion sets to run
    
    ## Generate list of tuples to feed into DW_experiments as parameters
    params_list = []
    for pair in delta_gammas:
        delta = pair[0]
        gamma = pair[1]
        for c in cs:
            for opinion_set in opinion_sets:

                matfile = (graph_type + str(n) + '/matfiles/'
                           + 'delta' + str(delta) + '-gamma' + str(gamma) 
                           + '/delta' + str(delta) + '-gamma' + str(gamma) 
                           + '--c' + str(c) + '-op' + str(opinion_set) + '.mat')

                try:
                    results = io.loadmat(matfile)

                except:
                    param_dict = {"delta": delta, "gamma": gamma,
                                  "c": c, "opinion_set": opinion_set}
                    params_list.append(param_dict) 
                    
    #Initialize experiment class
    experiment = HK_experiment('complete', n = n)
    experiment.generate_seed_files()

    l = multiprocessing.Lock()

    with multiprocessing.Pool(processes=45, initializer=init, initargs=(l,)) as pool:
        pool.map(experiment.run_HK, params_list)