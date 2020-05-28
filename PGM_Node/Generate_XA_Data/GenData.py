import os
import networkx as nx
import numpy as np
import math

import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors

import gengraph
import configs
import featgen
import utils

prog_args = configs.arg_parse()

if prog_args.dataset is not None:
    if prog_args.dataset == "enron":
        print("Generate enron dataset")
        
    elif prog_args.dataset == "ppi_essential":
        print("Generate ppi_essential dataset")
    
    elif prog_args.dataset == "eucore":
        print("Loading eucore dataset")
        G, labels, name = utils.read_eucore(feature_generator=featgen.ConstFeatureGen(np.ones(prog_args.input_dim, dtype=float)))
        utils.save_XAL(G,labels,prog_args)
        
    elif prog_args.dataset == "amazon":
        print("Loading amazon dataset")
        G, A, X, name = utils.read_amazon()
        pathA = os.path.join('XAL',prog_args.dataset+'_A')
        pathX = os.path.join('XAL',prog_args.dataset+'_X')
        np.save(pathA,A)
        np.save(pathX,X)
    
    elif prog_args.dataset == "bitcoinalpha":
        print("Loading bitcoinalpha dataset")
        G, labels, name = utils.read_bitcoinalpha(feature_generator=None)
        utils.save_XAL(G,labels,prog_args)
    
    elif prog_args.dataset == "bitcoinotc":
        print("Loading bitcoinotc dataset")
        G, labels, name = utils.read_bitcoinotc(feature_generator=None)
        utils.save_XAL(G,labels,prog_args)
    
    elif prog_args.dataset == "epinions":
        print("Loading epinions dataset")
        G, labels, name = utils.read_epinions(feature_generator=featgen.ConstFeatureGen(np.ones(prog_args.input_dim, dtype=float)))
        utils.save_XAL(G,labels,prog_args)
    
    else:
        generate_function = "gengraph.gen_" + prog_args.dataset
        G, labels, name = eval(generate_function)(
            feature_generator=featgen.ConstFeatureGen(np.ones(prog_args.input_dim, dtype=float)))
        utils.save_XAL(G,labels,prog_args)

