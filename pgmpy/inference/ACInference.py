#!/usr/bin/env python3
import networkx as nx
import numpy as np
from pgmpy.extern.six.moves import filter, range
from collections import defaultdict

from pgmpy.extern.six import string_types
from pgmpy.utils import StateNameDecorator, StateNameInit
from pgmpy.inference.EliminationOrder import WeightedMinFill
from pgmpy.factors.discrete import AlgebraicDecisionDiagram

import logging

class ACInference():

    @StateNameInit()
    def __init__(self, model, add_factors, global_node_id_gen):
        self.model = model
        self.variables = model.nodes()

        self.cardinality = {}
        self.factors = defaultdict(list)

        self.node_id_gen = global_node_id_gen

        for node in model.nodes():
                add = add_factors[node]
                cpd = model.get_cpds(node)
                self.cardinality[node] = cpd.variable_card
                for var in cpd.variables:
                    self.factors[var].append(add)