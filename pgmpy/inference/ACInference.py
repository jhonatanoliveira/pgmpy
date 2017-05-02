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
                add_factor = add_factors[node]
                cpd = model.get_cpds(node)
                self.cardinality[node] = cpd.variable_card
                for var in cpd.variables:
                    self.factors[var] += add_factor


    @StateNameDecorator(argument='evidence', return_val=None)
    def _variable_elimination(self, ac, operation, evidence=None, elimination_order=None):

        variables = []

        eliminated_variables = set()
        ### DEBUG
        print("-----> self.factors")
        print(self.factors)
        ###---DEBUG
        working_factors = {node: {factor for factor in self.factors[node]}
                           for node in self.factors}

        #TODO
        # Dealing with evidence. Reducing factors over it before VE is run.
        # if evidence:
        #     for evidence_var in evidence:
        #         for factor in working_factors[evidence_var]:
        #             factor_reduced = factor.reduce([(evidence_var, evidence[evidence_var])], inplace=False)
        #             for var in factor_reduced.scope():
        #                 working_factors[var].remove(factor)
        #                 working_factors[var].add(factor_reduced)
        #         del working_factors[evidence_var]

        if not elimination_order:
            vars_to_eliminate = list(set(self.variables) -
                                     set(variables) -
                                     set(evidence.keys() if evidence else []))
            elimination_order = WeightedMinFill(self.model).get_elimination_order(vars_to_eliminate)
        elif any(var in elimination_order for var in
                 set(variables).union(set(evidence.keys() if evidence else []))):
            raise ValueError("Elimination order contains variables which are in"
                             " variables or evidence args")
        logging.info("Running VE with query variables: {}, evidence: {}, and elimination order: {}".format(str(variables), str(evidence), str(elimination_order)))
        for var in elimination_order:
            logging.info("VE is eliminating variable {}".format(var))
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = []
            for factor in working_factors[var]:
                if not set(factor.variables).intersection(eliminated_variables):
                    factors.append(factor)
            logging.info("Product of tables involving variable {}".format(var))
            phi = ACInference._adds_product(factors, self.node_id_gen, ac)
            logging.info("Marginalize out variable {}".format(var))
            phi = phi.marginalize([var], self.node_id_gen, ac, inplace=False)
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].add(phi)
            eliminated_variables.add(var)

        final_distribution = set()
        for node in working_factors:
            factors = working_factors[node]
            for factor in factors:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add(factor)

        if len(final_distribution) != 0:
            raise ValueError("No factor should be left after marginalizing all variables")

    def compile(self, ac, evidence=None, elimination_order=None):
        return self._variable_elimination(ac, 'marginalize', evidence=evidence, elimination_order=elimination_order)

    @staticmethod
    def _adds_product(adds, global_node_id_gen, ac):
        phi = None
        is_first = True
        for factor in adds:
            if is_first:
                phi = factor
                is_first = False
            else:
                phi = phi.product(factor, global_node_id_gen, ac, inplace=False)
        return phi
