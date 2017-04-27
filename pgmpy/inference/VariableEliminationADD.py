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

class VariableEliminationADD():

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

    @StateNameDecorator(argument='evidence', return_val=None)
    def _variable_elimination(self, variables, operation, evidence=None, elimination_order=None):

        if isinstance(variables, string_types):
            raise TypeError("variables must be a list of strings")
        if isinstance(evidence, string_types):
            raise TypeError("evidence must be a list of strings")

        # Dealing with the case when variables is not provided.
        if not variables:
            all_factors = []
            for factor_li in self.factors.values():
                all_factors.extend(factor_li)
            return set(all_factors)

        eliminated_variables = set()
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
            ### DEBUG
            # print(">>>>> Involved tables:")
            # print([ f.variables for f in factors])
            # for f in factors:
            #     from networkx.drawing.nx_pydot import write_dot
            #     write_dot(f.graph, var+"_"+"_".join(f.variables)+"_prod.dot")
            ###---DEBUG
            phi = VariableEliminationADD._adds_product(factors, self.node_id_gen)
            ### DEBUG
            # print(">>>>> Product:")
            # from networkx.drawing.nx_pydot import write_dot
            # write_dot(phi.graph, var+"_prod.dot")
            # return
            ###---DEBUG
            phi = phi.marginalize([var], self.node_id_gen, inplace=False)
            ### DEBUG
            # print(">>>>> Marginalization:")
            from networkx.drawing.nx_pydot import write_dot
            write_dot(phi.graph, var+"_marg.dot")
            # if var == "smoke":
            #     return
            ###---DEBUG
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

        query_var_factor = {}
        for query_var in variables:
            phi = VariableEliminationADD._adds_product(final_distribution, self.node_id_gen)
            ### DEBUG
            # print(">>>>> Final Product:")
            from networkx.drawing.nx_pydot import write_dot
            write_dot(phi.graph, var+"fin_prod.dot")
            ###---DEBUG
            query_var_factor[query_var] = phi.marginalize(list(set(variables) - set([query_var])), self.node_id_gen, inplace=False)
            ### DEBUG
            # print(">>>>> Final Marginalization:")
            from networkx.drawing.nx_pydot import write_dot
            write_dot(query_var_factor[query_var].graph, var+"fin_marg.dot")
            ###---DEBUG
            # TODO: Can't normalize ADD
            # query_var_factor[query_var] = query_var_factor[query_var].normalize(inplace=False)
        return query_var_factor

    def query(self, variables, evidence=None, elimination_order=None):
        if not all([v in self.model.nodes() for v in variables]):
            raise ValueError("Variables {0} are not in given model.".format(",".join(variables)))
        return self._variable_elimination(variables, 'marginalize', evidence=evidence, elimination_order=elimination_order)

    @staticmethod
    def _adds_product(adds, global_node_id_gen):
        phi = None
        is_first = True
        for factor in adds:
            if is_first:
                phi = factor
                is_first = False
            else:
                phi = phi.product(factor, global_node_id_gen, inplace=False)
        return phi
