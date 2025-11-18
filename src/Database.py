from pandas import DataFrame
import pandas as pd


class Database:

    def __init__(self):

        regs_loc = 'data/regs.json'
        reg_relation_loc = 'data/reg_relation.json'
        rel_type_loc = 'data/rel_type.json'

        self.regs = pd.read_json(regs_loc)
        self.reg_relation = pd.read_json(reg_relation_loc)
        self.rel_type = pd.read_json(rel_type_loc)

    def get_database(self): 
        if self.regs.empty or self.reg_relation.empty or self.rel_type.empty:
            self.regs = pd.read_json(self.regs_loc)
            self.reg_relation = pd.read_json(self.reg_relation_loc)
            self.rel_type = pd.read_json(self.rel_type_loc)
        return self.regs, self.reg_relation, self.rel_type
    
    def get_all_text(self):
        regs, _, _ = self.get_database()
        return " ".join(regs['reg_description'].tolist())

    def get_statistics(self): 
        num_regs = len(self.regs)
        avg_reg_length = self.regs['reg_description'].apply(lambda x: len(x.split())).mean().item()
        num_relations = len(self.reg_relation)
        num_rel_types = len(self.reg_relation['rel_type_id'].unique())
        return {
            "num_regs_nodes": num_regs,
            "avg_reg_length_node_size": avg_reg_length,
            "num_relations_edges": num_relations,
            "num_rel_types_edge_types": num_rel_types,
            "avg_degree": (2*num_relations) / num_regs if num_regs > 0 else 0
        }