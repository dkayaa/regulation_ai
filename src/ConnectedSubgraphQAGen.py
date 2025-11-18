from itertools import combinations
from typing import List
from Database import Database
from KnowledgeGraphQAGen import KnowledgeGraphQAGenerator

#Class for generating QA Pairs using CS_n Approach
class ConnectedSubgraphQAGenerator(KnowledgeGraphQAGenerator):
    def __init__(self, export_path: str, do_export: bool = True, db=Database(), num_ctx_records: int = 50, ctx_per_record: int = 2, decode_batch_size: int = 1, do_path_extraction: bool = True):
        super().__init__(export_path, do_export, db, num_ctx_records, ctx_per_record, decode_batch_size=decode_batch_size, do_path_extraction=do_path_extraction)

    def get_all_connected_subgraphs(self, k: int, edge_types: List[str]=['includes']) -> List[List[str]]:
        edges = self.get_edges(r=edge_types)
        edge_combinations = combinations(edges, k-1)
        sglist = []
        for ec in edge_combinations: 
            nodes = set()
            for e in ec: 
                nodes.add(e[0])
                nodes.add(e[2])

            if len(nodes) == len(ec) + 1:
                sglist.append(list(nodes))
        
        return sglist

    def get_seed_vertice_sets(self, seeds_per_set: int=2, num_sets: int=50) -> List[List[str]]:
        return self.get_all_connected_subgraphs(self.ctx_per_record, ['refers', 'overrides','activates', 'modifies','conditioned on'])
    