from typing import List
from KnowledgeGraphQAGen import KnowledgeGraphQAGenerator
import random
from Database import Database 

#Class for generating QA Pairs using Random Vertices Approach
class RandomVerticeQAGenerator(KnowledgeGraphQAGenerator):
    def __init__(self, export_path: str, do_export: bool = True, db=Database(), num_ctx_records: int = 50, ctx_per_record: int = 2, decode_batch_size: int = 1):
        super().__init__(export_path, do_export, db, num_ctx_records, ctx_per_record, decode_batch_size=decode_batch_size)

    def get_random_nodes(self, n_nodes:int = 5, size:int = 20) -> List[List[str]]: #, filters=[no_filter]):
        regs, _, _ = self.db.get_database()
        df = regs
        if df['id'].tolist():
            nodes = df['id'].tolist()
            count = 0
            node_s = []
            while count < size: 
                nodes_selected = random.choices(nodes, k=n_nodes)
                node_s.append(nodes_selected)
                count += 1
            return node_s

    def get_seed_vertice_sets(self, seeds_per_set: int=2, num_sets: int=50) -> List[List[str]]:
        return self.get_random_nodes(n_nodes=self.ctx_per_record, size=self.num_ctx_records)