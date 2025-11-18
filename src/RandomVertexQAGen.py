from typing import List, Tuple
from KnowledgeGraphQAGen import KnowledgeGraphQAGenerator
import random
import json
from Database import Database 
import pandas as pd
import json
#Class for generating QA Pairs using Random Vertices Approach
class RandomVertexQAGenerator(KnowledgeGraphQAGenerator):
    def __init__(self, export_path: str, do_export: bool = True, db=Database(), num_ctx_records: int = 50, ctx_per_record: int = 2, decode_batch_size: int = 1, explainqar_results_fp: str = ""):
        super().__init__(export_path, do_export, db, num_ctx_records, ctx_per_record, decode_batch_size=decode_batch_size)
        self.explainqar_results_fp = explainqar_results_fp

    def get_random_nodes(self, n_nodes:int = 5, size:int = 20) -> List[List[str]]: #, filters=[no_filter]):

        regs, _, _ = self.db.get_database()
        df = regs

        if not df['id'].tolist():
            return []
        
        if self.explainqar_results_fp != "":
            explainqar_df = json.load(open(self.explainqar_results_fp, 'r'))
            explainqar_df = pd.DataFrame(explainqar_df)

        nodes = df['id'].tolist()
        node_s = []

        if explainqar_df is not None and not explainqar_df.empty:
            for _, row in explainqar_df.iterrows():
                seed_nodes = row['path']
                nodes_selected = random.choices(nodes, k=(n_nodes - len(seed_nodes)))
                nodes_selected.extend(seed_nodes)
                node_s.append(nodes_selected)
        else:
            count = 0
            while count < size: 
                nodes_selected = random.choices(nodes, k=n_nodes)
                node_s.append(nodes_selected)
                count += 1
        return node_s

    def get_seed_vertice_sets(self, seeds_per_set: int=2, num_sets: int=50) -> List[List[str]]:
        return self.get_random_nodes(n_nodes=self.ctx_per_record, size=self.num_ctx_records)
    
    # Overload path_context_extraction to ommit path
    def path_context_extraction(self, node_id: str) -> Tuple[str, list]: 
        path = [node_id]
        d = self._merge_docs_df(self._get_docs_df(path))
        return (d.page_content, path)