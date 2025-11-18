from KnowledgeGraphQAGen import KnowledgeGraphQAGenerator
from Database import Database
from ConnectedSubgraphQAGen import ConnectedSubgraphQAGenerator
from RandomVerticeQAGen import RandomVerticeQAGenerator
from EXPLAINQARGen import EXPLAINQARGenerator
from SlidingWindowQAGen import SlidingWindowQAGenerator
from RAGRandomQAGen import RAGRandomQAGenerator
from DRQAGen import DRQAGen
import model_dict
import dotenv
dotenv.load_dotenv()

generators = [] 

#generators.append(EXPLAINQARGenerator(export_path='output/EXPLAIN_QAR', do_export=True, db=Database(), num_ctx_records=50, ctx_per_record=2, decode_batch_size=4))

#Ablations
#generators.append(EXPLAINQARGenerator(export_path='output/EXPLAIN_QAR_noEP', do_export=True, db=Database(), num_ctx_records=50, ctx_per_record=2, decode_batch_size=4, do_entity_pathing=False))
#generators.append(EXPLAINQARGenerator(export_path='output/EXPLAIN_QAR_noEP_noPE', do_export=True, db=Database(), num_ctx_records=50, ctx_per_record=2, decode_batch_size=4, do_entity_pathing=False, do_path_extraction=False))

# Baselines
#generators.append(ConnectedSubgraphQAGenerator(export_path='output/default', do_export=True, db=Database(), num_ctx_records=1, ctx_per_record=2, decode_batch_size=2))
generators.append(RandomVerticeQAGenerator(export_path='output/RandomVerticeQA', do_export=True, db=Database(), num_ctx_records=50, ctx_per_record=2, decode_batch_size=4))
#generators.append(RAGRandomQAGenerator(export_path='output/RAGRandomQA', do_export=True, db=Database(), num_ctx_records=1000, ctx_per_record=5, decode_batch_size=4, explainqar_results_fp='output/EXPLAIN_QAR/qas.json'))
#generators.append(SlidingWindowQAGenerator(export_path='output/SlidingWindowQA', do_export=True, db=Database(), num_ctx_records=1000, ctx_per_record=10, decode_batch_size=4, explainqar_results_fp='output/EXPLAIN_QAR/qas.json', window_size=500))
#generators.append(DRQAGen(export_path='output/DRQA', do_export=True, db=Database(), num_ctx_records=1000, ctx_per_record=10, decode_batch_size=4, explainqar_results_fp='output/EXPLAIN_QAR/qas.json', chunk_size=500, k=2))

for generator in generators:
   generator.execute(model=model_dict.create()['claude'])