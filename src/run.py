from Database import Database
from RandomSeedSelectionQAGen import RandomSeedSelectionQAGenerator
from CRIKEYQAG import CRIKEYQAGGenerator
from SlidingWindowQAGen import SlidingWindowQAGenerator
from RandomVertexQAGen import RandomVertexQAGenerator
from DRQAGen import DRQAGen
import model_dict
import dotenv
dotenv.load_dotenv()

generators = [] 


generators.append(CRIKEYQAGGenerator(export_path='output/CRIKEY_QAG', do_export=True, db=Database(), num_ctx_records=50, ctx_per_record=2, decode_batch_size=4))

#Ablations
generators.append(CRIKEYQAGGenerator(export_path='output/CRIKEY_QAG_noEP', do_export=True, db=Database(), num_ctx_records=50, ctx_per_record=2, decode_batch_size=4, do_entity_pathing=False))
generators.append(RandomSeedSelectionQAGenerator(export_path='output/RandomSeedSelectionQA', do_export=True, db=Database(), num_ctx_records=50, ctx_per_record=2, decode_batch_size=4))

# Baselines
generators.append(RandomVertexQAGenerator(export_path='output/RandomVertexQA', do_export=True, db=Database(), num_ctx_records=1000, ctx_per_record=5, decode_batch_size=4, explainqar_results_fp='output/CRIKEY_QAG/qas.json'))
generators.append(SlidingWindowQAGenerator(export_path='output/SlidingWindowQA', do_export=True, db=Database(), num_ctx_records=1000, ctx_per_record=10, decode_batch_size=4, explainqar_results_fp='output/CRIKEY_QAG/qas.json', window_size=500))
generators.append(DRQAGen(export_path='output/DRQA', do_export=True, db=Database(), num_ctx_records=1000, ctx_per_record=10, decode_batch_size=4, explainqar_results_fp='output/CRIKEY_QAG/qas.json', chunk_size=500, k=2))

for generator in generators:
   generator.execute(model=model_dict.create()['claude'])