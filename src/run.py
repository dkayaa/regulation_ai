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

output_folder_name = 'output2'
ctx_per_record = 2
num_ctx_records = 50

generators.append(CRIKEYQAGGenerator(export_path=f'{output_folder_name}/CRIKEY_QAG', do_export=True, db=Database(), num_ctx_records=num_ctx_records, ctx_per_record=ctx_per_record, decode_batch_size=4))

#Ablations
generators.append(CRIKEYQAGGenerator(export_path=f'{output_folder_name}/ContextExtractionQA', do_export=True, db=Database(), num_ctx_records=num_ctx_records, ctx_per_record=ctx_per_record, decode_batch_size=4, do_entity_pathing=False))
generators.append(RandomSeedSelectionQAGenerator(export_path=f'{output_folder_name}/RandomSeedSelectionQA', do_export=True, db=Database(), num_ctx_records=num_ctx_records, ctx_per_record=ctx_per_record, decode_batch_size=4))

# Baselines
generators.append(RandomVertexQAGenerator(export_path=f'{output_folder_name}/RandomVertexQA', do_export=True, db=Database(), num_ctx_records=1000, ctx_per_record=10, decode_batch_size=4, explainqar_results_fp=f'{output_folder_name}/CRIKEY_QAG/qas.json'))
generators.append(SlidingWindowQAGenerator(export_path=f'{output_folder_name}/SlidingWindowQA', do_export=True, db=Database(), num_ctx_records=1000, ctx_per_record=ctx_per_record, decode_batch_size=4, explainqar_results_fp=f'{output_folder_name}/CRIKEY_QAG/qas.json', window_size=500))
generators.append(DRQAGen(export_path=f'{output_folder_name}/DRQA', do_export=True, db=Database(), num_ctx_records=1000, ctx_per_record=ctx_per_record, decode_batch_size=4, explainqar_results_fp=f'{output_folder_name}/CRIKEY_QAG/qas.json', chunk_size=500, k=2))

for generator in generators:
   generator.execute(model=model_dict.create()['claude'])