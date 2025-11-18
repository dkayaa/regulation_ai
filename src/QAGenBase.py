import os
from pandas import DataFrame
from typing import Any
from Database import Database 
import random 

class QuestionAnswerGeneratorBase: 
    def __init__(self, export_path: str = 'output/default', do_export: bool = True, db: Database = Database(), num_ctx_records: int = 50, ctx_per_record: int = 2, decode_batch_size: int = 1): 
        self.export_path = export_path
        self.do_export = do_export
        self.db = db
        self.num_ctx_records = num_ctx_records
        self.ctx_per_record = ctx_per_record
        self.decode_batch_size = decode_batch_size
        random.seed(42)

    def load(self):
        pass

    def create_contexts(self) -> DataFrame: 
        raise NotImplementedError("Subclasses should implement this!")
    
    def create_question_answers(self, contexts: DataFrame = DataFrame(), model: Any = None) -> DataFrame: 
        raise NotImplementedError("Subclasses should implement this!")

    def execute(self, model: Any = None) -> DataFrame: 
        self.load() 
        contexts = self.create_contexts()
        print("{0} contexts created.".format(len(contexts)))
        print("{0} Average length per context.".format(sum([len(c) for c in contexts['contexts']])/(len(contexts)*self.ctx_per_record) if len(contexts) > 0 else 0))
        if not os.path.exists(self.export_path):
            os.makedirs(self.export_path)
        if self.do_export and contexts is not None:
            o_fp = os.path.join(self.export_path, "contexts.json")
            contexts.to_json(o_fp, orient='records', indent=2)
        #print(contexts.head())
        qas = self.create_question_answers(contexts, model)
        if self.do_export and qas is not None: 
            o_fp = os.path.join(self.export_path, "qas.json")
            qas.to_json(o_fp, orient='records', indent=2)
        return qas    