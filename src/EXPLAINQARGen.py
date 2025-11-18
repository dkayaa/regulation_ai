import json
import random
from ConnectedSubgraphQAGen import ConnectedSubgraphQAGenerator
from Database import Database
from pandas import DataFrame
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Any
from DataModel import MHQuestionAnswers
from EntityPathExtractor import EntityPathExtractor

class EXPLAINQARGenerator(ConnectedSubgraphQAGenerator): 
    def __init__(self, export_path: str, do_export: bool = True, db=Database(), num_ctx_records: int = 50, ctx_per_record: int = 2, decode_batch_size: int = 1, do_path_extraction: bool = True, do_entity_pathing: bool = True):
        super().__init__(export_path, do_export, db, num_ctx_records, ctx_per_record, decode_batch_size=decode_batch_size, do_path_extraction=do_path_extraction)
        self.entity_path_extractor = EntityPathExtractor()
        self.do_entity_pathing = do_entity_pathing

    def create_question_answers(self, contexts: DataFrame = DataFrame(), model: Any = None) -> DataFrame: 

        contexts['result_dict'] = None
        contexts['result_str'] = None
        contexts['misc_info'] = None
        if model is None:
            raise ValueError("Model must be provided for question answer generation.")
        template="""
            You are given {num_contexts} related sections from a non-prescription medicines labels standard regulation. These sections depend on each other, so they must all be considered together when reasoning.

            Your task is to generate {batch_size} question and answer pairs that require multi-hop reasoning. That is, the questions must require using information from all of the provided sections of the regulation to answer correctly.
            The regulation is much broader in scope than the related sections. Therefore, the questions generated should state assumptions such that all necessary information is clearly present, requiring no significant assumptions or inferences. The contexts should directly address all aspects of the question, providing a clear path to a definitive answer with no ambiguity

            Each question should:
            Clearly state any assumptions that affect the answer.
            Be designed so the answer can only be determined by referencing all regulation sections.
            Should only use the information provided in the contexts.
            {guided_str}
            
            Each answer must:
            Provide an explanation that is completely accurate, comprehensive, well-structured, and directly references the specific relevant parts of the context. It fully addresses all aspects of the question with proper reasoning and leaves no room for ambiguity. 
            Justify the answer by referencing specific regulation IDs or clauses

            {context} 

            Your answer should strictly be a valid JSON object. Do not output any other text apart from the JSON object.
            """

        parser = JsonOutputParser(pydantic_object=MHQuestionAnswers)

        prompt = PromptTemplate(
            template=template + "\n {format_instructions}. You must always return valid JSON. Do not return any additional text.\n",
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = (
            prompt
            | model
            | StrOutputParser()
        )

        for i, context_record in contexts.iterrows():
            context_group = context_record.get('contexts', []) 
            e_paths = []
            ep=[]
            guided_str = ''
            if self.do_entity_pathing:
                guided_str = "Clearly link the following entities: \n"
                e_paths = self.entity_path_extractor.get_entity_paths(context_group)
                if len(e_paths) == 0: 
                    print("Guided Strategy Produced No Entity Paths!, Next Context!")
                    continue 
            
                ep = random.choices(e_paths, k=1)
                p_str = '->'.join([str(i) for i in ep])+'\n'
                guided_str += p_str
                
            path = context_record.get('path', [])
            print("Commencing for Path {0}".format(path))
            
            res = ""
            try:
                res = chain.invoke({
                    "num_contexts": len(contexts),
                    "batch_size": self.decode_batch_size,
                    "context": '\n\n'.join(['context: ' + str(c) for c in context_group]),
                    "guided_str" : guided_str
                    })
            except Exception as e: 
                print(e)
                continue

            res_d = {}
            try: 
                res_d = json.loads(res) 
            except Exception as e: 
                print(e) 
            
            contexts.iat[i, contexts.columns.get_loc('result_dict')] = res_d
            contexts.iat[i, contexts.columns.get_loc('result_str')] = res
            misc_info = {
                'guided_entity_path': ep
            }
            contexts.iat[i, contexts.columns.get_loc('misc_info')] = misc_info
        return contexts

