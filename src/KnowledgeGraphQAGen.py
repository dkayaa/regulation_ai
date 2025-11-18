from typing import List, Tuple, Any
from langchain.schema import Document
from QAGenBase import QuestionAnswerGeneratorBase
import random 
from pandas import DataFrame 
from Database import Database
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from DataModel import MHQuestionAnswers
import json 

class KnowledgeGraphQAGenerator(QuestionAnswerGeneratorBase):

    def __init__(self, export_path: str, do_export: bool = True, db=Database(), num_ctx_records: int = 50, ctx_per_record: int = 2, decode_batch_size: int = 1, do_path_extraction: bool = True):
        super().__init__(export_path, do_export, db, num_ctx_records, ctx_per_record, decode_batch_size)
        self.do_path_extraction = do_path_extraction

    def get_seed_vertice_sets(self, seeds_per_set: int=2, num_sets: int=50) -> List[List[str]]:
        raise NotImplementedError("Subclasses should implement this!")

    def create_contexts(self) -> DataFrame:
        records = [] 
        path_list = [] 
        visited = {}

        path_list = self.get_seed_vertice_sets(seeds_per_set=self.ctx_per_record, num_sets=self.num_ctx_records)

        random.shuffle(path_list)

        count = 0
        visited = {} 
        for path in path_list:
            if count >= self.num_ctx_records: 
                print("Context Limit Reached! Exiting!")
                break 
            key = ""
            contexts = []
            cpaths = []
            identical_cpath = False 

            for node_id in path:
                record = {} 
                if self.do_path_extraction:
                    (d, cpath) = self.path_context_extraction(node_id)
                else:
                    # retrieve single node text
                    regs, _, _ = self.db.get_database()
                    df = regs[regs['id'] == node_id][['id', 'reg_description']]
                    d = ""
                    if not df.empty:
                        d = df.iloc[0]['reg_description']
                    cpath = [node_id]

                contexts.append(d)
                cpaths.append(cpath)
                
            if identical_cpath: 
                continue 

            key+= ','.join([str(x) for x in sorted([x for xs in cpaths for x in xs])])
            #key+= ','.join([str(n) for n in sorted(path)]) + '-'
            if key in visited.keys():
                continue
            visited[key] = True
            cps = [] 

            skip = False
            
            if skip: 
                continue 

            print("context perplexities: {0}".format(cps))
            print("Accepted!, appending record {0}".format(count))

            record['path'] = path 
            record['contexts'] = contexts
            
            count+=1
            records.append(record) 
        return DataFrame(records) 


    def create_question_answers(self, contexts: DataFrame = DataFrame(), model: Any = None) -> DataFrame:
        # Implementation for creating question-answer pairs specific to explanations
        pass

    def get_edges(self, r: List[str]=[]): 
        regs, reg_relation, rel_type = self.db.get_database()

        if r:
            df = reg_relation[reg_relation['rel_type_id'].isin(
                rel_type[rel_type['rel_description'].isin(r)]['id'].tolist()
            )][['to_id', 'from_id', 'rel_type_id']]
            df = df.merge(rel_type[rel_type['rel_description'].isin(r)][['id', 'rel_description']], left_on='rel_type_id', right_on='id')[['to_id', 'from_id', 'rel_description']] 

        else: #select all edges. 
            df = reg_relation.merge(rel_type, left_on='rel_type_id', right_on='id')[['to_id', 'from_id', 'rel_description']]
        if df['to_id'].tolist():
            E = list(zip(df['from_id'].tolist(), df['rel_description'].tolist(), df['to_id'].tolist()))
            return E

    # HELPER METHODS 
    def _merge_docs_df(self, df: DataFrame) -> Document:
        contents = '' 
        if df['reg_description'].tolist():
            for c in df['reg_description'].tolist():
                contents = contents + c
                
        doc = Document(page_content = contents) 
        return doc

    def _get_docs_df(self, s: list) -> DataFrame: 
        regs, _, _ = self.db.get_database()
        df = regs[regs['id'].isin(s)][['id', 'reg_description']]
        return df

    def path_context_extraction(self, node_id: str) -> Tuple[str, list]: 
        to_root = self.get_path_reverse(node_id)
        to_leaf = self.get_path(node_id, self.select_random_child)
        path = to_root + to_leaf
        d = self._merge_docs_df(self._get_docs_df(path))
        return (d.page_content, path)

    #default handler - Passthrough
    def passthrough_handler(self, children: List[str], params: dict={}) -> List[str]: 
        return children

    #handler - 1 Random selector
    def select_random_child(self, children: List[str], params: dict={}) -> List[str]: 
        return random.choices(children,k=1)

    def get_parent(self, root: str, r: List[str] = ['includes'], parent_handler: callable=None) -> List[str]: 
        _, reg_relation, rel_type = self.db.get_database()
        if r:
            df = reg_relation.merge(rel_type[rel_type['rel_description'].isin(r)][['id', 'rel_description']], left_on='rel_type_id', right_on='id')[['to_id', 'from_id', 'rel_description']]
            df = df[df['to_id'].isin([root])]
        else: 
            return []

        if df['from_id'].tolist():
            p = parent_handler(children=df['from_id'].tolist(), params={})
            return p    
        else:
            return []

    def get_child(self, root: str, children_handler: callable=None, r: List[str] = []) -> List[str]: 
        _, reg_relation, rel_type = self.db.get_database()
        if r:
            df = reg_relation.merge(rel_type[rel_type['rel_description'].isin(r)][['id', 'rel_description']], left_on='rel_type_id', right_on='id')[['to_id', 'from_id', 'rel_description']]
            df = df[df['from_id'].isin([root])]
        else: 
            return []

        if df['to_id'].tolist():
            c = children_handler(df['to_id'].tolist())
            return c    
        else:
            return [] 

    def get_path(self, root: str, child_handler: callable, child_edge_types: List[str] = ['includes'], len_min: int = 1, len_max: int = 999) -> List[str]: 
        current = [root]
        path = []
        while current != []: 
            path.append(current[0]) 
            current = self.get_child(current[0], child_handler, r=child_edge_types)
            if current in path: 
                #cycle 
                break

        #check if path length within set bounds 
        if len(path) > len_max or len(path) < len_min:
            return []
        
        return path

    def get_path_reverse(self, leaf: str) -> List[str]: 
        current = [leaf]
        path_rev = []
        while current != []: 
            #print(current)
            path_rev.append(current) 
            current = self.get_parent(current[0], r=['includes'], parent_handler=self.passthrough_handler)

        path = [] 
        while len(path_rev) > 0: 
            path.append(path_rev.pop()[0])
        
        return path[:-1]
    

    def create_question_answers(self, contexts: DataFrame = DataFrame(), model: Any = None) -> DataFrame: 

        contexts['result_dict'] = None
        contexts['result_str'] = None

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
            
            path = context_record.get('path', [])
            print("Commencing for Path {0}".format(path))
            
            res = ""
            try:
                res = chain.invoke({
                    "num_contexts": len(contexts),
                    "batch_size": self.decode_batch_size,
                    "context": '\n\n'.join(['context: ' + str(c) for c in context_group]),
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

        return contexts