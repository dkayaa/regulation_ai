import json
from pydantic import BaseModel
import re 
from Database import Database

class CitationAccuracyInputModel(BaseModel):
    path: list[int]
    contexts: list[str]
    question: str 
    answer: str

class CitationAccuracyMetricCalculator:
    def __init__(self, ground_truth_file: str = "", regid_name_file: str = "data/regid_name_map.json"):
        self.ground_truth_citations = set()
        self.ground_truth_file = ground_truth_file
        self.regid_name_file = regid_name_file
        self.regex_pattern = r"(\d+)(\(\d+\)|\([a-zA-Z]+\)|\([ivxlc]+\))+"
        self.regid_name_map = {}

    def get_regex_pattern(self):
        return self.regex_pattern
    
    def load(self): 
        #with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
            #self.ground_truth_citations = set(json.load(f))
        with open(self.regid_name_file, 'r', encoding='utf-8') as f:
            temp_map = json.load(f)
            self.regid_name_map = {str(item['id']): item['reg_section'] for item in temp_map}

    def compute_precision_recall_f1_single(self, data): 
        try: 
            input = CitationAccuracyInputModel(**data)
        except Exception as e:
            print(f"Error parsing input data: {e}")
            return 0.0, 0.0, 0.0
        
        # Compute Ground Truth Citations for the given path
        path_citations = set()
        for reg_id in input.path:
            reg_name = self.regid_name_map.get(str(reg_id), "")
            if reg_name:
                path_citations.add(reg_name)

        ground_truth_matches = re.finditer(self.regex_pattern, "\n".join([c for c in input.contexts]))
        context_citations = set([match.group(0) for match in ground_truth_matches])
        ground_truth_citations = path_citations.union(context_citations)

        matches = re.finditer(self.regex_pattern, input.question + input.answer)
        citations = set([match.group(0) for match in matches])
        # Compute precision based on the ground truth citations and the provided data
        true_positives = sum(1 for item in citations if item in ground_truth_citations)
        false_positives = len(citations) - true_positives
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        # recall isnt really useful as this this assumes ground truth should have all citations. But we compute it anyway
        recall = true_positives / len(ground_truth_citations) if len(ground_truth_citations) > 0 else 0.0
        #recall = 1.0 # assume recall is perfect since we dont have negative examples
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1_score

    def compute_precision_recall_f1(self, data_list):
        precisions = []
        recalls = []
        f1_scores = []

        for data in data_list:
            precision, recall, f1_score = self.compute_precision_recall_f1_single(data)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return avg_precision, avg_recall, avg_f1_score