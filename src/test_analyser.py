#Class that analyses test results and generates reports

class TestAnalyser:
    def __init__(self, test_folder, trim_data=False, random_seed=42, test_set_size=100):
        self.test_folder = test_folder
        self.data = []
        self.test_results = []
        self.scores = {}
        self.trim_data = trim_data
        self.random_seed = random_seed
        self.test_set_size = test_set_size

    def load(self): 
        # Load test results from the specified folder
        import os
        import json

        if self.test_folder.endswith('.json'):
            with open(self.test_folder, 'r') as file:
                result = json.load(file)
                if isinstance(result, dict):
                    self.test_results.append(result)
                elif isinstance(result, list):
                    self.test_results.extend(result)
                return      
        for filename in os.listdir(self.test_folder):
            if filename.endswith('.json'):
                with open(os.path.join(self.test_folder, filename), 'r') as file:
                    result = json.load(file)
                    if isinstance(result, dict):
                        self.test_results.append(result)
                    elif isinstance(result, list):
                        self.test_results.extend(result)


    def trim(self):
        import random 
        if not self.trim_data:
            return
        if not self.data:
            raise ValueError("Data not transformed. Please run transform_data() before shuffling and selecting.")
        random.seed(self.random_seed)
        random.shuffle(self.data)
        self.data = self.data[:self.test_set_size]
    
    def transform_data(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def transform_and_trim(self): 
        self.transform_data()
        self.trim()

    def get_test_folder(self):
        return self.test_folder
    
    def score_data(self, scoring_fn):
        if not self.data:
            raise ValueError("Data not transformed. Please run transform_data() before scoring.")
        # Score each test result using the provided scoring function
        self.scores = scoring_fn(self.data)
    
    def print_statistics(self):
        # Print basic statistics about the test results
        print(f"{self.test_folder} \t Total data points: {len(self.data)} {','.join([f'{key}: {value:.3f}' for key, value in self.scores.items()])}")

    def print_scores(self, scores):
        # Print the scores in a readable format
        for key, value in scores.items():
            print(f"{key}:\t {value}")