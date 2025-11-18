from test_analyser import TestAnalyser 


class MyNewTestAnalyser(TestAnalyser):
    def transform_data(self):
        # Apply the transformation function to each test result
        transformed = []

        for test_result in self.test_results:
            for item in test_result.get('result_dict', {}).get('QuestionAnswers', []):
                transformed.append({ 
                    'contexts': test_result.get('input', {}).get('contexts', []),
                    'question': item.get('Question', ''), 
                    'answer': item.get('Answer', '') +'\n' + item.get('Answer Explanation', '')
                })
        self.data = [a for a in transformed if a['question'] and a['answer']]
        self.data = transformed


class MyTestAnalyser(TestAnalyser):
    def transform_data(self):
        # Apply the transformation function to each test result
        transformed = []

        for test_result in self.test_results:
            if test_result.get('result_dict', {}) is None:
                continue
            if 'QuestionAnswers' not in test_result.get('result_dict', {}):
                continue
            
            path = test_result.get('path', [])
            contexts = test_result.get('input', {}).get('contexts', [])
            if contexts == []:
                contexts = test_result.get('contexts', [])
            for item in test_result.get('result_dict', {}).get('QuestionAnswers', []):
                transformed.append({ 
                    'path': path,
                    'contexts': contexts,
                    'question': item.get('Question', ''), 
                    'answer': item.get('Answer', '') +'\n' + item.get('Answer Explanation', '')
                })
        self.data = [a for a in transformed if a['question'] and a['answer']]
        self.data = transformed


class ZSTestAnalyser(TestAnalyser):
    def transform_data(self):
        # Apply the transformation function to each test result
        transformed = []

        for test_result in self.test_results:
            for item in test_result.get('Baseline_Pure_Zero_Shot', {}).get('result_dict', {}).get('QuestionAnswers', []):
                transformed.append({ 
                    'contexts': test_result.get('input', {}).get('contexts', []),
                    'question': item.get('Question', ''), 
                    'answer': item.get('Answer', '') +'\n' + item.get('Answer Explanation', '')
                })
        self.data = [a for a in transformed if a['question'] and a['answer']]
        self.data = transformed

def score_entailment(data, scores, device): 
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from scipy.special import softmax
    import torch

    model_name = "microsoft/deberta-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    min_entailment_scores = []
    max_entailment_scores = []
    combined_entailment_scores = []
    for item in data:
        entailment_scores = []
        for i, c in enumerate(item['contexts']):
            premise = c
            hypothesis = item['question'] + ' ' + item['answer']
            inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = softmax(logits.cpu().numpy(), axis=1)[0]
            entailment_scores.append(probs[2])  # Index 2 corresponds to 'entailment' class
        
        min_entailment_scores.append(min(entailment_scores))
        max_entailment_scores.append(max(entailment_scores))

        premise = ' '.join(item['contexts'])
        hypothesis = item['question'] + ' ' + item['answer']
        inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = softmax(logits.cpu().numpy(), axis=1)[0]
        combined_entailment_scores.append(probs[2])  # Index 2 corresponds to 'entailment' class

    scores['avg_combined_entailment_score'] = sum(combined_entailment_scores)/len(combined_entailment_scores)
    scores['avg_min_entailment_score'] = sum(min_entailment_scores)/len(min_entailment_scores)
    scores['avg_max_entailment_score'] = sum(max_entailment_scores)/len(max_entailment_scores)

def score_cosine_similarity(data, scores, device):
    ## Compute cosine similarity between contexts and answers
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    all_question_answers = [item['question'] + '\n' + item['answer'] for item in data]
    all_contexts = [' '.join(item['contexts']) for item in data]
    embeddings_answers = model.encode(all_question_answers, convert_to_tensor=True)
    embeddings_contexts = model.encode(all_contexts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings_answers, embeddings_contexts)
    cosine_similarities = cosine_scores.diagonal()
    scores['avg_cosine_similarity'] = cosine_similarities.mean().item()

    #lets do this for each context individually and average
    max_cosine_similarities_individual = []
    min_cosine_similarities_individual = []
    for i, item in enumerate(data):
        cosine_similarities_individual = []
        for c in item['contexts']:
            embedding_answer = model.encode(item['question'] + '\n' + item['answer'], convert_to_tensor=True)
            embedding_context = model.encode(c, convert_to_tensor=True)
            cosine_score = util.cos_sim(embedding_answer, embedding_context)
            cosine_similarities_individual.append(cosine_score.item())
        max_cosine_similarities_individual.append(max(cosine_similarities_individual))
        min_cosine_similarities_individual.append(min(cosine_similarities_individual))

    scores['avg_max_cosine_similarity'] = sum(max_cosine_similarities_individual)/len(max_cosine_similarities_individual)
    scores['avg_min_cosine_similarity'] = sum(min_cosine_similarities_individual)/len(min_cosine_similarities_individual)

def score_bleurt(data, scores, device):
    from bleurt import score as bleurt_score
    #all_answers = [item['answer'] for item in data]
    all_question_answers = [item['question'] + '\n' + item['answer'] for item in data]

    all_contexts = [' '.join(item['contexts']) for item in data]
    scorer = bleurt_score.BleurtScorer("data/bleurt-base-128")
    bleurt_scores = scorer.score(references=all_contexts, candidates=all_question_answers, batch_size=8)
    scores['avg_bleurt_score'] = sum(bleurt_scores)/len(bleurt_scores)

def score_bertscore(data, scores, device):
    #### we will compute bertscore between contexts and answers
    from bert_score import score
    all_question_answers = [item['question'] + '\n' + item['answer'] for item in data]
    all_contexts = [' '.join(item['contexts']) for item in data]
    P, R, F1 = score(
        all_question_answers, 
        all_contexts, 
        lang="en", 
        verbose=True, 
        device=device,
        model_type='microsoft/deberta-large-mnli',
        batch_size=4
        )
    scores['precision_bertscore'] = P.mean().item()
    scores['recall_bertscore'] = R.mean().item()
    scores['f1_bertscore'] = F1.mean().item()


def score_rouge_12L(data, scores):
##compute rouge-l scores 
    from rouge_score import rouge_scorer
    all_question_answers = [item['question'] + '\n' + item['answer'] for item in data]
    all_contexts = [' '.join(item['contexts']) for item in data]
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_l_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    for ref, cand in zip(all_contexts, all_question_answers):
        score = scorer.score(ref, cand)
        rouge_l_scores.append(score['rougeL'].fmeasure)
        rouge_1_scores.append(score['rouge1'].fmeasure)
        rouge_2_scores.append(score['rouge2'].fmeasure)
    scores['avg_rougeL_score_combined'] = sum(rouge_l_scores)/len(rouge_l_scores)
    scores['avg_rouge1_score_combined'] = sum(rouge_1_scores)/len(rouge_1_scores)
    scores['avg_rouge2_score_combined'] = sum(rouge_2_scores)/len(rouge_2_scores)

def score_bleu(data, scores, device):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method4
    all_question_answers = [item['question'] + '\n' + item['answer'] for item in data]
    all_contexts = [' '.join(item['contexts']) for item in data]
    bleu_scores = []
    for ref, cand in zip(all_contexts, all_question_answers):
        reference = ref.split()
        candidate = cand.split()
        score = sentence_bleu([reference], candidate, smoothing_function=smoothie)
        bleu_scores.append(score)
    scores['avg_bleu_score'] = sum(bleu_scores)/len(bleu_scores)

def score_citation_accuracy(data, scores, device):
    from citation_accuracy_metric_calculator import CitationAccuracyMetricCalculator
    scorer = CitationAccuracyMetricCalculator(ground_truth_file='schema/sections.json')
    scorer.load()
    temp = {} 
    temp['citation_accuracy_precision'] = []
    temp['citation_accuracy_recall'] = []
    temp['citation_accuracy_f1'] = []
    for item in data:
        precision, recall, f1_score = scorer.compute_precision_recall_f1_single(item)
        temp['citation_accuracy_precision'].append(precision)
        temp['citation_accuracy_recall'].append(recall)
        temp['citation_accuracy_f1'].append(f1_score)

    scores['avg_citation_accuracy_precision'] = sum(temp['citation_accuracy_precision'])/len(temp['citation_accuracy_precision'])
    scores['avg_citation_accuracy_recall'] = sum(temp['citation_accuracy_recall'])/len(temp['citation_accuracy_recall'])
    scores['avg_citation_accuracy_f1'] = sum(temp['citation_accuracy_f1'])/len(temp['citation_accuracy_f1'])    
def scoring_function(data): 
    import torch

    scores = {}
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    score_entailment(data, scores, device)
    score_citation_accuracy(data, scores, device)
    score_cosine_similarity(data, scores, device)
    score_bertscore(data, scores, device)
    score_rouge_12L(data, scores)

    return scores

if __name__ == "__main__":

    analysts = []

    random_seed = 42

    test_size = 100 

    analysts.append(MyTestAnalyser(test_folder='output/EXPLAIN_QAR/qas.json', trim_data=False, test_set_size=test_size, random_seed=random_seed))
    analysts.append(MyTestAnalyser(test_folder='output/EXPLAIN_QAR_noEP/qas.json', trim_data=False, test_set_size=test_size, random_seed=random_seed))
    analysts.append(MyTestAnalyser(test_folder='output/RandomSeedSelectionQA/qas.json', trim_data=False, test_set_size=test_size, random_seed=random_seed))

    analysts.append(MyTestAnalyser(test_folder='output/DRQA/qas.json', trim_data=False, test_set_size=test_size, random_seed=random_seed))
    analysts.append(MyTestAnalyser(test_folder='output/RandomVertexQA/qas.json', trim_data=False, test_set_size=test_size, random_seed=random_seed))
    analysts.append(MyTestAnalyser(test_folder='output/SlidingWindowQA/qas.json', trim_data=False, test_set_size=test_size, random_seed=random_seed))

    for analyst in analysts:
        analyst.load()
        analyst.transform_and_trim()
        analyst.score_data(scoring_function)

    for analyst in analysts:
        analyst.print_statistics()