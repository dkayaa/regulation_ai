import queue 
import spacy
import re 
# Entity Pathing Methodology
# Given a set of contexts c = [c1, c2, c3, c4]
# 1. produce bipartite graph B as described in the methdology 
# 2. Iteratively compute shortest paths between entity pairs 
# 3. Filter and return paths that traverse all contexts

class EntityPathExtractor: 

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def get_entity_paths(self, contexts): 
        paths = self.get_paths(contexts)
        paths = [[i[0] for i in p] for p in paths] 
        return paths 

    def get_paths(self, contexts, filter='entity_node'): 
        b = self.contexts2bgraph(contexts)
        entities = [i for i in self.get_vertices(b) if i[1] == 'entity_node']
        paths = []
        for e1 in entities: 
            for e2 in entities: 
                if e1 == e2: 
                    continue 
                p = self.bfs(b, e1, e2) 
                pc = [i for i in p if i[1] == 'context_node'] #contexts traversed
                if len(pc) != len(contexts): 
                    continue 
                paths.append(p) 
        
        if filter != "": 
            paths = [[i for i in p if i[1] == filter] for p in paths]
        return paths 


    def contexts2bgraph(self, contexts):
        graph = {} 
        for (i, c) in enumerate(contexts): 
            context_node = 'context_node_{0}'.format(i)
            graph[(context_node, 'context_node')] = [(i, 'entity_node') for i in list(self.extract_keywords(c))]
            for n in graph[(context_node, 'context_node')]: 
                if n not in graph.keys(): 
                    graph[n] = [] 
                graph[n].append((context_node, 'context_node'))
        
        return graph

    def get_vertices(self, graph): 
        return list(graph.keys())

    def get_neighbours(self, graph, u): 
        #graph is a adjacency list representation 
        #dict[u] = [v1, v2, v3 ...]
        return graph[u]

    def bfs(self, graph, u, v): 
        #compute bfs from u -> v 
        found = False
        q = queue.Queue() 

        q.put(u) 
        bt = {} #back tracking
        visited = {} #visited dict 
        while not q.empty(): 
            c = q.get()
            if c == v:
                found = True
                break 
            visited[c] = True 
            for ch in self.get_neighbours(graph, c): 
                if ch not in visited.keys():
                    q.put(ch) 
                    bt[ch] = c 
        # Extract list
        path = [] 
        if found: 
            path = [v]
            c = v 
            while c != u: 
                path.append(bt[c])
                c = bt[c]
            
            path = path[::-1] 

        return path 

    def clean_phrase(self, phrase):
        phrase = phrase.replace('\n', ' ') #remove new lines
        phrase = re.sub(r'\([a-zA-Z0-9]+\)', '', phrase) #remove section numbers
        phrase = re.sub(r'\d+', '', phrase) # remove numbers
        phrase = re.sub(r'^(the|a|an)\s+', '', phrase, flags=re.IGNORECASE)
        phrase = re.sub(r'[^\w\s-]', '', phrase).strip()
        return phrase

    def extract_keywords(self, text):
        doc = self.nlp(text)

        phrases = []
        for chunk in doc.noun_chunks:
            # singularisation on to the nouns
            lemmatized_phrase = ' '.join([token.lemma_ if token.pos_ == 'NOUN' else token.text for token in chunk])
            phrases.append(lemmatized_phrase)

        # stopwords set for words to ignore
        stopwords = {'subject','section', 'subsection', 'paragraph', 'subparagraph', 'schedule'}
        stopwords = self.nlp.Defaults.stop_words.union(stopwords)
        th_word_pattern = re.compile(r'\bth\w*\b', re.IGNORECASE)

        cleaned_phrases = []
        for phrase in phrases:
            cleaned = self.clean_phrase(phrase)
            words = cleaned.lower().split()
            if cleaned and all(w not in stopwords and not th_word_pattern.match(w) for w in words):
                cleaned_phrases.append(cleaned)

        # basic filtering out of any phrases < 3 characters long
        cleaned_phrases = [i for i in cleaned_phrases if len(i) >= 3]

        return set(cleaned_phrases)