<h1>A Framework for QA Generation in a Regulatory Context</h1>

Note: API Keys must be stored in a folder ./keys in the root directory and must be labelled in accordance to the convention laid out in model_dict.py

We use `SpAcy` for Entity Pathing extraction of lemmatized noun phrases, please install its model with:
```
python3 -m spacy download en_core_web_sm
```

The Knowledge Graph is stored in a json file format under `./data` and is comprised of three files `reg_relation`, `regs` and `rel_type` which outline the graphs edge list, vertices and edge types respectively.

Outputs generated are stored in `./outputs` folder 

There are two main files to run the pipelines 
`src/run.py` is used to run our approach as well as each of the baselines and ablations. baselines and ablations will accept an input filepath which points to the seeds selected by the seed vertex selection strategy in our approach, for consistency. 

`src/test_analyse.py` is used to evaluate generated qa datasets and compute required metrics. 
