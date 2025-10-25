# Meta4XNLI-ptBR: Brazilian Portuguese extension of Meta4XNLI corpus

Meta4XNLI is a parallel dataset for metaphor detection and interpretation for English and Spanish. Meta4XNLI is an extension of metaphor detection annotations for Brazilian Portuguese. The texts were automatically translated using LLMs and manually human annotated.

##  Repository

This repository contains the scripts used to translate the texts to Brazilian Portuguese, the annotation guidelines and the main result of our work: the corpus Meta4XNLI-ptBR.

It contains the following scripts:
1. [prepare_datasets.py](prepare_datasets.py): script to download and preprocess the Meta4XNLI dataset.
2. [generate_sample.py](generate_sample.py): script to generate dataset sample.
3. [get_translations.py](get_translations.py): script to generate candidate translations.
4. [evaluate_translations.py](evaluate_translations.py): script to evaluate the candidate translations.
5. [select_best_translations.py](select_best_translations.py): script to select the best translation for each example.

The annotation guidelines (that are written in Brazilian Portuguese), can be found in [documents/annotation_guidelines.pdf](documents/annotation_guidelines.pdf).

And the corpus Meta4XNLI-ptBR in JSONL format can be found in [data/Meta4XNLI-ptBR.jsonl](data/Meta4XNLI-ptBR.jsonl).
