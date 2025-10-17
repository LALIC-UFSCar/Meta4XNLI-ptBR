# Meta4XNLI-ptBR: Extension of Meta4XNLI for Brazilian Portuguese

Meta4XNLI is a parallel dataset for metaphor detection and interpretation for English and Spanish. Meta4XNLI is an extension of metaphor detection annotations for Brazilian Portuguese. The texts were automatically translated using LLMs and manually annotated.

##  Repository

This repository contains the scripts used to translate the texts to Brazilian Portuguese.

It contains the following scripts:
1. [prepare_datasets.py](prepare_datasets.py): script to download and preprocess the Meta4XNLI dataset.
2. [generate_sample.py](generate_sample.py): script to generate dataset sample.
3. [get_translations.py](get_translations.py): script to generate candidate translations.
4. [evaluate_translations.py](evaluate_translations.py): script to evaluate the candidate translations.
5. [select_best_translations.py](select_best_translations.py): script to select the best translation for each example.

### Scripts

This section describes in more details the scripts listed in the previous section.

#### 1. prepare_datasets.py

