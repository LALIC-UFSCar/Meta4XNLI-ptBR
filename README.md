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

## Citation

If you use Meta4XNLI-ptBR in your research, please cite our LREC 2026 paper:

@inproceedings{johansson-etal-2026-meta4xnli,
  title = {Meta4XNLI-ptBR: Brazilian Portuguese Extension of Meta4XNLI Corpus},
  author = {Johansson, Karina and Assi, Fernanda and Silva, Isabella da and Passador, Rafael and Rodrigues, Isabela and Paes, Aline and Caseli, Helena},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {1668--1676},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/45566xcgz65x}
}
