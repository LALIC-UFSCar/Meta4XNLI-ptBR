"""
Script to select the best translations from a set of candidates.
It receives a source dataset and a set of candidate translations and their scores.
For each example, it selects the translation with the highest score.
It saves the information of the selected translations.
"""
import argparse
import logging
import json
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_best_translations(data: dict) -> list:
    # Choose the best candidate(s) for each example
    selected_translations = []
    for id, example in data.items():
        candidates = example['candidates']
        # Find the highest score
        max_score = max(c['score'] for c in candidates.values())
        # Find all candidates with the highest score
        best_candidates = [name for name, c in candidates.items() if c['score'] == max_score]
        # Use the first candidate's translation as representative (or you can choose another logic)
        best_translation = candidates[best_candidates[0]]['translation']
        selected_translations.append({
            'id': id,
            'text': best_translation,
            'score': max_score,
            'candidate': best_candidates if len(best_candidates) > 1 else best_candidates[0]
        })
    return selected_translations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', type=Path, required=True,
                        help='Path to the source dataset file.')
    parser.add_argument('--candidates_dataset_dir', type=Path, required=True,
                        help='Path to the candidates dataset directory, that contains .jsonl files with translations.')
    parser.add_argument('--candidates_score_dir', type=Path, required=True,
                        help='Path to the candidates score directories, that contains .tsv files with scores.')
    parser.add_argument('--candidates_name', type=str, nargs='+', required=True,
                        help='Names of the candidate datasets, must the file names of the .jsonl and .tsv files.')
    parser.add_argument('--output_file', type=Path, required=True, help='Path to the .jsonl output file. \
                        It has keys "id", "text", "score", and "candidate". "text" is the chosen translation.')
    parser.add_argument('--metric', type=str, default='comet_no_ref',
                        help='Evaluation metric to use for scoring candidates.')
    parser.add_argument('--dict', type=Path, required=False, default=None,
                        help='Path to export the dict data for debugging.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the source dataset
    source_df = pd.read_json(args.source_file, orient='records', lines=True, encoding='utf-8')

    data = {}
    for _, row in source_df.iterrows():
        data[row['id']] = {
            'source': row['text'],
            'candidates': {}
        }

    for candidate_name in args.candidates_name:
        
        candidate_dataset_path = args.candidates_dataset_dir / f"{candidate_name}.jsonl"
        candidate_score_path = args.candidates_score_dir / f"{candidate_name}.tsv"

        if not candidate_dataset_path.is_file():
            logger.info(f"Candidate dataset not found: {candidate_dataset_path}")
            continue

        if not candidate_score_path.is_file():
            logger.info(f"Candidate score file not found: {candidate_score_path}")
            continue

        # Load the candidate translation
        candidate_df = pd.read_json(candidate_dataset_path, lines=True, encoding='utf-8')
        # Load the candidate scores
        score_df = pd.read_csv(candidate_score_path, sep='\t')

        for id in data.keys():
            # Get the translation and score for the current example
            translation = candidate_df[candidate_df.id.eq(id)].text.values[0]
            score = score_df[score_df.id.eq(id)][args.metric].values[0]

            # Store the candidate translation and score
            data[id]['candidates'][candidate_name] = {
                'translation': translation,
                'score': score
            }

    # Save the selected translations to the output file
    selected_translations = select_best_translations(data)
    output_df = pd.DataFrame(selected_translations)
    output_df.to_json(args.output_file, lines=True, orient='records', index=False, force_ascii=False)

    # generate stats of scores
    print('Distribution of scores of the selected translations')
    print(output_df.score.describe())
    num_scores_one = len(output_df[output_df.score.eq(1)])
    print(f'Number of maximum scores: {num_scores_one} ({num_scores_one/len(output_df):.2%}%)')
    print('-' * 30)

    if args.dict is not None:
        with open(args.dict, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
