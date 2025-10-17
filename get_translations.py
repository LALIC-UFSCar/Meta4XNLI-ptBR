"""Script to generate Meta4XNLI translations using a language model,
based on given prompts and configurations."""
import argparse
import time
import os
import re
import sys
from pathlib import Path

import jsonlines
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from tqdm import tqdm

from utils.data_processing import filter_unprocessed_records
from utils.io import parse_yaml
from utils.llm_request import get_answer, fill_placeholders


def spans_to_text(spans: list) -> str:
    """Converts a list of text spans into a single formatted string.

    Args:
        spans (list): A list of text spans.

    Returns:
        str: A formatted string representing the spans.
    """
    pattern = re.compile(r'[,.!?]$')
    spans = [pattern.sub('', span) for span in spans]

    if len(spans) == 0:
        return ''
    if len(spans) == 1:
        return f'"{spans[0]}"'
    if len(spans) == 2:
        return f'"{spans[0]}" and "{spans[1]}"'
    return ', '.join(f'"{span}"' for span in spans[:-1])+ f' and "{spans[-1]}"'


def get_prompt(main_prompt: str, extra_prompt: str, row: pd.Series) -> str:
    """Generates a prompt for the language model.

    Args:
        main_prompt (str): main prompt template with placeholders
        extra_prompt (str): additional prompt template with placeholders
        row (pd.Series): row of data to fill in the placeholders

    Returns:
        str: final prompt for the language model
    """
    main_prompt = fill_placeholders(main_prompt, row.to_dict())
    if extra_prompt is None:
        return main_prompt
    extra_prompt = fill_placeholders(extra_prompt, row.to_dict())
    if row['has_metaphor']:
        return main_prompt
    return extra_prompt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=Path, required=True,
                        help='Path to dataset in .jsonl format.')
    parser.add_argument('-s', '--system', type=Path, required=True,
                        help='Path to system prompt file. If \
                        `additional_system` is passed, this parameter \
                        represents the system prompt for METAPHORICAL \
                        examples. Otherwise, it represents the system prompt \
                        for all examples.')
    parser.add_argument('-a', '--additional_system', type=Path, required=False,
                        help='If passed, represents the system prompt for \
                        LITERAL examples.')
    parser.add_argument('-u', '--user', type=Path, required=True,
                        help='Path to user prompt file. It can contain \
                        placeholders, indicated between braces.')
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help='Request configurations related to model, \
                        temperature, etc.')
    parser.add_argument('-ca', '--additional_config', type=Path, default=None,
                        help='Additional request configurations related to \
                            model, temperature, etc. If passed, represents \
                            the config for LITERAL examples.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name.')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Path to store the generations.')
    parser.add_argument('-z', '--sleep', type=int, required=False, default=2,
                        help='Seconds to wait after each request, to \
                            overcome the limit of requests per minute')
    parser.add_argument('-S', '--sample_size', type=int, required=False,
                        help='Size of head sample of the dataset to run.')
    parser.add_argument('-C', '--client', type=str,
                        choices=['openai', 'groq', 'maritalk'],
                        help='Define the client.', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    load_dotenv()

    main_system_prompt = args.system.read_text(encoding='utf-8')
    if args.additional_system is not None:
        extra_system_prompt=args.additional_system.read_text(encoding='utf-8')
    else:
        extra_system_prompt = None
    user_prompt = args.user.read_text(encoding='utf-8')

    main_config = parse_yaml(args.config)
    if args.additional_config is not None:
        additional_config = parse_yaml(args.additional_config)
    else:
        additional_config = None

    if args.client == 'groq':
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    elif args.client == 'openai':
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    else:
        client = OpenAI(api_key=os.getenv('MARITALK_API_KEY'),
                        base_url="https://chat.maritaca.ai/api")

    df = pd.read_json(args.dataset, orient='records', lines=True,
                      encoding='utf-8')

    df = filter_unprocessed_records(df, args.output)

    if len(df) == 0:
        print('No unprocessed records found in the dataset.')
        sys.exit()

    if args.sample_size and len(df) > args.sample_size:
        df = df.head(args.sample_size)

    df['user_prompt'] = [fill_placeholders(user_prompt, record) for record in \
                         df.to_dict(orient='records')]

    if '{metaphorical_spans_text}' in main_system_prompt \
        or '{metaphorical_spans_text}' in extra_system_prompt:
        df['metaphorical_spans_text'] = df.metaphorical_spans.\
            apply(spans_to_text)

    file = open(args.output, 'a+')
    with jsonlines.Writer(file) as writer:
        for _, row in tqdm(df.iterrows(),total=len(df),desc='Generating text'):
            if additional_config is None:
                config = main_config
            elif row['has_metaphor']:
                config = main_config
            else:
                config = additional_config
            system_prompt = get_prompt(main_system_prompt,extra_system_prompt,row)
            answer = get_answer(client, system_prompt, row['user_prompt'],
                                config, args.model)
            record = {'id': row['id'], 'text': answer}
            writer.write(record)
            time.sleep(args.sleep)

    writer.close()


if __name__ == '__main__':
    main()
