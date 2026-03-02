import sys
import csv
import json
import torch
import asyncio
import logging
import argparse
from pathlib import Path
import time
from sentence_transformers import SentenceTransformer

def get_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

repo_root = get_repo_root()
src_dir = repo_root / 'src'

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from wiki_bench import WikiBench


def resolve_device(device_arg: str):
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_encoder(encoder_name: str, device):
    logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.ERROR)
    return SentenceTransformer(encoder_name).to(device)


def save_config(path, config: dict):
    with path.open('w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def save_metrics_json(path, metrics: dict):
    with path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def pack_query_metrics(result):
    try:
        return {
            'ndcg_mean': float(result[0]),
            'r_precision_mean': float(result[1]),
        }
    except Exception:
        return {'raw': result}


def pack_outline_metrics(result):
    try:
        return {
            'precision': {
                'mean': float(result[0]),
                'ci_low': float(result[1]),
                'ci_high': float(result[2]),
            },
            'recall': {
                'mean': float(result[3]),
                'ci_low': float(result[4]),
                'ci_high': float(result[5]),
            },
            'f1': {
                'mean': float(result[6]),
                'ci_low': float(result[7]),
                'ci_high': float(result[8]),
            },
        }
    except Exception:
        return {'raw': result}


def pack_sections_metrics(result):
    try:
        return {
            'precision': {
                'mean': float(result[0]),
                'ci_low': float(result[1]),
                'ci_high': float(result[2]),
            },
            'recall': {
                'mean': float(result[3]),
                'ci_low': float(result[4]),
                'ci_high': float(result[5]),
            },
            'f1': {
                'mean': float(result[6]),
                'ci_low': float(result[7]),
                'ci_high': float(result[8]),
            },
            'rouge_l': {
                'mean': float(result[9]),
                'ci_low': float(result[10]),
                'ci_high': float(result[11]),
            },
            'bleu': {
                'mean': float(result[12]),
                'ci_low': float(result[13]),
                'ci_high': float(result[14]),
            },
        }
    except Exception:
        return {'raw': result}


def flatten_metrics(metrics: dict) -> dict:
    flat = {
        'model_name': metrics['model_name'],
        'number_of_articles': metrics['number_of_articles'],
    }

    ranking = metrics.get('ranking', {})
    flat['ranking_ndcg_mean'] = ranking.get('ndcg_mean')
    flat['ranking_r_precision_mean'] = ranking.get('r_precision_mean')

    outline = metrics.get('outline', {})
    for key in ('precision', 'recall', 'f1'):
        if key in outline:
            flat[f'outline_{key}_mean'] = outline[key].get('mean')
            flat[f'outline_{key}_ci_low'] = outline[key].get('ci_low')
            flat[f'outline_{key}_ci_high'] = outline[key].get('ci_high')

    sections = metrics.get('sections', {})
    for key in ('precision', 'recall', 'f1', 'rouge_l', 'bleu'):
        if key in sections:
            flat[f'sections_{key}_mean'] = sections[key].get('mean')
            flat[f'sections_{key}_ci_low'] = sections[key].get('ci_low')
            flat[f'sections_{key}_ci_high'] = sections[key].get('ci_high')

    return flat


def save_metrics_csv(path, metrics: dict):
    row = flatten_metrics(metrics)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


async def run_wiki_benchmark(
    *,
    api: str,
    key: str,
    model_name: str,
    concurrency: int,
    output_dir: str,
    number_of_articles: int = 5,
    encoder_name: str = 'sergeyzh/BERTA',
    device: str = 'auto',
    prepare_env: bool = False,
    neighbor_count: int = 0,
    description_mode: bool = True,
    clusterization_with_hint: bool = True,
    shared_encoder=None,
    shared_device=None,
):
    repo_root = get_repo_root()

    device_obj = shared_device if shared_device is not None else resolve_device(device)
    encoder = shared_encoder if shared_encoder is not None else build_encoder(encoder_name, device_obj)

    output_dir_path = Path(output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = repo_root / output_dir_path
    output_dir_path = output_dir_path.resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model_safe_name = model_name.replace('/', '_').replace(' ', '_')
    model_dir = output_dir_path / model_safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    save_config(model_dir / 'config.json', {
        'api': api,
        'model_name': model_name,
        'concurrency': concurrency,
        'number_of_articles': number_of_articles,
        'encoder_name': encoder_name,
        'device': str(device_obj),
        'prepare_env': prepare_env,
        'neighbor_count': neighbor_count,
        'description_mode': description_mode,
        'clusterization_with_hint': clusterization_with_hint,
        'output_dir': str(output_dir_path),
    })

    bench = WikiBench(
        key=key,
        url=api,
        model_name=model_name,
        model_safe_name=model_safe_name,
        device=device_obj,
        encoder=encoder,
        number_of_articles=number_of_articles,
        output_dir=str(output_dir_path),
        concurrency=concurrency,
    )

    if prepare_env:
        bench.prepare_env()
    else:
        bench.load_enviroment()  # сохраняю имя метода как в твоём коде

    start = time.perf_counter()
    score_query = await bench.rank_query()
    end = time.perf_counter()
    query_time = end - start
    
    start = time.perf_counter()
    score_outline = await bench.rank_outline(
        neighbor_count=neighbor_count,
        description_mode=description_mode,
        clusterization_with_hint=clusterization_with_hint,
    )
    end = time.perf_counter()
    outline_time = end - start

    start = time.perf_counter()
    score_sections = await bench.rank_sections()
    end = time.perf_counter()
    sections_time = end - start

    metrics = {
        'model_name': model_name,
        'number_of_articles': number_of_articles,
        'ranking': pack_query_metrics(score_query),
        'ranking_time': query_time,
        'outline': pack_outline_metrics(score_outline),
        'outline_time': outline_time,
        'sections': pack_sections_metrics(score_sections),
        'sections_time': sections_time
    }

    save_metrics_json(model_dir / 'metrics.json', metrics)
    save_metrics_csv(model_dir / 'metrics.csv', metrics)

    return metrics


async def run_many_wiki_benchmarks(
    model_names,
    api: str,
    key: str,
    concurrency: int,
    output_dir: str,
    number_of_articles: int = 5,
    encoder_name: str = 'sergeyzh/BERTA',
    device: str = 'auto',
    prepare_env: bool = False,
    neighbor_count: int = 0,
    description_mode: bool = True,
    clusterization_with_hint: bool = True,
):
    rows = []

    shared_device = resolve_device(device)
    shared_encoder = build_encoder(encoder_name, shared_device)

    for model_name in model_names:
        metrics = await run_wiki_benchmark(
            api=api,
            key=key,
            model_name=model_name,
            concurrency=concurrency,
            output_dir=output_dir,
            number_of_articles=number_of_articles,
            encoder_name=encoder_name,
            device=device,
            prepare_env=prepare_env,
            neighbor_count=neighbor_count,
            description_mode=description_mode,
            clusterization_with_hint=clusterization_with_hint,
            shared_encoder=shared_encoder,
            shared_device=shared_device,
        )
        rows.append(flatten_metrics(metrics))

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='RuWikiBench', description='Run WikiBench benchmark')

    parser.add_argument('--api', required=True, help='LLM API base URL')
    parser.add_argument('--key', required=True, help='LLM API key')
    parser.add_argument('--model-name', required=True, help='Model name for API calls')
    parser.add_argument('--concurrency', type=int, required=True, help='Async concurrency')
    parser.add_argument('--output-dir', required=True, help='Directory for outputs')

    parser.add_argument('--number-of-articles', type=int, default=5)
    parser.add_argument('--encoder-name', default='sergeyzh/BERTA')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])

    parser.add_argument('--prepare-env', action='store_true')

    parser.add_argument('--neighbor-count', type=int, default=0)
    parser.add_argument(
        '--no-description-mode',
        action='store_false',
        dest='description_mode',
        help='Disable description mode in cluster summarization'
    )
    parser.set_defaults(description_mode=True)

    parser.add_argument(
        '--no-clusterization-with-hint',
        action='store_false',
        dest='clusterization_with_hint',
        help='Disable clusterization with hint'
    )
    parser.set_defaults(clusterization_with_hint=True)

    return parser


async def run(args):
    return await run_wiki_benchmark(
        api=args.api,
        key=args.key,
        model_name=args.model_name,
        concurrency=args.concurrency,
        output_dir=args.output_dir,
        number_of_articles=args.number_of_articles,
        encoder_name=args.encoder_name,
        device=args.device,
        prepare_env=args.prepare_env,
        neighbor_count=args.neighbor_count,
        description_mode=args.description_mode,
        clusterization_with_hint=args.clusterization_with_hint,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
