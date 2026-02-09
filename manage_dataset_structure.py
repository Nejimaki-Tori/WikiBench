import json
from pathlib import Path
import re

JSON_TITLE = 'ruwikibench_articles.json'

def create_json():
    # CREATES JSON FROM DATASET
    base_dir = Path('.')
    articles_list = base_dir / 'small_articles_data.txt'
    html_dir = base_dir / 'Html'
    sources_dir = base_dir / 'Sources'
    sources_list_dir = base_dir / 'Downloaded_Sources_List'
    output_file = base_dir / JSON_TITLE

    with open(articles_list, 'r', encoding='utf-8') as f:
        article_titles = [line.strip() for line in f if line.strip()]

    dataset = []

    for title in sorted(article_titles):
        true_title = title
        title = re.sub(r'[<>:"/\\|?*]', '', title)
        entry = {'article_name': true_title, 'article_cleared_name': title}

        html_file = html_dir / f'{title}.html'
        with open(html_file, 'r', encoding='utf-8') as f:
            entry['html'] = f.read()

        sources_list_file = sources_list_dir / f'{title}.json'.replace(' ', '_')
        with open(sources_list_file, 'r', encoding='utf-8') as f:
            source_ids = json.load(f)
        print(f'Number of links for article {title}: {len(source_ids)}')
        article_sources_dir = sources_dir / title
        entry['sources'] = []

        source_files = sorted(
            article_sources_dir.glob('source_*.txt'),
            key=lambda x: int(x.stem.split('_')[1])
        )

        for idx, (source_id, source_file) in enumerate(zip(source_ids, source_files)):
            with open(source_file, 'r', encoding='utf-8') as f:
                source_text = f.read()

            entry['sources'].append({
                'source_id': source_id,
                'source_text': source_text,
                'file': source_file.name
            })

        dataset.append(entry)

        print(f'Processed: {title}')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f'Articles processed: {len(dataset)}')

def load_json():
    # CHECKS JSON DATA
    with open(JSON_TITLE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(len(data))
    i = 17
    print(data[i]['article_name'])
    print(data[i]['article_cleared_name'])
    print(data[i]['html'][:100])
    print(data[i]['sources'][10]['source_id'])
    print(data[i]['sources'][10]['source_text'][:100])
    print(data[i]['sources'][10]['file'])

def decompose_json():
    # DECOMPOSES JSON TO NORMAL FILE SYSTEM
    base_dir = Path('Articles')
    base_dir.mkdir(parents=True, exist_ok=True)
    json_file = JSON_TITLE

    html_dir = base_dir / 'Html'
    sources_dir = base_dir / 'Sources'
    sources_list_dir = base_dir / 'Downloaded_Sources_List'

    html_dir.mkdir(exist_ok=True)
    sources_dir.mkdir(exist_ok=True)
    sources_list_dir.mkdir(exist_ok=True)

    with open(json_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    article_titles = []

    for article in dataset:
        true_title = article['article_name']
        title = article['article_cleared_name']
        article_titles.append(true_title)

        html_file = html_dir / f'{title}.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(article['html'])

        source_ids = [source['source_id'] for source in article['sources']]
        sources_list_file = sources_list_dir / f'{title}.json'.replace(' ', '_')
        with open(sources_list_file, 'w', encoding='utf-8') as f:
            json.dump(source_ids, f, ensure_ascii=False, indent=2)

        article_sources_dir = sources_dir / title
        article_sources_dir.mkdir(exist_ok=True)

        for idx, source in enumerate(article['sources'], 1):
            source_file = article_sources_dir / source['file']
            with open(source_file, 'w', encoding='utf-8') as f:
                f.write(source['source_text'])

        print(f'Reinstalled: {title}')

    articles_list = 'small_articles_data.txt'
    with open(articles_list, 'w', encoding='utf-8') as f:
        f.write('\n'.join(article_titles))

    print(f'\nDone!')
    
