import os
import re
from wiki_extract import Extracter
from wiki_gen import WikiGen
from openai_utils import LlmCompleter, AsyncList
from wiki_metrics import WikiEval
from wiki_utils import WikiUtils
from tqdm import tqdm

article_names = [
    'Python',
    'Летние Олимпийские игры 2024',
    'Квантовый компьютер',
    'Присоединение Крыма к Российской Федерации',
    'Сколково (инновационный центр)',
    'Tomb Raider (игра, 2013)',
    'Чёрная дыра',
    'Экономика США',
    'Искусственный интеллект',
    'COVID-19',
    'Применение искусственного интеллекта',
    'РИА Новости',
    'Uncharted 4: A Thief’s End',
    'Экономика КНР',
    'Иннополис',
    'Летние Олимпийские игры 2020',
    'Солнечная система',
    'C++',
    'Си (язык программирования)',
    'Сбербанк России',
    'Чёрная смерть',
    'Яндекс',
    'Google (компания)',
    'Большой адронный коллайдер',
    'Геморрагическая лихорадка Эбола',
    'Вирус иммунодефицита человека',
    'Dota 2',
    'TikTok',
    'Марс',
    'YouTube',
    'Portal 2'
]

class WikiBench:
    def __init__(self, url, key, model_name='llama3-70b', pre_load=False):
        self.client = LlmCompleter(api_address=url, api_key=key, model_name=model_name)
        self.env_prepared = pre_load
        self.wiki_writer = WikiGen(self.client)
        self.article_names = article_names
        self.model_name = model_name
        self.wiki_utility = WikiUtils(False) if not self.env_prepared else None
        self.wiki_evaluater = WikiEval(self.wiki_writer, True) if self.env_prepared else None
        self.query_logger = []
        self.outline_logger = []
        self.article_gen_logger = []
        self.retry_attempts = 5

    def get_article(self, name, retrieve_sources=False, is_downloaded=False, verbose=False, html=True, needs_saving=True):
        if verbose:
            print('Article name: ', name)
        page = Extracter(name, is_downloaded, verbose, html, needs_saving)
        page.get_references()
        page.get_outline()
        if retrieve_sources and not is_downloaded:
            page.fast_extracter()
        if is_downloaded or retrieve_sources:
            page.get_filtered_outline()
        return page

    def prepare_texts(self):
        for name in self.article_names:
            self.get_article(name, retrieve_sources=True, verbose=True)

    def prepare_env(self, texts_ready=True):
        if not texts_ready:
            self.prepare_texts()
        self.wiki_utility.load_corpus()
        for name in self.article_names:
            self.wiki_utility.get_embeddings(name, True)
        self.get_annotations()
        self.wiki_evaluater = WikiEval(self.wiki_writer, True)
        self.env_prepared = True

    async def get_annotations(self):
        annotations = AsyncList()
        pages = [self.get_article(name) for name in self.article_names]
        
        for page in pages:
            annotations.append(self.wiki_writer.get_ref_subqueries(page))
            annotations.append(self.wiki_writer.get_ref_subqueries(page, 1))
        
        await annotations.complete_couroutines(batch_size=40)
        annotations = await annotations.to_list()
        dir_name = os.path.join("Generation", "Subqueries_ref", "Annotations")
        os.makedirs(dir_name, exist_ok=True)
        for i in range(0, len(annotations), 2):
            name = re.sub(r'[<>:"/\\|?*]', '', annotations[i][0])
            file_name = os.path.join(dir_name, name + '.txt')
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(annotations[i][1] + "\n")
                f.write(annotations[i + 1][1]) 

        
    async def rank_query(self, reference_mode=1):
        if reference_mode:
            texts = []
            for subdir, _, files in os.walk(r'Generation\Subqueries_ref\Annotations'):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(subdir, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                            texts.append(text)
        
        base_dir = os.path.join('Articles', 'Sources')
        
        folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        
        snippets = self.wiki_evaluater.get_texts_from_disk()
        topR = []
        for folder in folders:
            snip = [1 if snippet[0] == folder else 0 for snippet in snippets.keys()]
            topR.append((folder, sum(snip)))
            
        self.query_logger = []
        ndcg_s = 0
        pr_r_s = 0
        count = 0
        if not reference_mode:
            annotations = AsyncList()
            for name, _ in topR:
                annotations.append(self.wiki_writer.get_ref_subqueries(None, 0, 0, name))
                annotations.append(self.wiki_writer.get_ref_subqueries(None, 1, 0, name))
        
            await annotations.complete_couroutines(batch_size=40)
            annotations = await annotations.to_list()
            new_texts = annotations
            texts = []
            for i in range(0, len(new_texts), 2):
                new_text = new_texts[i] + '\n' + new_texts[i + 1]
                texts.append(new_text)
            
        for annotation, (name, topK) in zip(texts, topR):
            print(name)
            count += 1
            ndcg, pr_r_score = await self.wiki_evaluater.rank_one_query(annotation, 3*topK, name)
            print(ndcg)
            print(pr_r_score)
            print()
            self.query_logger.append((ndcg, pr_r_score))
            ndcg_s += ndcg
            pr_r_s += pr_r_score
        return ndcg_s / count, pr_r_s / count

    async def rank_outline(self, neighbor_count=0, description_mode=0, mode=1):
        p_s = 0
        r_s = 0
        f_s = 0
        count = 0
        self.outline_logger = []
        for name in self.article_names:
            print(name)
            count += 1
            page = self.get_article(name, is_downloaded=True)
            try:
                p, r, f = await self.wiki_evaluater.rank_outline(name, mode=mode, page=page, neighbor_count=neighbor_count, description_mode=description_mode)
                print('Pr: ', p, ' rec: ', r, ' f: ', f)
                print()
                self.outline_logger.append((p, r, f))
                p_s += p
                r_s += r
                f_s += f
            except:
                print('error while generating')
                continue
        return p_s / count, r_s / count, f_s / count

    async def rank_sections(self):
        self.article_gen_logger = []
        for name in self.article_names:
            print(name)
            page = self.get_article(name, False, True)
            pr, rec, f1 = await self.wiki_evaluater.rank_sections(page, self.model_name)
            self.article_gen_logger.append((pr, rec, f1))