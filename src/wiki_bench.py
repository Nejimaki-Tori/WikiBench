import os
import re
from wiki_extract import Extracter
from wiki_gen import WikiGen
from openai_utils import LlmCompleter, AsyncList
from wiki_evaluater import WikiEvaluater
from wiki_agent import WikiAgent
from wiki_utils import WikiUtils
from tqdm import tqdm

#article_names = [
#    'Python',
#    'Летние Олимпийские игры 2024',
#    'Квантовый компьютер',
#    'Присоединение Крыма к Российской Федерации',
#    'Сколково (инновационный центр)',
#    'Tomb Raider (игра, 2013)',
#    'Чёрная дыра',
#    'Экономика США',
#    'Искусственный интеллект',
#    'COVID-19',
#    'Применение искусственного интеллекта',
#    'РИА Новости',
#    'Uncharted 4: A Thief’s End',
#    'Экономика КНР',
#    'Иннополис',
#    'Летние Олимпийские игры 2020',
#    'Солнечная система',
#    'C++',
#    'Си (язык программирования)',
#    'Сбербанк России',
#    'Чёрная смерть',
#    'Яндекс',
#    'Google (компания)',
#    'Большой адронный коллайдер',
#    'Геморрагическая лихорадка Эбола',
#    'Вирус иммунодефицита человека',
#    'Dota 2',
#    'TikTok',
#    'Марс',
#    'YouTube',
#    'Portal 2'
#]

class WikiBench:
    def __init__(self, url, key, model_name='llama3-70b', pre_load=False):
        self.client = LlmCompleter(api_address=url, api_key=key, model_name=model_name)
        self.env_prepared = pre_load
        self.model_name = model_name
        self.wiki_writer = WikiGen(self.client, self.model_name)
        with open('small_articles_data.txt', 'r', encoding='utf-8') as file:
            self.article_names = file.read().split('\n')
        self.wiki_utility = WikiUtils(False) if not self.env_prepared else None
        self.wiki_agent = WikiAgent(self.wiki_writer, True) if self.env_prepared else None
        self.wiki_evaluater = WikiEvaluater(self.wiki_agent.device, self.wiki_agent.encoder) if self.wiki_agent else None
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

    def prepare_env(self, texts_ready=True, emb_ready=True, ann_ready=True):
        if not texts_ready:
            self.prepare_texts()
        self.wiki_utility.load_corpus()
        if not emb_ready:
            for name in self.article_names:
                self.wiki_utility.get_embeddings(name, True)
        if not ann_ready:
            self.get_annotations()
        self.wiki_agent = WikiAgent(self.wiki_writer, True)
        self.wiki_evaluater = WikiEvaluater(self.wiki_agent.device, self.wiki_agent.encoder)
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
        topR = self.wiki_agent.get_number_of_snippets()
        
        if reference_mode: # получение заранее сгенерированных аннотаций
            texts = self.wiki_agent.get_annotations_from_disk() 
        else:              # генерация аннотаций оцениваемой моделью
            texts = await self.wiki_writer.get_annotations(topR)

        self.query_logger = []
        ndcg_s = 0
        pr_r_s = 0
        count = 0
        
        for annotation, (name, topK) in zip(texts, topR):
            print(name)
            count += 1
            ranked_docs = await self.wiki_agent.create_ranking(annotation, 3*topK, name)     
            ndcg, pr_r_score = self.wiki_evaluater.rank_query(ranked_docs, name)
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
                outline = await self.wiki_agent.create_outline(
                    name, 
                    mode=mode, 
                    page=page, 
                    neighbor_count=neighbor_count, 
                    description_mode=description_mode
                )
                p, r, f = self.wiki_evaluater.rank_outline(outline, page)
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
            sections = await self.wiki_agent.create_sections(page, self.model_name)
            pr, rec, f1 = self.wiki_evaluater.rank_sections(sections, page)
            self.article_gen_logger.append((pr, rec, f1))