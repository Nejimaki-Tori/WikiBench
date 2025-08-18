import os
import re
from wiki_gen import WikiGen
from openai_utils import LlmCompleter, AsyncList
from wiki_evaluater import WikiEvaluater
from wiki_agent import WikiAgent
from wiki_utils import WikiUtils

class WikiBench:
    def __init__(self, url, key, model_name='llama3-70b', device='cpu', encoder=None, pre_load=False):
        self.client = LlmCompleter(api_address=url, api_key=key, model_name=model_name)
        self.env_prepared = pre_load
        self.model_name = model_name
        self.device = device
        self.encoder = encoder
        self.wiki_writer = WikiGen(self.client, self.model_name)
        self.evaluater = LlmCompleter(api_address=url, api_key=key, model_name='Qwen3-235B-A22B-Instruct-2507')
        with open('small_articles_data.txt', 'r', encoding='utf-8') as file:
            self.article_names = file.read().split('\n')
        self.wiki_utility = WikiUtils(device=self.device, encoder=self.encoder, pre_load=False, evaluater=self.evaluater) if not self.env_prepared else None
        self.wiki_agent = WikiAgent(self.wiki_writer, self.device, self.encoder, True, self.evaluater) if self.env_prepared else None
        self.wiki_evaluater = WikiEvaluater(self.wiki_agent.device, self.wiki_agent.encoder, self.evaluater) if self.wiki_agent else None
        self.query_logger = []
        self.outline_logger = []
        self.article_gen_logger = []
        self.stream_results = False

    def prepare_texts(self):
        for name in self.article_names:
            self.wiki_utility.get_article(name, retrieve_sources=True, verbose=True)

    async def prepare_env(self, texts_ready=True, window_size=300, emb_ready=True, ann_ready=True):
        if not texts_ready:
            self.prepare_texts()
        self.wiki_utility.load_corpus(window_size=window_size)
        if not emb_ready:
            for name in self.article_names:
                self.wiki_utility.get_embeddings(name, True)
        if not ann_ready:
            await self.get_annotations()
        self.wiki_agent = WikiAgent(self.wiki_writer, self.device, self.encoder, True)
        self.wiki_evaluater = WikiEvaluater(self.wiki_agent.device, self.wiki_agent.encoder)
        self.env_prepared = True

    async def get_annotations(self):
        annotations = AsyncList()
        pages = [self.wiki_agent.get_article(
            name, 
            retrieve_sources=False, 
            is_downloaded=True, 
            verbose=False,
            html=True,
            needs_saving=False
        ) for name in self.article_names]
        
        for page in pages:
            annotations.append(self.wiki_writer.get_ref_subqueries(page))
            annotations.append(self.wiki_writer.get_ref_subqueries(page, 1))
        
        await annotations.complete_couroutines(batch_size=40)
        annotations = await annotations.to_list()
        dir_name = os.path.join("Generation", "Annotations")
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
        
        for annotation, (name, topK) in zip(texts, topR):
            print(name)
            ranked_docs = await self.wiki_agent.create_ranking(annotation, 3*topK, name)
            ndcg, pr_r_score = self.wiki_evaluater.rank_query(ranked_docs, name)
            self.query_logger.append((ndcg, pr_r_score))
        return self.wiki_evaluater.mean_value(query_logger)

    async def query_stream(self, reference_mode=1):
        topR  = self.wiki_agent.get_number_of_snippets()
        
        if reference_mode: # получение заранее сгенерированных аннотаций
            texts = self.wiki_agent.get_annotations_from_disk() 
        else:              # генерация аннотаций оцениваемой моделью
            texts = await self.wiki_writer.get_annotations(topR)
    
        for annotation, (name, topK) in zip(texts, topR):
            #print(name)
            ranked = await self.wiki_agent.create_ranking(annotation, 3 * topK, name)
            yield name, ranked, annotation  
    

    async def rank_outline(self, neighbor_count=0, description_mode=0, mode=1):
        self.outline_logger = []
        for name in self.article_names:
            #print(name)
            page = self.wiki_agent.get_article(
                name, 
                retrieve_sources=False, 
                is_downloaded=True, 
                verbose=False,
                html=True,
                needs_saving=False
            )
            outline = await self.wiki_agent.create_outline(
                name, 
                mode=mode, 
                page=page, 
                neighbor_count=neighbor_count, 
                description_mode=description_mode
            )
            p, r, f = self.wiki_evaluater.rank_outline(outline, page)
            self.outline_logger.append((p, r, f))
                
        return self.wiki_evaluater.calc(self.outline_logger, is_flat=True)

    async def outline_stream(self, neighbor_count=0, description_mode=0, mode=1):
        for name in self.article_names:
            #print(name)
            page = self.wiki_agent.get_article(
                name, 
                retrieve_sources=False, 
                is_downloaded=True, 
                verbose=False,
                html=True,
                needs_saving=False
            )
            outline = await self.wiki_agent.create_outline(
                name, 
                mode=mode, 
                page=page, 
                neighbor_count=neighbor_count, 
                description_mode=description_mode
            )
            yield name, outline

    async def rank_sections(self):
        self.article_gen_logger = []
        for name in self.article_names:
            #print(name)
            page = self.wiki_agent.get_article(name, False, True)
            sections = await self.wiki_agent.create_sections(page)
            pr, rec, f1, qa_cov, qa_sim = self.wiki_evaluater.rank_sections(sections, page, self.model_name)
            self.article_gen_logger.append((pr, rec, f1, qa_cov, qa_sim))
         
        return self.wiki_evaluater.calc(self.article_gen_logger, is_flat=False)

    async def sections_stream(self):
        for name in self.article_names:
            #print(name)
            page = self.wiki_agent.get_article(name, False, True)
            sections = await self.wiki_agent.create_sections(page)
            yield name, sections