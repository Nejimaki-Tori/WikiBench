import logging
from tqdm.asyncio import tqdm
from wiki_gen import WikiGen
from openai_utils import LlmCompleter, AsyncList
from wiki_evaluater import WikiEvaluater
from wiki_agent import WikiAgent
from wiki_utils import WikiUtils

class WikiBench:
    '''Main class for runnig benchmark pipeline'''
    def __init__(
        self, 
        url: str, 
        key: str, 
        model_name: str, 
        device, 
        encoder,
        log_level=logging.INFO
    ):
        self.model_name = model_name
        self.device = device
        self.encoder = encoder
        self.client = LlmCompleter(api_address=url, api_key=key, model_name=model_name)
        self.wiki_writer = WikiGen(self.client, self.model_name)
        self.logger = self._setup_logger(level=log_level)
        with open('small_articles_data.txt', 'r', encoding='utf-8') as file:
            self.article_names = [x for x in file.read().split('\n') if x.strip()]
        self.wiki_utility = WikiUtils(device=self.device, encoder=self.encoder)
        self.wiki_agent = WikiAgent(utils=self.wiki_utility, client=self.wiki_writer)
        self.is_env_prepared = False
        self.wiki_evaluater = WikiEvaluater(self.wiki_agent.device, self.wiki_agent.encoder)
        self.query_logger = []
        self.outline_logger = []
        self.article_gen_logger = []
        self.stream_results = False
        self.logger.info(f'WikiBench initialized: model={self.model_name}, articles={len(self.article_names}')

    def _setup_logger(self, level):
        logger = logging.getLogger("wikibench")
        logger.setLevel(level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(fmt)
            logger.addHandler(handler)

        logger.propagate = False
        return logger

    def prepare_env(self, are_texts_ready=True, window_size=600, overlap=0):
        self.logger.info(f'Preparing env: are_texts_ready={are_texts_ready}, window={window_size}, overlap={overlap}')
        if self.is_env_prepared:
            self.logger.info('Env already prepared. Loading created corpus')
            self.wiki_utility.load_created_corpus()
            self.wiki_agent.utils = self.wiki_utility
            return
            
        if not are_texts_ready:
            self.logger.info('Creating article corpus from scratch...')
            self.wiki_utility.create_article_corpus_from_scratch(article_bare_names=self.article_names)

        self.logger.info('Creating main corpus from scratch...')
        self.wiki_utility.create_corpus_from_scratch(article_names=self.article_names, window_size=window_size, overlap=overlap)
        self.wiki_agent.utils = self.wiki_utility
        self.is_env_prepared = True
        self.logger.info("Enviroment prepared!")
        
    async def rank_query(self):  
        self.logger.info('Stage: rank_query started')
        query_logger = []
        for article_name in tqdm(self.article_names, desc='rank_query', unit='article'):
            try:
                ranked_docs = await self.wiki_agent.create_ranking(article_name=article_name)
                ndcg, pr_r_score = self.wiki_evaluater.rank_query(ranked_docs, article_name)
                query_logger.append((ndcg, pr_r_score))
            except Exception:
                self.logger.exception(f'rank_query failed for article={article_name}')
                break

        result = self.wiki_evaluater.mean_value(query_logger)
        self.logger.info(f'Final result for stage rank_query: {result}')
        return result

    async def rank_outline(self, neighbor_count=0, description_mode=0, clusterization_with_hint: bool = True):
        self.logger.info(f'Stage: rank_outline started: neighbor_count={neighbor_count}, description_mode={description_mode}, clusterization_with_hint={clusterization_with_hint}')
        self.outline_logger = []
        for article_name in tqdm(self.article_names, desc='rank_outline', unit='article'):
            try:
                outline = await self.wiki_agent.create_outline(
                    article_name=article_name, 
                    clusterization_with_hint=clusterization_with_hint, 
                    neighbor_count=neighbor_count, 
                    description_mode=description_mode
                )
                p, r, f = self.wiki_evaluater.rank_outline(outline, page)
                self.outline_logger.append((p, r, f))
            except Exception:
                self.logger.exception(f'rank_outline failed for article={article_name}')
                break

        result = self.wiki_evaluater.calc(self.outline_logger, is_flat=True)
        self.logger.info(f'Final result for stage rank_outline: {result}')
        return result

    async def rank_sections(self):
        self.logger.info('Stage: rank_sections started')
        self.article_gen_logger = []
        for article_name in tqdm(self.article_names, desc='rank_sections', unit='article'):
            try:
                sections = await self.wiki_agent.create_sections(article_name=article_name)
                pr, rec, f1, qa_cov, qa_sim = self.wiki_evaluater.rank_sections(sections, page, self.model_name)
                self.article_gen_logger.append((pr, rec, f1, qa_cov, qa_sim))
            except Exception:
                self.logger.exception(f'rank_sections failed for article={article_name}')
                break
        result = self.wiki_evaluater.calc(self.article_gen_logger, is_flat=False)
        self.logger.info(f'Final result for stage rank_sections: {result}')
        return result