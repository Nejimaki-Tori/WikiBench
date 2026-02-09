import logging
from tqdm.asyncio import tqdm
from wiki_gen import WikiGen
from openai_utils import LlmCompleter, AsyncList
from wiki_evaluater import WikiEvaluater
from wiki_agent import WikiAgent
from wiki_utils import WikiUtils
from results_to_table import ranking_results, outline_results, section_results

class WikiBench:
    '''Main class for runnig benchmark pipeline'''
    def __init__(
        self, 
        url: str, 
        key: str, 
        model_name: str, 
        device, 
        encoder,
        number_of_articles: int = 100,
        log_level=logging.INFO
    ):
        self.model_name = model_name
        self.device = device
        self.encoder = encoder
        self.client = LlmCompleter(api_address=url, api_key=key, model_name=model_name)
        self.wiki_writer = WikiGen(self.client, self.model_name)
        self.logger = self._setup_logger(level=log_level)
        self.number_of_articles = number_of_articles
        with open('small_articles_data.txt', 'r', encoding='utf-8') as file:
            self.article_names = [x for x in file.read().split('\n') if x.strip()][:self.number_of_articles]
        self.wiki_utility = WikiUtils(device=self.device, encoder=self.encoder)
        self.wiki_agent = WikiAgent(utils=self.wiki_utility, client=self.wiki_writer)
        self.is_env_prepared = False
        self.wiki_evaluater = WikiEvaluater(self.wiki_agent.device, self.wiki_agent.encoder)
        self.query_logger = []
        self.outline_logger = []
        self.article_gen_logger = []
        self.stream_results = False
        self.logger.info(f'WikiBench initialized: model={self.model_name}, articles={len(self.article_names)}')

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

    def format_bootstrap_results(self, result):
        pr = (result[0], result[1], result[2])
        rec = (result[3], result[4], result[5])
        f = (result[6], result[7], result[8])

        def format_one_result(metric, data):
            mean, low, high = data
            return f'{metric}={mean:.4f} [{low:.4f}; {high:.4f}]'

        if len(result) > 9:
            r = (result[9], result[10], result[11])
            b = (result[12], result[13], result[14])
            return " | ".join([format_one_result('P', pr), format_one_result('R', rec), format_one_result('F', f), format_one_result('Rouge', r), format_one_result('BLEU', b)])

        return " | ".join([format_one_result('P', pr), format_one_result('R', rec), format_one_result('F', f)])

    def prepare_env(self, are_texts_ready=True, window_size=600, overlap=0):
        self.logger.info(f'Preparing env: are_texts_ready={are_texts_ready}, window={window_size}, overlap={overlap}')
        if self.is_env_prepared:
            self.logger.info('Env already prepared. Loading created corpus...')
            self.wiki_utility.load_created_corpus()
            self.wiki_agent.utils = self.wiki_utility
            return
            
        if not are_texts_ready:
            self.logger.info('Creating article corpus from scratch...')
            self.wiki_utility.create_article_corpus_from_scratch(article_bare_names=self.article_names)

        self.logger.info('Creating main corpus from scratch...')
        self.wiki_agent.utils.create_corpus_from_scratch(article_names=self.article_names, window_size=window_size, overlap=overlap)
        self.is_env_prepared = True
        self.logger.info('Enviroment prepared!')

    def load_enviroment(self):
        self.logger.info('Loading enviroment...')
        self.wiki_agent.utils.load_created_corpus()
        self.logger.info('Enviroment loaded!')
        
    async def rank_query(self):  
        self.logger.info('Stage: rank_query started')
        self.query_logger = []
        for article_name in tqdm(self.article_names, desc='rank_query', unit='article'):
            try:
                ranked_docs = await self.wiki_agent.create_ranking(article_name=article_name)
                ndcg, pr_r_score = self.wiki_evaluater.rank_query(ranked_docs, article_name)
                self.query_logger.append((ndcg, pr_r_score))
            except Exception:
                self.logger.exception(f'rank_query failed for article={article_name}')
                break

        result = self.wiki_evaluater.mean_value(self.query_logger)
        ranking_results(self.query_logger, self.article_names[:self.number_of_articles])
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
                p, r, f = self.wiki_evaluater.rank_outline(outline, article_name)
                self.outline_logger.append((p, r, f))
            except Exception:
                self.logger.exception(f'rank_outline failed for article={article_name}')
                break

        result = self.wiki_evaluater.bootstrap(self.outline_logger, is_flat=True)
        outline_results(self.outline_logger, self.article_names[:self.number_of_articles])
        self.logger.info(f'Final result for stage rank_outline: {self.format_bootstrap_results(result)}')
        return result

    async def rank_sections(self):
        self.logger.info('Stage: rank_sections started')
        self.article_gen_logger = []
        for article_name in tqdm(self.article_names, desc='rank_sections', unit='article'):
            try:
                sections = await self.wiki_agent.create_sections(article_name=article_name)
                out = self.wiki_evaluater.rank_sections(sections, article_name)
                self.article_gen_logger.append((
                    out['precision'], 
                    out['recall'], 
                    out['f1'], 
                    out['rouge_l'], 
                    out['bleu']
                ))
            except Exception:
                self.logger.exception(f'rank_sections failed for article={article_name}')
                break
        result = self.wiki_evaluater.bootstrap(self.article_gen_logger, is_flat=False)
        section_results(self.article_gen_logger, self.article_names[:self.number_of_articles])
        self.logger.info(f'Final result for stage rank_sections: {self.format_bootstrap_results(result)}')
        return result