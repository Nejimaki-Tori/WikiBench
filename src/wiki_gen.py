# -*- coding: utf-8 -*-

from openai_utils import AsyncList, LlmCompleter
import math
import re

REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT = """
Ты — интеллектуальный помощник, который составляет аннотацию для статьи.
По названию статьи и по названиям всех параграфов верхнего уровня напиши, о чем будет вся статья в целом.
Аннотация должна состоять из 2–5 предложений и давать представление о ключевых темах статьи.

Входные данные:

Название статьи: {}

Заголовки параграфов:
{}

Ответ:
"text": ""
"""

REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT_ENG = """
You are an intelligent assistant tasked with writing a summary for an article.
Given the article's title and the titles of all top-level sections (which may be in Russian), write a short overview of what the article is generally about.
The summary should consist of 2–5 sentences and provide an understanding of the article’s main topics.
Your response must be written in English.

Input:

Article title: {}

Section titles:
{}

Output:
"text": ""
"""

REFERENCE_SUBQUERIES_NO_HEADERS_PROMPT = """
Ты — интеллектуальный помощник, который пишет, о чем будет вся статья с данным названием.
По названию статьи распиши, используя 2-5 предложений, о чем может быть вся статья.
Отвечай на русском языке. Напиши только сам текст.

Ввод:

Article title: "{}",

"text": ""
"""

REFERENCE_SUBQUERIES_NO_HEADERS_PROMPT_ENG = """
You are an intelligent assistant tasked with writing a summary for an article.
Given the article's title (which may be in Russian), write a short overview of what the article is generally about.
Based on the title of the article, describe in 2-5 sentences what the entire article could be about.
Your response *must* be written in English.

Input:

Article title: {}

Output:
"text": ""
"""

RANKING_PROMPT = """
Ты - эксперт в написании статей Википедии и оцениваешь найденные документы: подходят ли они для написания статьи на заданную тему. Тебе нужно по данному названию статьи и тексту статьи-источника определить, является ли он релевантным для этой темы. Текст является релевантным, если информация в нем может быть использована для составления текста статьи по данной теме.

Тема:
{}
Текст:
{}

Соответствует ли запрос теме? Начни свой ответ с {}, если соответствует или c {}, если не соответствует.
"""

CLUSTER_OUTLINE_PROMPT = """
Ты — эксперт по написанию статей РУССКОЯЗЫЧНОЙ Википедии. На основе темы статьи и предложенных источников составь план ОДНОЙ секции, которая логически объединяет представленные материалы.

Твоя задача:
1. Придумай подходящее название для этой секции, отражающее суть текстов.
2. Построй структуру раздела с помощью заголовков: 
   - "#" — для заголовка секции, 
   - "##" — для подразделов, 
   - "###" — для подподразделов, и так далее.
3. Не используй никаких описаний, пояснений или комментариев — только заголовки.
4. Используй от 2 до 5 заголовков в общей структуре.
5. Не выходи за рамки ОДНОГО раздела.

Тема статьи:  
{}

Источник(и):  
{}
"""


CLUSTER_DESCRIPTION_PROMPT = """
Ты — эксперт по написанию статей для Википедии. Твоя задача — на основе предложенного текста составить краткое аннотированное описание, которое отражает его суть.
Пожалуйста, следуй следующим инструкциям:
1. Извлеки только ключевые факты и основные идеи, содержащиеся именно в этом источнике. Не используй внешние знания.
2. Сформулируй краткое описание в 2–5 предложениях, соблюдая нейтральный и энциклопедический стиль.
3. Не добавляй заголовков, пояснений, вводных или заключительных фраз. Просто выведи краткое содержание.
Это описание в дальнейшем будет использоваться для составления плана раздела статьи на Википедии, поэтому оно должно быть ёмким и информативным.

Тема статьи: {}
Источник:
{}
"""

OUTLINE_FROM_DESCRIPTION = """
Ты — опытный редактор Википедии. На основе данной тебе темы статьи и приведённого ниже краткого описания составь план одной секции статьи.
Пожалуйста, строго следуй этим инструкциям:
1. Используй "# Title" для заголовка раздела, "## Title" для заголовков подразделов, "### Title" для подподразделов и так далее.
2. Не пиши ничего, кроме структуры из заголовков. Не добавляй пояснений, описаний или текста.
3. Ограничь структуру 2–5 заголовками (всех уровней суммарно).
Составь план только для одной секции статьи, на основе смысловых блоков в описании.

Тема статьи: {}
Краткое описание:
{}
"""

COMBINE_CLUSTER_PROMPT = """
Ты - эксперт в написании статей Википедии и занимаешься составлением планов секций. У тебя уже есть несколько черновых вариантов планов, подготовленных экспертами для одной секции и тема статьи, в которой будет эта секция. Объедини эти планы в общий короткий план, который раскроет основные темы этой секции.
Вот формат в котором нужно все описать:
1. Используй "# Title" для заголовка раздела, "## Title" для заголовков подразделов, "### Title" для подподразделов и так далее, только такой формат с решетками и никакой другой!
2. Ничего другого писать не нужно.
3. Старайся ограничиться небольшим числом заголовков (2-5 штук).
4. Создай план только для одной секции.

Тема статьи: {}

Планы секций:
{}
"""


COMBINE_OUTLINE_PROMPT = """Тебе нужно написать короткий план для страницы РУССКОЯЗЫЧНОЙ Википедии. У тебя уже есть несколько планов секций из этой статьи, подготовленных экспертами по теме статьи. Объедини эти планы в общий короткий план, который раскроет основные темы статьи.
Вот формат в котором нужно все описать:
1. Начни план с названия интересующей темы "Topic"
2. Используй "# Title" для заголовков разделов, "## Title" для заголовков подразделов, "### Title" для подподразделов и так далее, только такой формат с решетками и никакой другой!
3. В конце плана напиши **END**
4. Ничего другого писать не нужно, соблюдай формат ответа.

Тема статьи: {}

Планы секций:
{}
"""

WRITE_DESCRIPTION_PROMPT = """
Ты - опытный редактор Википедии и занимаешься написанием текстов для секций. Другие редакторы уже собрали необходимые тексты-источники для данного раздела, тебе нужно только оформить информацию из этого текста в текст для статьи.
Вот формат в котором нужно все описать:
1. Извлеки только ключевые факты и основные идеи, содержащиеся именно в этих источниках. Не используй внешние знания.
2. Сформулируй связный, нейтральный, энциклопедический текст объёмом от 3 до 6 предложений.
3. Не добавляй заголовков, пояснений, вводных или заключительных фраз, напиши только текст для секции.
4. Текст должен соответствовать как теме статьи, так и названию секции.
5. Если название секции совпадает с названием статьи, то эта секция - "Введение".

Тема статьи: {}

Название секции: {}

Тексты источников:
{}

Пиши в стиле Википедии!
"""

SUMMARIZATION_PROMPT = """
Ты - опытный редактор Википедии и занимаешься написанием текстов для секций. Другие редакторы уже собрали необходимые тексты-источники для данного раздела, тебе нужно только оформить информацию из этого текста в текст для статьи.
Вот формат в котором нужно все описать:
1. Извлеки только ключевые факты и основные идеи, содержащиеся именно в этих источниках. Не используй внешние знания.
2. Сформулируй связный, нейтральный, энциклопедический текст объёмом от 5 до 8 предложений.
3. Не добавляй заголовков, пояснений, вводных или заключительных фраз, напиши только текст для секции.
4. Текст должен соответствовать как теме статьи, так и названию секции.
5. Если название секции совпадает с названием статьи, то эта секция - "Введение".
6. Устрани дублирование информации между разными фрагментами.

Тема статьи: {}

Название секции: {}

Тексты источников:
{}

Пиши в стиле Википедии!
"""

COMBINE_DESCRIPTION_PROMPT = """
Ты - опытный редактор Википедии и занимаешься написанием текстов для секций. Твоя задача — на основе черновика секции и новых фрагментов‑источников дописать и улучшить текст секции статьи.
Вот формат в котором нужно все описать:
1. Не пиши с нуля — работай с уже сгенерированным текстом (черновиком).
2. Извлеки только ключевые факты и основные идеи, содержащиеся именно в этом источнике. Не используй внешние знания.
3. Добавляй новую информацию в логичное место текста или — если не получается — просто допиши новый параграф через "\n\n".
4. Добавляй НЕ БОЛЕЕ двух абзацев.
5. Ответ должен состоять из краткого, связного текста.
6. Если источники не предоставляют полезной информации, оставь текст без изменений.

Тема статьи: {}

Название секции: {}

Заготовка текста:
{}

Тексты источников:
{}

Дополни заготовку текста новыми абзацами, если это возможно. Пиши в стиле Википедии!
"""

WRITE_GROUP_PROMPT = """
Ты - опытный редактор Википедии. Твоя задача — на основе предложенного текста составить краткое аннотированное описание, которое отражает его суть.
Пожалуйста, следуй следующим инструкциям:
1. Извлеки только ключевые факты и основные идеи, содержащиеся именно в этом источнике. Не используй внешние знания.
2. Сформулируй краткое описание в 4–6 предложениях.
3. Не добавляй заголовков, пояснений, вводных или заключительных фраз. Просто выведи краткое содержание.
Это описание в дальнейшем будет использоваться для составления плана раздела статьи на Википедии, поэтому оно должно быть ёмким и информативным.

Текст источника:
{}
"""

COMBINE_GROUP_PROMPT = """
Ты - опытный редактор Википедии. Твоя задача — на основе уже заготовленного текста и текста источников составить краткое аннотирование текста, которое отражает его суть.
Пожалуйста, следуй следующим инструкциям:
1. Извлеки только ключевые факты и основные идеи, содержащиеся именно в этом источнике. Не используй внешние знания.
2. Сформулируй краткое описание в 2–5 предложениях.
3. Не добавляй заголовков, пояснений, вводных или заключительных фраз. Просто выведи краткое содержание.
4. Не пиши с нуля — работай с уже заготовленным текстом (черновиком).
Это описание в дальнейшем будет использоваться для составления текста статьи на Википедии, поэтому оно должно быть ёмким и информативным.

Заготовка текста:
{}

Текст источника:
{}
"""

class WikiGen:
    def __init__(
        self, 
        client, 
        concurrency: int = 40,
        is_think_mode_disabled: bool = True
    ):
        self.client = client
        self.concurrency = concurrency
        self.thinking_pass = ' /no_think' if is_think_mode_disabled else ""

    def extract_response(self, response):
        text = response.choices[0].message.content.strip() if response.choices else None
        return re.sub(r"<\/?think>", "", text).strip() if text else None

    # --- Helper functions block ---
    
    async def complete_prompt_template(
        self,
        template: str,
        *template_args,
        max_tokens: int = 512,
        logprobs=False,
        is_extraction_needed: bool = True,
        **template_kwargs
    ):
        '''Returns response from LLM. Uses given prompt and formats it with given arguments'''     
        prompt = template.format(*template_args, **template_kwargs) + self.thinking_pass
        result = await self.client.get_completion(prompt, max_tokens=max_tokens, logprobs=logprobs)
        return self.extract_response(result) if is_extraction_needed else result

    async def hierarchical_merge(
        self,
        items: list[str],
        summarize_fn,
        group_size: int = 5,
        batch_size: int = 10
    ):
        '''Function that produces a hierarchical merge of the given list of texts'''
        summaries = [s for s in items if s]
        if not summaries:
            return ''

        while len(summaries) > 1:
            groups = [
                summaries[i : i + group_size]
                for i in range(0, len(summaries), group_size)
            ]

            tasks = AsyncList()
            for group in groups:
                clean_group = [s for s in group if s]
                if not clean_group:
                    continue
                    
                tasks.append(summarize_fn(clean_group))

            if not tasks:
                break

            await tasks.complete_couroutines(batch_size=batch_size)
            new_summaries = await tasks.to_list()

            summaries = [s for s in new_summaries if s]

            if not summaries:
                return ''

        return summaries[0] if summaries else ''
        
    
    # --- ANNOTATIONS BLOCK ---
    
    async def get_ref_subqueries(
        self, 
        is_in_english: bool = False, 
        is_reference_mode: bool = True, 
        article_name: str = '',
        page = None
    ):
        '''
        Function for generating reference article description based on given h2 headers.
        Can return both english and russian description.
        '''
        if is_reference_mode:
            if not page or not page.outline:
                print('Empty page!')
                return None
                
            if is_in_english:
                myprompt = REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT_ENG
            else:
                myprompt = REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT          
            headers_h2 = [header[1] for header in page.outline.keys() if header[0] == 'h2']
            headers_h2_formatted = '\n'.join(f'- {h2}' for h2 in headers_h2)
            
            response = await self.complete_prompt_template(
                myprompt,
                page.name,
                headers_h2_formatted
            )
        else:
            if is_in_english:
                myprompt = REFERENCE_SUBQUERIES_NO_HEADERS_PROMPT_ENG
            else:
                myprompt = REFERENCE_SUBQUERIES_NO_HEADERS_PROMPT

            response = await self.complete_prompt_template(
                myprompt,
                article_name
            )

        if is_reference_mode:
            return (page.name, response)
        else:
            return response

    async def get_bm25_query(self, article_name: str):
        russian_query = await self.get_ref_subqueries(
            is_in_english=False,
            is_reference_mode=False,
            article_name=article_name
        )

        english_query = await self.get_ref_subqueries(
            is_in_english=True,
            is_reference_mode=False,
            article_name=article_name
        )

        query = f'{russian_query}\n{english_query}'

        return query
        
    # --- SOURCE RANKING BLOCK ---
    
    async def process_text(
        self, 
        snippet_text: str, 
        article_name: str, 
        positive_choice: str, 
        negative_choice: str
    ):
        '''Returns the probability of wether the given snippet is relevant to the given article name'''
        response = await self.complete_prompt_template(
            RANKING_PROMPT,
            article_name,
            snippet_text,
            positive_choice,
            negative_choice,
            max_tokens=5,
            logprobs=True,
            is_extraction_needed=False
        )

        response = response.choices[0]
        probs = {positive_choice: [], negative_choice: []}

        for token_info in response.logprobs.content:
            for variant in token_info.top_logprobs:
                key = variant.token.strip().upper()
                if key == positive_choice or key == negative_choice:
                    probs[key].append(math.exp(variant.logprob))

        prob_pos = max(probs[positive_choice], default=0.0)
        prob_neg = max(probs[negative_choice], default=0.0)

        if prob_neg > prob_pos:
            prob_val = 1 - prob_neg
        else:
            prob_val = prob_pos
        
        return prob_val
        
    async def filter_sources(self, name, texts):
        '''Returns the list of probs'''
        probs = AsyncList() 

        positive_choice = 'YES'
        negative_choice = 'NO'
        
        for text in texts:
            probs.append(
                self.process_text(
                    text, 
                    name, 
                    positive_choice, 
                    negative_choice
                )
            )

        await probs.complete_couroutines(batch_size=self.concurrency)
        final_probs = await probs.to_list()
        return final_probs
        

    # --- OUTLINE BLOCK ---
    
    async def get_cluster_description(
        self, 
        article_name: str, 
        cluster_text: str, 
        description_mode: bool = False, 
        cluster_level_refine: bool = False
    ):
        '''Returns an outline for one cluster'''
        if not description_mode:
            outline = await self.complete_prompt_template(
                CLUSTER_OUTLINE_PROMPT,
                article_name,
                cluster_text
            )

            return outline
        description = await self.complete_prompt_template(
            CLUSTER_DESCRIPTION_PROMPT,
            article_name,
            cluster_text
        )
        outline = await self.complete_prompt_template(
            OUTLINE_FROM_DESCRIPTION,
            article_name,
            description
        )

        if cluster_level_refine:
            refined_cluster_description = await self.complete_prompt_template(
                COMBINE_CLUSTER_PROMPT,
                article_name,
                outline
            )

            outline = refined_cluster_description

        return outline

    async def generate_outline(
        self, 
        clusters: dict, 
        article_name: str, 
        description_mode: bool = False
    ):
        '''Creates an outline, using combined outline for each text cluster'''
        
        cluster_outlines = AsyncList()
        for label, sample_snippet in clusters.items():
            cluster_outlines.append(
                self.get_cluster_description(
                    article_name, 
                    sample_snippet, 
                    description_mode
                )
            )
            
        await cluster_outlines.complete_couroutines(batch_size=self.concurrency)
        cluster_outlines = [o for o in await cluster_outlines.to_list() if o]
        collected_outlines = "\n\n".join(cluster_outlines)
        combined_outline = await self.complete_prompt_template(
            COMBINE_OUTLINE_PROMPT,
            article_name,
            collected_outlines
        )

        return combined_outline


    # --- SECTIONS BLOCK --- #

    async def generate_section(
        self, 
        article_name: str, 
        section: tuple[str, str], 
        snippets: list[str],
        group_size: int = 5
    ):
        '''Generates section using hierarchical merging'''
        async def summarize_for_one_section(group: list[str]):
            combined_snippet_text = '\n'.join(group)
            section_title = section[1]
            return await self.complete_prompt_template(
                SUMMARIZATION_PROMPT, 
                article_name, 
                section_title, 
                combined_snippet_text
            )
        section_text = await self.hierarchical_merge(
            snippets,
            summarize_fn=summarize_for_one_section,
            group_size=group_size,
            batch_size=self.concurrency
        )

        return section_text

    async def get_group_description(
        self, 
        group_texts: list[str]
    ):
        async def summarize_for_one_group(group: list[str]):
            combined_group_text = '\n'.join(group)
            return await self.complete_prompt_template(
                WRITE_GROUP_PROMPT,
                combined_group_text
            )

        group_summary = await self.hierarchical_merge(
            group_texts,
            summarize_fn=summarize_for_one_group,
            group_size=5,
            batch_size=self.concurrency
        )

        return group_summary

    async def summarize_groups(
        self, 
        groups: list[list[str]] # List of groups; each group is a list of snippets
    ):
        group_summary = []
        for group in groups:
            group_summary.append(await self.get_group_description(group))

        return group_summary