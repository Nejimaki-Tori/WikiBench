# -*- coding: utf-8 -*-

from openai_utils import AsyncList, LlmCompleter
import math

REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT = """
Ты — интеллектуальный помощник, который составляет аннотацию для статьи.
По названию статьи и по названиям всех параграфов верхнего уровня напиши, о чем будет вся статья в целом.
Аннотация должна состоять из 2–5 предложений и давать представление о ключевых темах статьи.

Входные данные:

Название статьи: {article_title}

Заголовки параграфов:
{headers}

Ответ:
"text": ""
"""

REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT_ENG = """
You are an intelligent assistant tasked with writing a summary for an article.
Given the article's title and the titles of all top-level sections (which may be in Russian), write a short overview of what the article is generally about.
The summary should consist of 2–5 sentences and provide an understanding of the article’s main topics.
Your response must be written in English.

Input:

Article title: {article_title}

Section titles:
{headers}

Output:
"text": ""
"""

REFERENCE_SUBQUERIES_NO_HEADERS_PROMPT = """
Ты — интеллектуальный помощник, который пишет, о чем будет вся статья с данным названием.
По названию статьи распиши, используя 2-5 предложений, о чем может быть вся статья.
Отвечай на русском языке. Напиши только сам текст.

Ввод:

Article title: "{article_title}",

"text": ""
"""

REFERENCE_SUBQUERIES_NO_HEADERS_PROMPT_ENG = """
You are an intelligent assistant tasked with writing a summary for an article.
Given the article's title (which may be in Russian), write a short overview of what the article is generally about.
Based on the title of the article, describe in 2-5 sentences what the entire article could be about.
Your response *must* be written in English.

Input:

Article title: {article_title}

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

QA_QUESTIONS_PROMPT = """
Тебе нужно задать несколько вопросов по данному тексту секции из статьи Википедии. Ответы на данные вопросы должны содержаться в тексте секции. Задай не более 10 вопросов.

Текст:
{}
"""

QA_ANSWERS_PROMPT = """
Тебе нужно ответить на вопрос, опираясь на данный тебе текст. Используй информацию только из текста. Не используй внешние знания.

Текст:
{}

Вопрос:
{}
"""

QA_COMPARE_QUESTIONS_PROMPT = """
Дан вопрос и даны два ответа на него. Скажи {}, если ответы совпадают или {} иначе.
"""
class WikiGen:
    def __init__(self, client, model_name=None):
        self.client = client
        self.thinking_pass = " /no_think" if model_name == 'Qwen3-235B-A22B' or model_name == 'RefalMachine/RuadaptQwen3-32B-Instruct-v2' else ""

    def extract_response(self, response):
        return response.choices[0].message.content.strip() if response.choices else None
        
    #-----------------------------ANNOTATIONS------------------------------------#
    async def get_ref_subqueries(self, page, mode=0, reference_mode=1, name=None):
        '''
        Функция для получения эталонного плана статьи по данным заголовкам
        '''
        if reference_mode:
            if not page or not page.outline:
                print('Empty page!')
                return
            if mode == 0:
                myprompt = REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT
            else:
                myprompt = REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT_ENG
            headers = [header[1] for header in page.outline.keys() if header[0] == 'h2']
            headers_formatted = "\n".join(f"- {h}" for h in headers)
            myprompt = myprompt.format(article_title=page.name, headers=headers_formatted) + self.thinking_pass
        else:
            if mode == 0:
                myprompt = REFERENCE_SUBQUERIES_NO_HEADERS_PROMPT
            else:
                myprompt = REFERENCE_SUBQUERIES_NO_HEADERS_PROMPT_ENG
            myprompt = myprompt.format(article_title=name) + self.thinking_pass

        res = await self.client.get_completion(myprompt)
        response = self.extract_response(res)
        if not response:
            print("Couldn't get the answer")
            return
        if reference_mode:
            return (page.name, response)
        else:
            return response

    async def get_annotations(self, topR):
        annotations = AsyncList()
        for name, _ in topR:
            annotations.append(self.get_ref_subqueries(None, 0, 0, name))
            annotations.append(self.get_ref_subqueries(None, 1, 0, name))
    
        await annotations.complete_couroutines(batch_size=40)
        annotations = await annotations.to_list()
        new_texts = annotations
        texts = []
        for i in range(0, len(new_texts), 2):
            new_text = new_texts[i] + '\n' + new_texts[i + 1]
            texts.append(new_text)
            
        return texts
            
    #--------------------------------------------OUTLINE-------------------------------------------------------#
    async def get_cluster_description(self, name, cluster_text, description_mode=0, cluster_level_refine=False):
        if description_mode:
            myprompt = CLUSTER_DESCRIPTION_PROMPT.format(name, cluster_text) + self.thinking_pass
        else:
            myprompt = CLUSTER_OUTLINE_PROMPT.format(name, cluster_text) + self.thinking_pass
            
        res = await self.client.get_completion(myprompt)
        response = self.extract_response(res)
        
        if description_mode and response:
            myprompt = OUTLINE_FROM_DESCRIPTION.format(name, response) + self.thinking_pass
            res = await self.client.get_completion(myprompt)
            response = self.extract_response(res)
            
            if cluster_level_refine and response:
                myprompt = COMBINE_CLUSTER_PROMPT.format(name, response) + self.thinking_pass
                res = await self.client.get_completion(myprompt)
                response = self.extract_response(res)

        if not response:
            print("Couldn't get the answer!")
        return response

    async def generate_outline(self, clusters, name, description_mode=0):
        cluster_outlines = AsyncList()
        for label, sample_snippet in clusters.items():
            cluster_outlines.append(self.get_cluster_description(name, sample_snippet, description_mode))
        await cluster_outlines.complete_couroutines(batch_size=40)
        cluster_outlines = await cluster_outlines.to_list()
        mega_outline = "\n\n".join(f"{outline}" for outline in cluster_outlines)
        #print('Length of generated outline: ', len(mega_outline))
        myprompt =  COMBINE_OUTLINE_PROMPT.format(name, mega_outline) + self.thinking_pass
        res = await self.client.get_completion(myprompt)
        outline = self.extract_response(res)
        if not outline:
            print("Couldn't get the answer!")
        return outline

    #-----------------------------------------------RANKING------------------------------------------------#
    async def process_text(self, i1: int, text: str, name: str, positive_choice: str, negative_choice: str):
        """
        Функция обрабатывает один сниппет:
        - Формирует prompt и получает вероятности
        - Возвращает (индекс, вероятность)
        """
        myprompt = RANKING_PROMPT.format(name, text, positive_choice, negative_choice) + self.thinking_pass
        response = await self.client.get_probability(myprompt, rep_penalty=1.0, max_tokens=10)
        response = response.choices[0]
        snippet_result = None
        probs = {positive_choice: [], negative_choice: []}

        for token_info in response.logprobs.content:
            for variant in token_info.top_logprobs:
                key = variant.token.strip()
                if key == positive_choice or key == negative_choice:
                    probs[key].append(math.exp(variant.logprob))

        prob_pos = max(probs[positive_choice], default=0.0)
        prob_neg = max(probs[negative_choice], default=0.0)

        if prob_neg > prob_pos:
            prob_val = 1 - prob_neg
        else:
            prob_val = prob_pos
        
        return (i1, prob_val)
        
    async def filter_sources(self, name, texts):
        """
        Функция проходится по всем текстам и возвращает список вероятностей:
        - Возвращает [(индекс, вероятность)]
        """
        probs = AsyncList() 

        positive_choice = 'YES'
        negative_choice = 'NO'
        
        for i1, text in enumerate(texts):
            probs.append(
                self.process_text(
                    i1, 
                    text, 
                    name, 
                    positive_choice, 
                    negative_choice
                )
            )

        await probs.complete_couroutines(batch_size=40)
        final_probs = await probs.to_list()
        final_probs = sorted(final_probs)
        return final_probs

    #--------------------------------SECTIONS------------------------------------#
    #async def get_first_description(self, name, section, snippet_text, pred_text):
    #    if pred_text:
    #        myprompt = COMBINE_DESCRIPTION_PROMPT.format(name, section[1], pred_text, snippet_text)
    #    else:
    #        myprompt = WRITE_DESCRIPTION_PROMPT.format(name, section[1], snippet_text)
    #    res = await self.client.get_completion(myprompt)
    #    if res.choices is not None:
    #        response = res.choices[0].message.content.strip()
    #        return response
    #    else:
    #        print("Couldn't get the answer")

    #async def generate_section(self, section, batched_snippets, name):
    #    pred_text = None
    #    description = None
    #    #print('batch count ', len(batched_snippets))
    #    for snippets in batched_snippets:
    #        #print('whaaaaaaaaa ', len(snippets))
    #        #print(snippets[0])
    #        snippet_text =  "\n\n".join(f"{snippet}" for snippet in snippets)
    #        description = await self.get_first_description(name, section, snippet_text, pred_text)
    #        #print('DRAFT DRAFT')
    #        #print(description)
    #        #print()
    #        #print()
    #        #print('-'*100)
    #        pred_text = description
    #    #print(description)
    #    #print()
    #    #print()
    #    #print('-'*100)
    #    return description

    async def get_description(self, name, section, combined_text):
        myprompt = SUMMARIZATION_PROMPT.format(name, section[1], combined_text) + self.thinking_pass
        res = await self.client.get_completion(myprompt)
        response = self.extract_response(res)
        return response

    async def generate_section(self, section, snippets, name):
        summaries = snippets
        while len(summaries) > 1:
            groups = [summaries[i:i + 5] for i in range(0, len(summaries), 5)]
            next_summaries = AsyncList()
            for group in groups:
                snippet_text = "\n".join(group)
                next_summaries.append(self.get_description(name, section, snippet_text))

            await next_summaries.complete_couroutines(batch_size=10)
            summaries = await next_summaries.to_list()

        return summaries[0]

    #async def get_one_group(self, text, pred_text):
    #    if pred_text:
    #        myprompt = COMBINE_GROUP_PROMPT.format(pred_text, text)
    #    else:
    #        myprompt = WRITE_GROUP_PROMPT.format(text)
    #    res = await self.client.get_completion(myprompt)
    #    if res.choices is not None:
    #        response = res.choices[0].message.content.strip()
    #        return response
    #    else:
    #        print("Couldn't get the answer")
    #        
    #async def get_group_description(self, group):
    #    if len(group) == 1:
    #        return group[0]
    #    batched_snippets = []
    #    i = 0
    #    while i < len(group):
    #        batched_snippets.append(group[i:i+5])
    #        i += 5
    #    pred_summary = None
    #    #print('group num ', len(group))
    #    #print('group len ', len(group[0]))
    #    #print('group text ', group[0][:10])
    #    #print('groups batches ', len(batched_snippets))
    #    for batch in batched_snippets:
    #        batch_text = " ".join([text for text in batch])
    #        group_description = await self.get_one_group(batch_text, pred_summary)
    #        #print('one group ', group_description[:200])
    #        pred_summary = group_description
    #    #print('summed group ', len(group_description))
    #    #print('-'*100)
    #    #print()
    #    return group_description

    async def get_one_group(self, text):
        myprompt = WRITE_GROUP_PROMPT.format(text) + self.thinking_pass
        res = await self.client.get_completion(myprompt)
        response = self.extract_response(res)
        return response
        
    async def get_group_description(self, group_text):
        summaries = group_text
        i = 0
        while len(summaries) > 1:
            groups = [summaries[i:i + 5] for i in range(0, len(summaries), 5)]
            #print('Iteration number for groups: ', i)
            #print('Number of groups for groups: ', len(groups))
            i += 1
            next_summaries = AsyncList()
            for group in groups:
                batched_text = "\n".join(group)
                next_summaries.append(self.get_one_group(batched_text))
            
            await next_summaries.complete_couroutines(batch_size=5)
            summaries = await next_summaries.to_list()
            
        return summaries[0] if summaries else ""

    async def summarize_groups(self, groups):
        group_summary = AsyncList()
        for group in groups:
            group_summary.append(self.get_group_description(group))
        await group_summary.complete_couroutines(batch_size=10)
        group_summary = await group_summary.to_list()
        return group_summary