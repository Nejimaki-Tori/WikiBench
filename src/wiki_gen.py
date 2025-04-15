# -*- coding: utf-8 -*-

import os
import json
import random
from tqdm.auto import tqdm
from openai_utils import AsyncList, LlmCompleter

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
По названию статьи распиши, используя только 5 предложений, о чем может быть вся статья.

Составь ответ в поле "text" для следующего ввода:
"article_title": "{article_title}",

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
Ты — эксперт по написанию статей Википедии. На основе темы статьи и предложенных источников составь план ОДНОЙ секции, которая логически объединяет представленные материалы.

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
1. Используй "# Title" для заголовка раздела, "## Title" для заголовков подразделов, "### Title" для подподразделов и так далее.
2. Ничего другого писать не нужно.
3. Старайся ограничиться небольшим числом заголовков (2-5 штук).
4. Создай план только для одной секции.

Тема статьи: {}

Планы секций:
{}
"""


COMBINE_OUTLINE_PROMPT = """Тебе нужно написать короткий план для страницы Википедия.
У тебя уже есть несколько планов секций из этой статьи, подготовленных экспертами по теме статьи.
Объедини эти планы в общий короткий план, который раскроет основные темы статьи.
Вот формат в котором нужно все описать:
1. Начни план с названия интересующей темы "Topic"
2. Используй "# Title" для заголовков разделов, "## Title" для заголовков подразделов, "### Title" для подподразделов и так далее.
3. В конце плана напиши **END**
4. Ничего другого писать не нужно.

Тема статьи: {}

Планы секций:
{}
"""

class WikiGen:
    def __init__(self, client):
        self.client = client

    async def get_ref_subqueries(self, page, mode=0):
        '''
        Функция для получения эталонного плана статьи по данным заголовкам
        '''
        if not page.outline:
            print('Empty page!')
            return
        if mode == 0:
            myprompt = REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT
        else:
            myprompt = REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT_ENG
        headers = [header[1] for header in page.outline.keys() if header[0] == 'h2']
        headers_formatted = "\n".join(f"- {h}" for h in headers)
        myprompt = myprompt.format(article_title=page.name, headers=headers_formatted)
        res = await self.client.get_completion(myprompt)
        
        if res.choices is not None:
            response = res.choices[0].message.content.strip()
            return (page.name, response)
        else:
            print("Couldn't get the answer")

    async def get_cluster_description(self, name, cluster_text, description_mode=0, cluster_level_refine=False):
        if description_mode:
            myprompt = CLUSTER_DESCRIPTION_PROMPT.format(name, cluster_text)
        else:
            myprompt = CLUSTER_OUTLINE_PROMPT.format(name, cluster_text)
        res = await self.client.get_completion(myprompt)
        if res.choices is not None:
            response = res.choices[0].message.content.strip()
            if description_mode:
                myprompt = OUTLINE_FROM_DESCRIPTION.format(name, response)
                res = await self.client.get_completion(myprompt)
                if res.choices is not None:
                    response = res.choices[0].message.content.strip()
                    if cluster_level_refine:
                        myprompt = COMBINE_CLUSTER_PROMPT.format(name, response)
                        res = await self.client.get_completion(myprompt)
                    if res.choices is not None:
                        response = res.choices[0].message.content.strip()
            return response
        else:
            print("Couldn't get the answer")

    async def combine_cluster(self, name, clusters_description):
        myprompt = COMBINE_CLUSTER_PROMPT.format(name, clusters_description)
        res = await self.client.get_completion(myprompt)
        if res.choices is not None:
            response = res.choices[0].message.content.strip()
            return response
        else:
            print("Couldn't get the answer")

    async def combine_outline(self, name, outlines):
        myprompt = COMBINE_OUTLINE_PROMPT.format(name, outlines)
        res = await self.client.get_completion(myprompt)
        if res.choices is not None:
            response = res.choices[0].message.content.strip()
            return response
        else:
            print("Couldn't get the answer")
            
    async def process_text(self, i1: int, text: str, name: str, positive_choice: str, negative_choice: str):
        """
        Функция обрабатывает ОДИН текст:
        - Разбивает его на сниппеты по 2000 слов с перекрытием 512
        - Для каждого сниппета формирует запрос `prompt` и получает вероятности
        - Возвращает (индекс, вероятность)
        """
        words = text.split()
        window_size = 2000
        overlap = 512
        windows = []
        i = 0
        while i < len(words):
            snippet = ' '.join(words[i: i + window_size])
            windows.append(snippet)
            if i + window_size >= len(words):
                break
            i += window_size - overlap
            
        snippet_probs = AsyncList()
    
        for k, snippet in enumerate(windows):
            prompt = RANKING_PROMPT.format(name, snippet, positive_choice, negative_choice)
            snippet_probs.append(
                self.client.get_probability(
                    prompt, 
                    rep_penalty=1.0, 
                    max_tokens=10
                )
            )
    
        await snippet_probs.complete_couroutines(batch_size=20)
        snippet_results = await snippet_probs.to_list()
        combined_prob = 0.0
        count = 0
        prob_val_neg = 0
        prob_val_pos = 0
        prob_val = 0
        for sp in snippet_results:
            if negative_choice in sp:
                prob_val_neg = sp[negative_choice]
            if positive_choice in sp:
                prob_val_pos = sp[positive_choice]

            if prob_val_neg > prob_val_pos:
                prob_val = 1 - prob_val_neg
            elif prob_val_pos != 0:
                prob_val = prob_val_pos
    
            combined_prob += prob_val
            count += 1
    
        if count > 0:
            combined_prob /= count
    
        return (i1, combined_prob)
        
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
    
    async def generate_outline(self, clusters, name, description_mode=0):
        cluster_outlines = AsyncList()
        for label, sample_snippet in clusters.items():
            cluster_outlines.append(self.get_cluster_description(name, sample_snippet, description_mode))
        await cluster_outlines.complete_couroutines(batch_size=20)
        cluster_outlines = await cluster_outlines.to_list()
        mega_outline = "\n\n".join(f"{outline}" for outline in cluster_outlines)
        outline = await self.combine_outline(name, mega_outline)
        return outline

        
    '''
    async def get_one_cluster(self, name, sample_snippets, description_mode=0):
        cluster_descriptions = AsyncList()
        for sn in sample_snippets:
            cluster_descriptions.append(self.get_cluster_description(name, sn, description_mode))
        await cluster_descriptions.complete_couroutines(batch_size=3)
        cluster_descriptions = await cluster_descriptions.to_list()
        draft_cluster_outline = "\n\n".join(f"- {desc}" for desc in cluster_descriptions)
        cluster_outline = await self.combine_cluster(name, draft_cluster_outline)
        return cluster_outline
    
    async def generate_outline(self, clusters, name):
        cluster_outlines = AsyncList()
        for label, sample_snippets in clusters.items():
            cluster_outlines.append(self.get_one_cluster(name, sample_snippets))
        await cluster_outlines.complete_couroutines(batch_size=20)
        cluster_outlines = await cluster_outlines.to_list()
        mega_outline = "\n\n".join(f"{outline}" for outline in cluster_outlines)
        outline = await self.combine_outline(name, mega_outline)
        return outline
    '''