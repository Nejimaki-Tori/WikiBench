import os
import json
from tqdm.auto import tqdm
from openai_utils import AsyncList, LlmCompleter

SUBQUERIES_PROMPT = """
Ты — интеллектуальный помощник, который составляет подзапросы для поиска и анализа источников и прояснения ключевых аспектов темы.
Исходя из заданного названия статьи, тебе нужно:

1. Сформулировать 5–7 подзапросов, которые помогут найти релевантные источники.
2. Каждый подзапрос должен быть четко связан с основным заголовком и охватывать различные аспекты темы, помогая глубже ее раскрыть.
3. Подзапросы должны быть точными, конкретными и подходящими для поиска информации.
4. Подзапросы должны быть разнообразными: рассматривай исторические, научные, технические, социальные, экономические и другие аспекты темы, если это применимо.
5. Избегай повторений в вопросах — каждый подзапрос должен раскрывать новый аспект темы.
6. Ответ должен быть строго в формате JSON.


Пример ответа:
{{
  "article_title": "Черная смерть",
  "subqueries": [
    "Каковы основные причины возникновения пандемии чумы в средневековье?",
    "Когда произошла вспышка чумы?",
    "Какие пути распространения чумы способствовали ее быстрому охвату Европы?",
    "Как реагировали государства и общество на пандемию чумы?",
    "Какие методы лечения чумы использовались в Средневековье?",
    "Какие изменения в медицине и санитарии произошли после эпидемии?",
    "Есть ли современные вспышки чумы, и как она лечится сегодня?"
  ]
}}

Другой пример ответа:
{{
  "article_title": "Ядерная катастрофа в Чернобыле",
  "subqueries": [
    "Каковы основные причины аварии на Чернобыльской АЭС?",
    "Как происходила ликвидация последствий катастрофы?",
    "Как радиация повлияла на здоровье людей и окружающую среду?",
    "Какие меры были приняты для предотвращения подобных аварий в будущем?",
    "Какое влияние катастрофа оказала на развитие атомной энергетики?",
  ]
}}


Теперь составь аналогичный JSON-ответ для следующего названия статьи:
{{
  "article_title": "{name}",
  "subqueries": [
    
  ]
}}
"""

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

REFERENCE_SUBQUERIES_NO_HEADERS_PROMPT = """
Ты — интеллектуальный помощник, который пишет, о чем будет вся статья с данным названием.
По названию статьи распиши, используя только 5 предложений, о чем может быть вся статья.

Составь ответ в поле "text" для следующего ввода:
"article_title": "{article_title}",

"text": ""
"""

RANKING_PROMPT = """
Ты - эксперт в написании статей Википедии и оцениываешь найденные документы: подходят ли они для написания статьи на заданную тему. Тебе нужно по данному названию статьи и тексту статьи-источника определить, является ли он релевантным для этой темы. Текст является релевантным, если информация в нем может быть использована для составления текста статьи по данной теме.

Тема:
{}
Текст:
{}

Соответствует ли запрос теме? Начни свой ответ с {} или {}.
"""

class WikiGen:
    def __init__(self, client):
        self.client = client

    async def get_subqueries(self, name):
        myprompt = SUBQUERIES_PROMPT
        myprompt = myprompt.format(name=name)
        res = await self.client.get_completion(myprompt)
        if res.choices is not None:
            json_str = res.choices[0].message.content.strip()
            return json_str
        else:
            print("Couldn't get the answer")

    async def get_ref_subqueries(self, page):
        '''
        Функция для получения эталонного плана статьи по данным заголовкам
        '''
        if not page.outline:
            print('Empty page!')
            return
        myprompt = REFERENCE_SUBQUERIES_ALL_HEADERS_PROMPT
        headers = [header[1] for header in page.outline.keys() if header[0] == 'h2']
        headers_formatted = "\n".join(f"- {h}" for h in headers)
        myprompt = myprompt.format(article_title=page.name, headers=headers_formatted)
        res = await self.client.get_completion(myprompt)
        
        if res.choices is not None:
            response = res.choices[0].message.content.strip()
            return (page.name, response)
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
        
    async def filter_sources(self, name, subqueries, texts):
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

        await probs.complete_couroutines(batch_size=20)
        final_probs = await probs.to_list()
        final_probs = sorted(final_probs)
        return final_probs

