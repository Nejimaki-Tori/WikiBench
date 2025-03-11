from openai import OpenAI
import os
import json


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

class WikiGen:
    def __init__(self, open_ai_client, model_name='qwen2.5-72b', repetition_penalty=1.0) -> None:
        self.client = open_ai_client
        self.model = model_name
        self.temperature = 0.01
        self.top_p = 0.9
        self.max_tokens=1000
        self.rep_pen = repetition_penalty
        self.extra_body={
            "repetition_penalty": self.rep_pen,
            "guided_choice": None,
            "add_generation_prompt": True,
            "guided_regex": None
        }

    
    def get_subqueries(self, name):
        myprompt = SUBQUERIES_PROMPT
        myprompt = myprompt.format(name=name)
        res = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": myprompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    # logprobs=True,
                    # top_logprobs=2,
                    max_tokens=self.max_tokens,
                    extra_body=self.extra_body
                )
        if res.choices is not None:
            json_str = res.choices[0].message.content.strip()
            return json_str
        else:
            print("Couldn't get the answer")
