from openai import OpenAI
import os
import json
from tqdm.auto import tqdm

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

RANKING_PROMPT = """
Ты - эксперт в написании статей Википедии и оцениываешь найденные документы: подходят ли они для написания статьи на заданную тему. Тебе нужно по данному названию статьи и тексту статьи-источника определить, является ли он релевантным для этой темы. Текст должен быть 

Тема:
{article_name}
Запрос:
{source_text}

Соответствует ли запрос теме? Начни свой ответ с Да или Нет.
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


    def prepare_messages(self, query, system, examples, answer_prefix):
        msgs = []
        if system is not None:
            msgs += [{"role": "system", "content": system}]
        if examples is not None:
            assert isinstance(examples, list)
            for q, a in examples:
                msgs += [{"role": "user", "content": q}]
                msgs += [{"role": "assistant", "content": a}]
        msgs += [{"role": "user", "content": query}]
        if answer_prefix is not None:
            msgs += [{"role": "assistant", "content": answer_prefix}]
        return msgs

    def get_completion(self, query, system=None, examples=None,
                             choices=None, rep_penalty=1.0,
                             regex_pattern=None, max_tokens=512,
                             use_beam_search=False, beam_width=1,
                             answer_prefix=None):
        assert sum(map(lambda x: x is not None, [choices, regex_pattern])) < 2, "Only one guided mode is allowed"
        msgs = self.prepare_messages(query, system, examples, answer_prefix)

        # print(await self.client.models.list())

        needs_generation_start = answer_prefix is None

        if use_beam_search:
            beam_width = max(3, beam_width)
        completion = self.client.chat.completions.create(
            model=self.path,
            messages=msgs,
            temperature=0.001,
            top_p=0.9,
            max_tokens=max_tokens,
            n=beam_width,
            extra_body={
                "repetition_penalty": rep_penalty,
                "guided_choice": choices,
                "add_generation_prompt": needs_generation_start,
                "continue_final_message": not needs_generation_start,
                "guided_regex": regex_pattern,
                "use_beam_search": use_beam_search,
            }
        )
        response = completion
        response = response.choices[0].message.content.strip()
        return response

    def get_probability(self, query, system=None, examples=None,
                              choices=None, rep_penalty=1.0,
                              regex_pattern=None, max_tokens=10):
        assert sum(map(lambda x: x is not None, [choices, regex_pattern])) < 2, "Only one guided mode is allowed"
        msgs = self.prepare_messages(query, system, examples, None)

        # assert choices is not None
        # print(await self.client.models.list())

        completion = self.client.chat.completions.create(
            model=self.path,
            messages=msgs,
            temperature=0.0,
            top_p=1,
            logprobs=True,
            top_logprobs=10,
            max_tokens=max_tokens,
            extra_body={
                "repetition_penalty": 1.0,
                "guided_choice": choices,
                "add_generation_prompt": True,
            }
        )
        response = completion
        response = response.choices[0]
        ch, probs = list(
            zip(*((tok.token, tok.logprob) for tok in response.logprobs.content[0].top_logprobs))
        )
        probs = softmax(probs).tolist()
        response = dict(zip(ch, probs))
        return response

    def batched_completion(self, idx, prompt, get_probs=False, **kwargs):
        if get_probs:
            res = self.get_probability(prompt, **kwargs)
        else:
            res = self.get_completion(prompt, **kwargs)
        return idx, res

    def submit_task_batch(self, prompts, **kwargs):
        tasks = []
        for p, prompt in enumerate(prompts):
            t = asyncio.create_task(self.batched_completion(p, prompt, **kwargs))
            tasks.append(t)

        return tasks

    def fetch_tasks_batch_results(self, tasks, disable_tqdm=True):
        results = []
        for task in tqdm(
            asyncio.as_completed(tasks),
            disable=disable_tqdm,
            total=len(tasks)
        ):
            value = task
            results.append(value)
        results = sorted(results)
        return results
