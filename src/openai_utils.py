# -*- coding: utf-8 -*-

import asyncio
import inspect
from openai import AsyncOpenAI
import inspect

def extract_response_text(response):
    choices = getattr(response, 'choices', None)
    if not choices:
        return None

    first = choices[0]
    if hasattr(first, 'message'):
        text = first.message.content
    elif isinstance(first, dict) and 'message' in first:
        text = first['message']['content']
    else:
        return None

    if text is None:
        return None

    text = str(text).strip()
    text = re.sub(r'</?think>', '', text)
    
    return text.strip() or None


class LlmCompleter:
    def __init__(self, api_address, api_key, model_name='qwen2.5-72b', repetition_penalty=1.0) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_address)
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

    
    async def get_completion(self, query, system=None, examples=None,
                             choices=None, rep_penalty=1.0,
                             regex_pattern=None, max_tokens=512,
                             use_beam_search=False, beam_width=1,
                             answer_prefix=None):
        assert sum(map(lambda x: x is not None, [choices, regex_pattern])) < 2, "Only one guided mode is allowed"
        msgs = self.prepare_messages(query, system, examples, answer_prefix)
        needs_generation_start = answer_prefix is None

        if use_beam_search:
            beam_width = max(3, beam_width)
        completion = self.client.chat.completions.create(
            model=self.model,
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
        response = await completion
        return response

    
    async def get_probability(self, query, system=None, examples=None,
                              choices=None, rep_penalty=1.0,
                              regex_pattern=None, max_tokens=10):
        assert sum(map(lambda x: x is not None, [choices, regex_pattern])) < 2, "Only one guided mode is allowed"
        msgs = self.prepare_messages(query, system, examples, None)

        completion = self.client.chat.completions.create(
            model=self.model,
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
        response = await completion
        return response
        #print(msgs)
        #print(response.choices[0].message.content.strip())
        #print()
        #print()
        ### response = response.choices[0]
        ### ch, probs = list(
        ###     zip(*((tok.token, tok.logprob) for tok in response.logprobs.content[0].top_logprobs))
        ### )
        ### probs = softmax(probs).tolist()
        ### response = dict(zip(ch, probs))
        ### #print(response)
        ### return response

class AsyncList:
    def __init__(self):
        self.contents = []
        self.couroutine_ids = []

    def append(self, item):
        self.contents.append(item)
        if inspect.iscoroutine(item):
            self.couroutine_ids.append(len(self.contents) - 1)

    async def complete_couroutines(self, batch_size=10):
        while len(self.couroutine_ids) > 0:
            tasks = [self.contents[i] for i in self.couroutine_ids[:batch_size]]
            res = await asyncio.gather(*tasks)
            for i, r in zip(self.couroutine_ids, res):
                self.contents[i] = r
            self.couroutine_ids = self.couroutine_ids[batch_size:]

    def __getitem__(self, key):
        return self.contents[key]

    def __repr__(self):
        return repr(self.contents)

    async def to_list(self):
        await self.complete_couroutines(batch_size=1)
        return self.contents