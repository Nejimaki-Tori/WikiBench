import asyncio
from tqdm.auto import tqdm
from openai import AsyncOpenAI
from scipy.special import softmax
import inspect


class LlmCompleter:
    def __init__(self, api_address, api_key, model_name_or_path):
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_address)
        self.model_name = model_name_or_path
        self.path = model_name_or_path
        # if "workdir" not in model_name_or_path:
        #     self.path = f'/workdir/models/{self.model_name}'

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
        # print(query)
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
        response = await completion
        response = response.choices[0].message.content.strip()
        return response

    async def get_probability(self, query, system=None, examples=None,
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
        response = await completion
        response = response.choices[0]
        ch, probs = list(
            zip(*((tok.token, tok.logprob) for tok in response.logprobs.content[0].top_logprobs))
        )
        probs = softmax(probs).tolist()
        response = dict(zip(ch, probs))
        return response

    async def batched_completion(self, idx, prompt, get_probs=False, **kwargs):
        if get_probs:
            res = await self.get_probability(prompt, **kwargs)
        else:
            res = await self.get_completion(prompt, **kwargs)
        return idx, res


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