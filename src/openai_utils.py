import asyncio
import inspect


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