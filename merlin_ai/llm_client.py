import openai


class MerlinChatCompletion(openai.ChatCompletion):
    @classmethod
    def create(cls, *args, **kwargs):
        return super().create(*args, **kwargs)

    @classmethod
    async def acreate(cls, *args, **kwargs):
        return await super().acreate(*args, **kwargs)
