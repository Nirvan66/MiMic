class Smart:
    def __init__(self, preProcessor, pairs):
        self.preProcessor = preProcessor
        self._pairs = pairs

    def chat(self, sentence):
        for (statement, response) in self._pairs:
            if statement == sentence:
                return response
        return "I don't know what you said!"
