class SparsePrint:
    def __init__(self):
        self.counter = 0

    def print(self, messages, nmod=1):
        self.counter += 1
        if self.counter % nmod == 0:
            print(messages)
            self.counter = 0

    def __call__(self, *args, **kwargs):
        self.print(*args)