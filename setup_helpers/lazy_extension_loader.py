"""Provides a way to define extensions without importing modules
not installed before setup
"""


class LazyList(list):
    """Evaluates extension list lazyly.
    pattern taken from http://tinyurl.com/qb8478q"""

    def __init__(self, generator):
        super().__init__()
        self._list = None
        self._generator = generator

    def get(self):
        if self._list is None:
            self._list = list(self._generator)
        return self._list

    def __iter__(self):
        for e in self.get():
            yield e

    def __getitem__(self, i):
        return self.get()[i]

    def __len__(self):
        return len(self.get())
