import json
import os


class Model(object):

    def __init__(self, name, phase):
        self.name = name
        self.phase = phase
        self.sample = []

    def fill_sample(self, path):
        self.sample = os.listdir(path)


class JsonObj(object):

    def __init__(self, path=None):
       self._items = []
       self._path = path

    def fill_items(self):
        dirs = os.listdir(self._path)
        for index, item in enumerate(dirs):
            path = os.path.join(self._path, item, 'blur')
            m = Model(item, 'test')
            m.fill_sample(path)
            self._items.append(m.__dict__)

    def obj_dump(self, path):
        f = open(path, 'w')
        json.dump(self._items, f)
        f.close()

    def load_json(self, path):
        f = open(path, 'r')
        self._items = json.load(f)
        f.close()
        return self
