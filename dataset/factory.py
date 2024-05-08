from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pascal_voc import pascal_voc

__sets = {}

for year in ['2007']:
    for split in ['trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))


def get_imdb(name):
    return __sets[name]()


def list_imdbs():
    return list(__sets.keys())
