from __future__ import annotations

import itertools
import os
from collections import defaultdict

from .trainset import Trainset


class Dataset:
    def __init__(self, reader=None):
        self.reader = reader

    @classmethod
    def load_from_df(cls, df, reader):
        return DatasetAutoFolds(reader=reader, df=df)

    @classmethod
    def load_from_file(cls, file_path, reader):
        return DatasetAutoFolds(ratings_file=file_path, reader=reader)

    @classmethod
    def load_builtin(cls, *args, **kwargs):
        raise NotImplementedError(
            "bundled dataset downloaders are outside mojo-surprise's covered subset"
        )

    def read_ratings(self, file_name):
        with open(os.path.expanduser(file_name), encoding="utf-8") as handle:
            return [
                self.reader.parse_line(line)
                for line in itertools.islice(handle, self.reader.skip_lines, None)
            ]

    def construct_trainset(self, raw_trainset):
        raw_users = {}
        raw_items = {}
        ur = defaultdict(list)
        ir = defaultdict(list)
        for raw_user, raw_item, rating, _ in raw_trainset:
            user = raw_users.setdefault(raw_user, len(raw_users))
            item = raw_items.setdefault(raw_item, len(raw_items))
            ur[user].append((item, float(rating)))
            ir[item].append((user, float(rating)))
        return Trainset(
            ur,
            ir,
            len(raw_users),
            len(raw_items),
            len(raw_trainset),
            self.reader.rating_scale,
            raw_users,
            raw_items,
        )

    def construct_testset(self, raw_testset):
        return [(user, item, rating) for user, item, rating, _ in raw_testset]


class DatasetAutoFolds(Dataset):
    def __init__(self, ratings_file=None, reader=None, df=None):
        super().__init__(reader)
        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(ratings_file)
        elif df is not None:
            self.df = df
            self.raw_ratings = [
                (user, item, float(rating), None)
                for user, item, rating in df.itertuples(index=False)
            ]
        else:
            raise ValueError("Must specify ratings file or dataframe.")

    def build_full_trainset(self):
        return self.construct_trainset(self.raw_ratings)

