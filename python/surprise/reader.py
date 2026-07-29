from __future__ import annotations


class Reader:
    def __init__(
        self,
        name=None,
        line_format="user item rating",
        sep=None,
        rating_scale=(1, 5),
        skip_lines=0,
    ):
        if name is not None:
            raise ValueError("built-in dataset readers are not included in mojo-surprise")
        self.rating_scale = rating_scale
        self.sep = sep
        self.skip_lines = skip_lines
        self.indexes = {field: index for index, field in enumerate(line_format.split())}

    def parse_line(self, line):
        fields = line.split(self.sep)
        user = fields[self.indexes["user"]].strip()
        item = fields[self.indexes["item"]].strip()
        rating = float(fields[self.indexes["rating"]])
        timestamp = (
            fields[self.indexes["timestamp"]].strip()
            if "timestamp" in self.indexes
            else None
        )
        return user, item, rating, timestamp

