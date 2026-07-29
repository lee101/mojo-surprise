from collections import namedtuple


class PredictionImpossible(Exception):
    pass


class Prediction(namedtuple("Prediction", ["uid", "iid", "r_ui", "est", "details"])):
    __slots__ = ()

    def __str__(self):
        true = f"{self.r_ui:1.2f}" if self.r_ui is not None else "None"
        return (
            f"user: {str(self.uid):<10} item: {str(self.iid):<10} "
            f"r_ui = {true}   est = {self.est:1.2f}   {self.details}"
        )

