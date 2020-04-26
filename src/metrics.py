import neleval.coref_metrics as metrics


class Score:

    def __init__(self):
        # precision, recall, f1
        self.prf = (0.0, 0.0, 0.0)
        self._add_count = 0
        pass

    def precision(self):
        return self.prf[0] / self._add_count

    def recall(self):
        return self.prf[1] / self._add_count

    def f1(self):
        return self.prf[2] / self._add_count

    def add(self, prf):
        # prf = tuple of (precision, recall, F1)
        # usually as a result of some metric function below (muc, b_cubed, ceaf_e)
        self.prf = (
                self.prf[0] + prf[0],
                self.prf[1] + prf[1],
                self.prf[2] + prf[2],
        )
        self._add_count += 1  # used for normalization

    def __str__(self):
        return f"prec={self.precision():.3f}, rec={self.recall():.3f}, f1={self.f1():.3f}"


def muc(gold, resp):
    return metrics._prf(*metrics.muc(gold, resp))


def b_cubed(gold, resp):
    return metrics._prf(*metrics.b_cubed(gold, resp))


def ceaf_e(gold, resp):
    return metrics._prf(*metrics.entity_ceaf(gold, resp))