import neleval.coref_metrics as metrics


# input into metric functions should be formatted as dictionary of {int -> set(str)},
# where keys (ints) are clusters and values (string sets) are mentions in a cluster. Example:
# {
#  1: {'rc_1', 'rc_2', ...}
#  2: {'rc_5', 'rc_8', ...}
#  3: ...
# }


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


def conll_12(muc_score, b_cubed_score, ceaf_e_score):
    # CoNLL-12 metric is an average of MUC, B3 and CEAF metric.
    s = Score()
    s.add((muc_score.precision(), muc_score.recall(), muc_score.f1()))
    s.add((b_cubed_score.precision(), b_cubed_score.recall(), b_cubed_score.f1()))
    s.add((ceaf_e_score.precision(), ceaf_e_score.recall(), ceaf_e_score.f1()))
    return s


def muc(gold, resp):
    return metrics._prf(*metrics.muc(gold, resp))


def b_cubed(gold, resp):
    return metrics._prf(*metrics.b_cubed(gold, resp))


def ceaf_e(gold, resp):
    return metrics._prf(*metrics.entity_ceaf(gold, resp))