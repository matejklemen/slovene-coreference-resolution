import neleval.coref_metrics as metrics


def muc(gold, resp):
    return metrics._prf(*metrics.muc(gold, resp))


def b_cubed(gold, resp):
    truegold, trueresp = ({1: {'A', 'B', 'C', 'D'}}, {1: {'A', 'B'}, 2: {'C', 'D'}})
    return metrics._prf(*metrics.b_cubed(gold, resp))


def ceaf_e(gold, resp):
    return metrics._prf(*metrics.entity_ceaf(gold, resp))