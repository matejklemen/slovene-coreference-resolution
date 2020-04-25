import neleval.coref_metrics as metrics


def muc(gold, resp):
    return metrics._prf(*metrics.muc(gold, resp))


def b_cubed(gold, resp):
    return metrics._prf(*metrics.b_cubed(gold, resp))


def ceaf_e(gold, resp):
    return metrics._prf(*metrics.entity_ceaf(gold, resp))