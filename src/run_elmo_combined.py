import logging
import sys

import numpy as np
import torch

from contextual_model_elmo import ContextualControllerELMo, parser
from data import read_corpus
from utils import fixed_split, split_into_sets

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
parser.add_argument("--source_dataset", type=str, default="senticoref")
parser.add_argument("--target_dataset", type=str, default="coref149")


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    args = parser.parse_args()

    if args.random_seed:
        torch.random.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    src_docs = read_corpus(args.source_dataset)
    tgt_docs = read_corpus(args.target_dataset)

    def create_model_instance(model_name, **override_kwargs):
        _model = ContextualControllerELMo(model_name=model_name,
                                          fc_hidden_size=override_kwargs.get("fc_hidden_size", args.fc_hidden_size),
                                          hidden_size=override_kwargs.get("hidden_size", args.hidden_size),
                                          dropout=override_kwargs.get("dropout", args.dropout),
                                          pretrained_embeddings_dir="../data/slovenian-elmo",
                                          freeze_pretrained=override_kwargs.get("freeze_pretrained", args.freeze_pretrained),
                                          learning_rate=override_kwargs.get("learning_rate", args.learning_rate),
                                          max_segment_size=override_kwargs.get("max_segment_size", args.max_segment_size),
                                          layer_learning_rate={"lr_embedder": 10e-4} if not args.freeze_pretrained else None,
                                          dataset_name=args.target_dataset)
        return _model

    if args.target_dataset == "coref149":
        raise NotImplementedError()
    else:
        logging.info(f"Using single train/dev/test split...")
        if args.fixed_split:
            logging.info("Using fixed dataset split")
            train_docs, dev_docs, test_docs = fixed_split(tgt_docs, args.target_dataset)
        else:
            train_docs, dev_docs, test_docs = split_into_sets(tgt_docs, train_prop=0.7, dev_prop=0.15, test_prop=0.15)

        combined_train = src_docs + train_docs

        model = create_model_instance(args.model_name)
        model.train(epochs=args.num_epochs, train_docs=combined_train, dev_docs=dev_docs)
        # Reload best checkpoint
        model = ContextualControllerELMo.from_pretrained(model.path_model_dir)
        model.evaluate(test_docs)
        model.visualize()
