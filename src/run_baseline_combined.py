from baseline import BaselineController, MentionPairFeatures, AllInOneModel, EachInOwnModel
from data import read_corpus
from utils import fixed_split
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--source_dataset", type=str, default="coref149")  # {'senticoref', 'coref149'}
parser.add_argument("--target_dataset", type=str, default="senticoref")  # {'senticoref', 'coref149'}


logging.basicConfig()

if __name__ == "__main__":
    args = parser.parse_args()
    baseline = BaselineController(MentionPairFeatures.num_features(),
                                  model_name=f"baseline_{args.source_dataset}_{args.target_dataset}_{args.learning_rate}",
                                  learning_rate=args.learning_rate,
                                  dataset_name=args.target_dataset)

    src_docs = read_corpus(args.source_dataset)
    tgt_docs = read_corpus(args.target_dataset)

    tgt_train, tgt_dev, tgt_test = fixed_split(tgt_docs, args.target_dataset)
    combined_train = src_docs + tgt_train
    logging.info(f"Using fixed dataset split: {len(combined_train)} train, {len(tgt_dev)} dev, {len(tgt_test)} test docs")

    if not baseline.loaded_from_file:
        # train only if it was not loaded
        baseline.train(args.num_epochs, combined_train, tgt_dev)

    baseline.evaluate(tgt_test)

    aioModel = AllInOneModel(baseline)
    aioModel.evaluate(tgt_test)
    eioModel = EachInOwnModel(baseline)
    eioModel.evaluate(tgt_test)

    baseline.visualize()
