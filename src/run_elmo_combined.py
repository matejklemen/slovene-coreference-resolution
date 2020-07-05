import torch
import argparse
import logging
from data import read_corpus
from utils import fixed_split, extract_vocab
from contextual_model_elmo import ContextualControllerELMo


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--fc_hidden_size", type=int, default=150)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--source_dataset", type=str, default="coref149")
parser.add_argument("--target_dataset", type=str, default="senticoref")
parser.add_argument("--fixed_split", action="store_true")
parser.add_argument("--freeze_pretrained", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    src_docs = read_corpus(args.source_dataset)
    tgt_docs = read_corpus(args.target_dataset)

    tgt_train, tgt_dev, tgt_test = fixed_split(tgt_docs, args.target_dataset)
    combined_train = src_docs + tgt_train
    logging.info(f"Using fixed dataset split: {len(combined_train)} train, {len(tgt_dev)} dev, {len(tgt_test)} test docs")

    model = ContextualControllerELMo(model_name=f"elmo_{args.source_dataset}_{args.target_dataset}_"
                                                f"hs{args.hidden_size}_fchs{args.fc_hidden_size}_"
                                                f"lr{args.learning_rate}_dr{args.dropout}",
                                     embedding_size=1024,
                                     fc_hidden_size=args.fc_hidden_size,
                                     hidden_size=args.hidden_size,
                                     dropout=args.dropout,
                                     pretrained_embs_dir="../data/slovenian-elmo",
                                     freeze_pretrained=args.freeze_pretrained,
                                     learning_rate=args.learning_rate,
                                     dataset_name=args.target_dataset)
    if not model.loaded_from_file:
        model.train(epochs=args.num_epochs, train_docs=combined_train, dev_docs=tgt_dev)
        # Reload best checkpoint
        model._prepare()

    model.evaluate(tgt_test)
    model.visualize()
