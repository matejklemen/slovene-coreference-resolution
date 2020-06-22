import torch
import argparse
import logging
import codecs
import os
from data import read_corpus
from utils import fixed_split, extract_vocab
from noncontextual_model import NoncontextualController


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--fc_hidden_size", type=int, default=150)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--source_dataset", type=str, default="coref149")
parser.add_argument("--target_dataset", type=str, default="senticoref")
parser.add_argument("--fixed_split", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    src_docs = read_corpus(args.source_dataset)
    tgt_docs = read_corpus(args.target_dataset)

    tgt_train, tgt_dev, tgt_test = fixed_split(tgt_docs, args.target_dataset)
    combined_train = src_docs + tgt_train
    logging.info(f"Using fixed dataset split: {len(combined_train)} train, {len(tgt_dev)} dev, {len(tgt_test)} test docs")

    tok2id, id2tok = extract_vocab(combined_train, lowercase=True)
    # Note: pretrained word2vec embeddings we use are uncased
    logging.info("Loading pretrained Slovene word2vec embeddings")
    with codecs.open(os.path.join("..", "data", "model.txt"), "r", encoding="utf-8", errors="ignore") as f:
        num_tokens, embedding_size = list(map(int, f.readline().split(" ")))
        embs = {}
        for line in f:
            stripped_line = line.strip().split(" ")
            embs[stripped_line[0]] = list(map(lambda num: float(num), stripped_line[1:]))

    pretrained_embs = torch.rand((len(tok2id), embedding_size))
    for curr_token, curr_id in tok2id.items():
        # leave out-of-vocab token embeddings as random [0, 1) vectors
        pretrained_embs[curr_id, :] = torch.tensor(embs.get(curr_token.lower(), pretrained_embs[curr_id, :]),
                                                   device=DEVICE)

    model = NoncontextualController(model_name=f"nc_w2v{embedding_size}_unfrozen_hs{args.fc_hidden_size}_"
                                               f"{args.source_dataset}_{args.target_dataset}_"
                                               f"{args.learning_rate}_dr{args.dropout}",
                                    vocab=tok2id,
                                    embedding_size=embedding_size,
                                    dropout=args.dropout,
                                    fc_hidden_size=args.fc_hidden_size,
                                    learning_rate=args.learning_rate,
                                    pretrained_embs=pretrained_embs,
                                    freeze_pretrained=False,
                                    dataset_name=args.target_dataset)
    if not model.loaded_from_file:
        model.train(epochs=args.num_epochs, train_docs=combined_train, dev_docs=tgt_dev)
        # Reload best checkpoint
        model._prepare()

    model.evaluate(tgt_test)
    model.visualize()
