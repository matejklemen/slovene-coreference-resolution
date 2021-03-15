import argparse
import codecs
import json
import logging
import os
import time
from typing import Optional, Mapping, Iterable, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

from common import ControllerBase, NeuralCoreferencePairScorer
from data import read_corpus, Document
from utils import extract_vocab, split_into_sets, fixed_split

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)

parser.add_argument("--fc_hidden_size", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.4)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=30)

parser.add_argument("--dataset", type=str, default="senticoref")
parser.add_argument("--max_vocab_size", type=int, default=9_999_999,
                    help="Limit the maximum vocabulary size. Set to a high number if you don't want to limit it")
parser.add_argument("--use_pretrained_embs", type=str, default="fastText", choices=["fastText", "word2vec", None],
                    help="Which (if any) pretrained embeddings to use")
parser.add_argument("--embedding_path", type=str,
                    default="/home/matej/Documents/projects/slovene-coreference-resolution/data/ft_sl_reduced100")
parser.add_argument("--embedding_size", type=int, default=None,
                    help="Size of word embeddings. Required if --use_pretrained_embs is None")
parser.add_argument("--freeze_pretrained", action="store_true")
parser.add_argument("--random_seed", type=int, default=13)
parser.add_argument("--fixed_split", action="store_true")


logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class FastTextEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, num_embeddings: int, embedding_dim: int, word2inds: Mapping):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.word2inds = {word: np.array(inds) for word, inds in word2inds.items()}

    @staticmethod
    def from_dir(model_dir):
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf8") as f_config:
            config = json.load(f_config)

        with open(os.path.join(model_dir, "word2inds.json"), "r", encoding="utf8") as f:
            word2inds = json.load(f)

        instance = FastTextEmbeddingBag(num_embeddings=config["num_embeddings"],
                                        embedding_dim=config["embedding_dim"],
                                        word2inds=word2inds)
        instance.load_state_dict(torch.load(os.path.join(model_dir, "embeddings.th")))
        return instance

    def save_pretrained(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf8") as f_config:
            json.dump({
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim
            }, fp=f_config, indent=4)

        with open(os.path.join(model_dir, "word2inds.json"), "w") as f:
            json.dump({word: inds.tolist() for word, inds in self.word2inds.items()},
                      fp=f, indent=4)

        torch.save(self.state_dict(),
                   os.path.join(model_dir, "embeddings.th"))

    def forward(self, words: Iterable[str]):
        word_subinds = np.empty([0], dtype=np.int64)
        word_offsets = [0]
        for word in words:
            subinds = self.word2inds[word]
            word_subinds = np.concatenate((word_subinds, subinds))
            word_offsets.append(word_offsets[-1] + len(subinds))
        word_offsets = word_offsets[:-1]
        ind = torch.tensor(word_subinds, dtype=torch.long, device=DEVICE)
        offsets = torch.tensor(word_offsets, dtype=torch.long, device=DEVICE)
        return super().forward(ind, offsets)


class NoncontextualController(ControllerBase):
    def __init__(self, vocab: Mapping,
                 dropout: float,
                 dataset_name: str,
                 fc_hidden_size: int = 150,
                 learning_rate: float = 0.001,
                 max_span_size: int = 10,
                 num_embeddings: Optional[int] = None,
                 embedding_size: Optional[int] = None,
                 embedding_type: Optional[str] = None,
                 pretrained_embs: Optional[Union[str, torch.Tensor]] = None,
                 freeze_pretrained: bool = False,
                 model_name: Optional[str] = None):
        """
        Parameters
        ----------
        vocab:
            Mapping from tokens (str) to IDs (int)
        dropout:
            Probability of dropout
        dataset_name:
            Used dataset for training/evaluation.
        fc_hidden_size:
            Size of the hidden layers in coreference scorer
        learning_rate:
            Learning rate used to train the model
        max_span_size:
            Span size, which all spans are padded/truncated to
        num_embeddings:
            The first dimension of embedding matrix. Set this explicitly if you want to initialize an embedding matrix
            larger than the vocabulary size
        embedding_size:
            The second dimension of embedding matrix. Only required if 'pretrained_embs' is None
        embedding_type:
            Type of embeddings used, either 'fastText', 'word2vec' or None (from scratch)
        pretrained_embs:
            Pretrained embeddings to be loaded into embedding module. If using fastText embeddings, should be the path
            to a fastText model.
        freeze_pretrained:
            Whether to keep embeddings frozen or not
        model_name:
            Name given to the model. If not given, constructed from current timestamp
        """
        effective_model_name = time.strftime("%Y%m%d_%H%M%S") if model_name is None else model_name
        self.path_model_dir = os.path.join(self.model_base_dir, effective_model_name)

        self.vocab = vocab
        self.dropout = dropout
        self.fc_hidden_size = fc_hidden_size
        self.max_span_size = max_span_size
        self.embedding_type = embedding_type
        self.freeze_pretrained = freeze_pretrained
        self.embeddings_path = None  # None or points to pretrained fastText

        eff_num_embeddings = num_embeddings if num_embeddings is not None else len(self.vocab)
        eff_embedding_size = embedding_size

        if pretrained_embs is None:
            if embedding_size is None:
                raise ValueError("'embedding_size' is required if pretrained embeddings are not provided")
            # Randomly initialize embeddings in [-0.001, 0.001]
            eff_pretrained_embs = -0.001 + (0.001 - (-0.001)) / (1.0 - 0.0) * \
                                  torch.rand((eff_num_embeddings, eff_embedding_size), dtype=torch.float32)
        else:
            eff_pretrained_embs = pretrained_embs

        if embedding_type == "fastText":
            self.embeddings_path = eff_pretrained_embs
            self.embedder = FastTextEmbeddingBag.from_dir(eff_pretrained_embs).to(DEVICE)
        elif embedding_type in ["word2vec", None]:
            self.embedder = nn.Embedding.from_pretrained(eff_pretrained_embs, freeze=freeze_pretrained).to(DEVICE)
        else:
            raise ValueError(f"'{embedding_type}' is not a valid embedding_type")

        if embedding_type == "fastText":
            # pass sequence (List[str]) directly to embedder as it internally splits words into subwords
            self.embed_sequence = lambda seq: self.embedder(seq).to(DEVICE)
        else:
            # encode sequence (List[int]), then pass it to embedder
            self.embed_sequence = lambda seq: self.embedder(torch.tensor([self.vocab.get(i, self.vocab["<UNK>"])
                                                                          for i in seq], device=DEVICE))

        self.num_embeddings = self.embedder.num_embeddings
        self.embedding_size = self.embedder.embedding_dim

        self.scorer = NeuralCoreferencePairScorer(num_features=self.embedding_size,
                                                  hidden_size=fc_hidden_size,
                                                  dropout=dropout).to(DEVICE)
        self.optimizer = optim.Adam(self.scorer.parameters() if freeze_pretrained else
                                    (list(self.scorer.parameters()) + list(self.embedder.parameters())),
                                    lr=learning_rate)

        super().__init__(learning_rate=learning_rate, dataset_name=dataset_name, model_name=effective_model_name)

    @staticmethod
    def from_pretrained(model_dir):
        vocab_path = os.path.join(model_dir, "vocab.txt")
        with open(vocab_path, "r") as f_vocab:
            pre_tok2id = {token.strip(): i for i, token in enumerate(f_vocab)}

        controller_config_path = os.path.join(model_dir, "controller_config.json")
        with open(controller_config_path, "r") as f_config:
            pre_config = json.load(f_config)

        instance = NoncontextualController(vocab=pre_tok2id,
                                           **pre_config)
        # Load the proper module states from files
        instance.load_checkpoint()

        return instance

    def save_pretrained(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Write vocabulary by ascending token ID, writing down gap tokens as `[unused<i>]`
        vocab_path = os.path.join(model_dir, "vocab.txt")
        with open(vocab_path, "w", encoding="utf-8") as f_vocab:
            id2tok = {idx: token for token, idx in self.vocab.items()}
            max_idx = max(list(id2tok.keys()))

            for idx in range(max_idx + 1):
                print(id2tok.get(idx, f"[unused{idx}]"), file=f_vocab)

        # Write controller config (used for instantiation)
        controller_config_path = os.path.join(model_dir, "controller_config.json")
        with open(controller_config_path, "w", encoding="utf-8") as f_config:
            json.dump({
                "model_name": self.model_name,
                "dropout": self.dropout,
                "dataset_name": self.dataset_name,
                "embedding_type": self.embedding_type,
                "num_embeddings": self.num_embeddings,
                "embedding_size": self.embedding_size,
                "pretrained_embs": os.path.join(model_dir, "fastText") if self.embedding_type == "fastText" else None,
                "fc_hidden_size": self.fc_hidden_size,
                "learning_rate": self.learning_rate,
                "max_span_size": self.max_span_size,
                "freeze_pretrained": self.freeze_pretrained
            }, fp=f_config, indent=4)

        # Write weights (module state)
        if self.embedding_type == "fastText":
            self.embedder.save_pretrained(os.path.join(model_dir, "fastText"))
        else:
            torch.save(self.embedder.state_dict(), os.path.join(model_dir, "embeddings.th"))
        torch.save(self.scorer.state_dict(), os.path.join(model_dir, "scorer.th"))

    @property
    def model_base_dir(self):
        return "noncontextual_model"

    def train_mode(self):
        self.embedder.train()
        self.scorer.train()

    def eval_mode(self):
        self.embedder.eval()
        self.scorer.eval()

    def load_checkpoint(self):
        """ Handles loading of weights for instantiated modules. """
        path_to_scorer = os.path.join(self.path_model_dir, "scorer.th")
        path_to_embeddings = os.path.join(self.path_model_dir, "embeddings.th")

        self.scorer.load_state_dict(torch.load(path_to_scorer, map_location=DEVICE))
        if self.embedding_type != "fastText":
            self.embedder.load_state_dict(torch.load(path_to_embeddings, map_location=DEVICE))
        self.loaded_from_file = True

    def save_checkpoint(self):
        logging.warning("save_checkpoint() is deprecated. Use save_pretrained() instead")
        self.save_pretrained(self.path_model_dir)

    def _prepare_doc(self, curr_doc: Document) -> Dict:
        """ Returns a cache dictionary with preprocessed data. This should only be called once per document, since
        data inside same document does not get shuffled. """
        ret = {}

        preprocessed_sents, max_len = [], 0
        for curr_sent in curr_doc.raw_sentences():
            # TODO: uncased/cased option
            curr_processed_sent = list(map(lambda s: s.lower().strip(), curr_sent)) + ["<PAD>"]
            preprocessed_sents.append(curr_processed_sent)
            if len(curr_processed_sent) > max_len:
                max_len = len(curr_processed_sent)

        for i in range(len(preprocessed_sents)):
            preprocessed_sents[i].extend(["<PAD>"] * (max_len - len(preprocessed_sents[i])))

        cluster_sets = []
        mention_to_cluster_id = {}
        for i, curr_cluster in enumerate(curr_doc.clusters):
            cluster_sets.append(set(curr_cluster))
            for mid in curr_cluster:
                mention_to_cluster_id[mid] = i

        all_candidate_data = []
        for idx_head, (head_id, head_mention) in enumerate(curr_doc.mentions.items(), start=1):
            gt_antecedent_ids = cluster_sets[mention_to_cluster_id[head_id]]

            # Note: no data for dummy antecedent (len(`features`) is one less than `candidates`)
            candidates, candidate_data = [None], []
            starts, ends = [], []
            candidate_attention = []
            correct_antecedents = []

            curr_head_data = [[], []]
            for curr_token in head_mention.tokens:
                curr_head_data[0].append(curr_token.sentence_index)
                curr_head_data[1].append(curr_token.position_in_sentence)

            num_tokens = len(head_mention.tokens)
            if num_tokens > self.max_span_size:
                curr_head_data[0] = curr_head_data[0][:self.max_span_size]
                curr_head_data[1] = curr_head_data[1][:self.max_span_size]
            else:
                curr_head_data[0] += [head_mention.tokens[0].sentence_index] * (self.max_span_size - num_tokens)
                curr_head_data[1] += [-1] * (self.max_span_size - num_tokens)

            head_start = 0
            head_end = num_tokens
            head_attention = torch.ones((1, self.max_span_size), dtype=torch.bool)
            head_attention[0, num_tokens:] = False

            for idx_candidate, (cand_id, cand_mention) in enumerate(curr_doc.mentions.items(), start=1):
                if idx_candidate >= idx_head:
                    break

                candidates.append(cand_id)

                # Maps tokens to positions inside document (idx_sent, idx_inside_sent) for efficient indexing later
                curr_candidate_data = [[], []]
                for curr_token in cand_mention.tokens:
                    curr_candidate_data[0].append(curr_token.sentence_index)
                    curr_candidate_data[1].append(curr_token.position_in_sentence)

                num_tokens = len(cand_mention.tokens)
                if num_tokens > self.max_span_size:
                    curr_candidate_data[0] = curr_candidate_data[0][:self.max_span_size]
                    curr_candidate_data[1] = curr_candidate_data[1][:self.max_span_size]
                else:
                    curr_candidate_data[0] += [cand_mention.tokens[0].sentence_index] * (self.max_span_size - num_tokens)
                    curr_candidate_data[1] += [-1] * (self.max_span_size - num_tokens)

                candidate_data.append(curr_candidate_data)
                starts.append(0)
                ends.append(num_tokens)

                curr_attention = torch.ones((1, self.max_span_size), dtype=torch.bool)
                curr_attention[0, num_tokens:] = False
                candidate_attention.append(curr_attention)

                is_coreferent = cand_id in gt_antecedent_ids
                if is_coreferent:
                    correct_antecedents.append(idx_candidate)

            if len(correct_antecedents) == 0:
                correct_antecedents.append(0)

            candidate_attention = torch.cat(candidate_attention) if len(candidate_attention) > 0 else []

            all_candidate_data.append({
                "head_id": head_id,
                "head_data": torch.tensor([curr_head_data]),
                "head_attention": head_attention,
                "head_start": head_start,
                "head_end": head_end,
                "candidates": candidates,
                "candidate_data": torch.tensor(candidate_data),
                "candidate_attention": candidate_attention,
                "correct_antecedents": correct_antecedents
            })

        ret["preprocessed_sents"] = preprocessed_sents
        ret["steps"] = all_candidate_data

        return ret

    def _train_doc(self, curr_doc: Document, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return {}, (0.0, 0)

        if not hasattr(curr_doc, "_cache_nc"):
            curr_doc._cache_nc = self._prepare_doc(curr_doc)
        cache = curr_doc._cache_nc  # type: dict

        embedded_doc = []
        for curr_sent in cache["preprocessed_sents"]:
            embedded_doc.append(self.embed_sequence(curr_sent))
        embedded_doc = torch.stack(embedded_doc)  # [num_sents, max_tokens_in_any_sent + 1, embedding_size]

        doc_loss, n_examples = 0.0, len(cache["steps"])
        preds = {}

        for curr_step in cache["steps"]:
            head_id = curr_step["head_id"]
            head_data = curr_step["head_data"]

            candidates = curr_step["candidates"]
            candidate_data = curr_step["candidate_data"]
            correct_antecedents = curr_step["correct_antecedents"]

            # Note: num_candidates includes dummy antecedent + actual candidates
            num_candidates = len(candidates)
            if num_candidates == 1:
                curr_pred = 0
            else:
                idx_sent = candidate_data[:, 0, :]
                idx_in_sent = candidate_data[:, 1, :]

                # [num_candidates, max_span_size, embedding_size]
                candidate_data = embedded_doc[idx_sent, idx_in_sent]
                # [1, head_size, embedding_size]
                head_data = embedded_doc[head_data[:, 0, :], head_data[:, 1, :]]
                head_data = head_data.repeat((num_candidates - 1, 1, 1))

                candidate_scores = self.scorer(candidate_data, head_data,
                                               curr_step["candidate_attention"],
                                               curr_step["head_attention"].repeat((num_candidates - 1, 1)))
                # [1, num_candidates]
                candidate_scores = torch.cat((torch.tensor([0.0], device=DEVICE),
                                              candidate_scores.flatten())).unsqueeze(0)

                curr_pred = torch.argmax(candidate_scores)
                doc_loss += self.loss(candidate_scores.repeat((len(correct_antecedents), 1)),
                                      torch.tensor(correct_antecedents, device=DEVICE))

            # { antecedent: [mention(s)] } pair
            existing_refs = preds.get(candidates[int(curr_pred)], [])
            existing_refs.append(head_id)
            preds[candidates[int(curr_pred)]] = existing_refs

        if not eval_mode:
            doc_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return preds, (float(doc_loss), n_examples)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.random_seed:
        torch.random.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    documents = read_corpus(args.dataset)
    all_tok2id, _ = extract_vocab(documents, lowercase=True, top_n=10**9)
    logging.info(f"Total vocabulary size: {len(all_tok2id)} tokens")

    pretrained_embs = None
    embedding_size = args.embedding_size

    if args.use_pretrained_embs == "word2vec":
        # Note: pretrained word2vec embeddings we use are uncased
        logging.info("Loading pretrained Slovene word2vec embeddings")
        with codecs.open(args.embedding_path, "r", encoding="utf-8", errors="ignore") as f:
            num_tokens, embedding_size = list(map(int, f.readline().split(" ")))
            embs = {}
            for line in f:
                stripped_line = line.strip().split(" ")
                embs[stripped_line[0]] = list(map(lambda num: float(num), stripped_line[1:]))

        pretrained_embs = torch.rand((len(all_tok2id), embedding_size))
        for curr_token, curr_id in all_tok2id.items():
            # leave out-of-vocab token embeddings as random [0, 1) vectors
            pretrained_embs[curr_id, :] = torch.tensor(embs.get(curr_token.lower(), pretrained_embs[curr_id, :]),
                                                       device=DEVICE)
    elif args.use_pretrained_embs == "fastText":
        pretrained_embs = args.embedding_path
    else:
        assert args.embedding_size is not None

    def create_model_instance(model_name, **override_kwargs):
        used_embedding_type = override_kwargs.get("use_pretrained_embs", args.use_pretrained_embs)
        used_embs = override_kwargs.get("pretrained_embs",
                                        pretrained_embs if used_embedding_type == "fastText" else pretrained_embs.clone())

        return NoncontextualController(model_name=model_name,
                                       vocab=override_kwargs.get("tok2id", all_tok2id),
                                       embedding_size=override_kwargs.get("embedding_size", embedding_size),
                                       dropout=override_kwargs.get("dropout", args.dropout),
                                       fc_hidden_size=override_kwargs.get("fc_hidden_size", args.fc_hidden_size),
                                       learning_rate=override_kwargs.get("learning_rate", args.learning_rate),
                                       embedding_type=used_embedding_type,
                                       pretrained_embs=used_embs,
                                       freeze_pretrained=override_kwargs.get("freeze_pretrained", args.freeze_pretrained),
                                       dataset_name=override_kwargs.get("dataset", args.dataset))

    # Train model
    if args.dataset == "coref149":
        INNER_K, OUTER_K = 3, 10
        logging.info(f"Performing {OUTER_K}-fold (outer) and {INNER_K}-fold (inner) CV...")
        test_metrics = {"muc_p": [], "muc_r": [], "muc_f1": [],
                        "b3_p": [], "b3_r": [], "b3_f1": [],
                        "ceafe_p": [], "ceafe_r": [], "ceafe_f1": [],
                        "avg_p": [], "avg_r": [], "avg_f1": []}

        for idx_outer_fold, (train_dev_index, test_index) in enumerate(KFold(n_splits=OUTER_K, shuffle=True).split(documents)):
            curr_train_dev_docs = [documents[_i] for _i in train_dev_index]
            curr_test_docs = [documents[_i] for _i in test_index]
            curr_tok2id, _ = extract_vocab(curr_train_dev_docs, lowercase=True, top_n=args.max_vocab_size)
            curr_tok2id = {tok: all_tok2id[tok] for tok in curr_tok2id}
            logging.info(f"Fold#{idx_outer_fold} vocabulary size: {len(curr_tok2id)} tokens")

            best_metric, best_name = float("inf"), None
            for idx_inner_fold, (train_index, dev_index) in enumerate(KFold(n_splits=INNER_K).split(curr_train_dev_docs)):
                curr_train_docs = [curr_train_dev_docs[_i] for _i in train_index]
                curr_dev_docs = [curr_train_dev_docs[_i] for _i in dev_index]

                curr_model = create_model_instance(
                    model_name=f"fold{idx_outer_fold}_{idx_inner_fold}",
                    tok2id=curr_tok2id
                )
                dev_loss = curr_model.train(epochs=args.num_epochs, train_docs=curr_train_docs, dev_docs=curr_dev_docs)
                logging.info(f"Fold {idx_outer_fold}-{idx_inner_fold}: {dev_loss: .5f}")
                if dev_loss < best_metric:
                    best_metric = dev_loss
                    best_name = curr_model.path_model_dir

            logging.info(f"Best model: {best_name}, best loss: {best_metric: .5f}")
            curr_model = NoncontextualController.from_pretrained(best_name)
            curr_test_metrics = curr_model.evaluate(curr_test_docs)
            curr_model.visualize()
            for metric, metric_value in curr_test_metrics.items():
                test_metrics[f"{metric}_p"].append(float(metric_value.precision()))
                test_metrics[f"{metric}_r"].append(float(metric_value.recall()))
                test_metrics[f"{metric}_f1"].append(float(metric_value.f1()))

        logging.info(f"Final scores (over {OUTER_K} folds)")
        for metric, metric_values in test_metrics.items():
            logging.info(f"- {metric}: mean={np.mean(metric_values): .4f} +- sd={np.std(metric_values): .4f}\n"
                         f"\t all fold scores: {metric_values}")
    else:
        logging.info(f"Using single train/dev/test split...")
        if args.fixed_split:
            logging.info("Using fixed dataset split")
            train_docs, dev_docs, test_docs = fixed_split(documents, args.dataset)
        else:
            train_docs, dev_docs, test_docs = split_into_sets(documents, train_prop=0.7, dev_prop=0.15,
                                                              test_prop=0.15)
        curr_tok2id, _ = extract_vocab(train_docs, lowercase=True, top_n=args.max_vocab_size)
        curr_tok2id = {tok: all_tok2id[tok] for tok in curr_tok2id}

        model = create_model_instance(args.model_name, tok2id=curr_tok2id)
        model.train(epochs=args.num_epochs, train_docs=train_docs, dev_docs=dev_docs)
        # Reload best checkpoint
        model = NoncontextualController.from_pretrained(model.path_model_dir)

        model.evaluate(test_docs)
        model.visualize()

