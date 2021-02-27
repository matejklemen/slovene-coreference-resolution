import codecs
import logging
import sys

import numpy as np
import torch
from sklearn.model_selection import KFold

from data import read_corpus
from noncontextual_model import NoncontextualController, parser
from utils import fixed_split, extract_vocab, split_into_sets, KFoldStateCache

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser.add_argument("--source_dataset", type=str, default="senticoref")
parser.add_argument("--target_dataset", type=str, default="coref149")
parser.add_argument("--kfold_state_cache_path", type=str, default=None)

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

    all_tok2id, _ = extract_vocab(src_docs + tgt_docs, lowercase=True, top_n=10**9)
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
                                       dataset_name=override_kwargs.get("dataset", args.target_dataset))

    if args.target_dataset == "coref149":
        INNER_K, OUTER_K = 3, 10
        logging.info(f"Performing {OUTER_K}-fold (outer) and {INNER_K}-fold (inner) CV...")

        save_path = "cache_run_noncontextual_combined.json"
        if args.kfold_state_cache_path is None:
            train_test_folds = KFold(n_splits=OUTER_K, shuffle=True).split(tgt_docs)
            train_test_folds = [{
                "train_docs": [tgt_docs[_i].doc_id for _i in train_dev_index],
                "test_docs": [tgt_docs[_i].doc_id for _i in test_index]
            } for train_dev_index, test_index in train_test_folds]

            fold_cache = KFoldStateCache(script_name="run_noncontextual_combined.py",
                                         script_args=vars(args),
                                         main_dataset=args.target_dataset,
                                         additional_dataset=args.source_dataset,
                                         fold_info=train_test_folds)
        else:
            fold_cache = KFoldStateCache.from_file(args.kfold_state_cache_path)
            OUTER_K = fold_cache.num_folds

        for curr_fold_data in fold_cache.get_next_unfinished():
            curr_train_dev_docs = list(filter(lambda doc: doc.doc_id in set(curr_fold_data["train_docs"]), tgt_docs))
            curr_test_docs = list(filter(lambda doc: doc.doc_id in set(curr_fold_data["test_docs"]), tgt_docs))

            curr_tok2id, _ = extract_vocab(src_docs + curr_train_dev_docs, lowercase=True, top_n=args.max_vocab_size)
            curr_tok2id = {tok: all_tok2id[tok] for tok in curr_tok2id}
            logging.info(f"Fold#{curr_fold_data['idx_fold']} vocabulary size: {len(curr_tok2id)} tokens")

            best_metric, best_name = float("inf"), None
            for idx_inner_fold, (train_index, dev_index) in enumerate(KFold(n_splits=INNER_K).split(curr_train_dev_docs)):
                curr_train_docs = src_docs + [curr_train_dev_docs[_i] for _i in train_index]
                curr_dev_docs = [curr_train_dev_docs[_i] for _i in dev_index]

                curr_model = create_model_instance(
                    model_name=f"fold{curr_fold_data['idx_fold']}_{idx_inner_fold}",
                    tok2id=curr_tok2id
                )
                dev_loss = curr_model.train(epochs=args.num_epochs, train_docs=curr_train_docs, dev_docs=curr_dev_docs)
                logging.info(f"Fold {curr_fold_data['idx_fold']}-{idx_inner_fold}: {dev_loss: .5f}")
                if dev_loss < best_metric:
                    best_metric = dev_loss
                    best_name = curr_model.path_model_dir

            logging.info(f"Best model: {best_name}, best loss: {best_metric: .5f}")
            curr_model = NoncontextualController.from_pretrained(best_name)
            curr_test_metrics = curr_model.evaluate(curr_test_docs)
            curr_model.visualize()

            curr_test_metrics_expanded = {}
            for metric, metric_value in curr_test_metrics.items():
                curr_test_metrics_expanded[f"{metric}_p"] = float(metric_value.precision())
                curr_test_metrics_expanded[f"{metric}_r"] = float(metric_value.recall())
                curr_test_metrics_expanded[f"{metric}_f1"] = float(metric_value.f1())
            fold_cache.add_results(idx_fold=curr_fold_data["idx_fold"], results=curr_test_metrics_expanded)
            fold_cache.save(save_path)

        logging.info(f"Final scores (over {OUTER_K} folds)")
        aggregated_metrics = {}
        for curr_fold_data in fold_cache.fold_info:
            for metric, metric_value in curr_fold_data["results"].items():
                existing = aggregated_metrics.get(metric, [])
                existing.append(metric_value)

                aggregated_metrics[metric] = existing

        for metric, metric_values in aggregated_metrics.items():
            logging.info(f"- {metric}: mean={np.mean(metric_values): .4f} +- sd={np.std(metric_values): .4f}\n"
                         f"\t all fold scores: {metric_values}")
    else:
        logging.info(f"Using single train/dev/test split...")
        if args.fixed_split:
            logging.info("Using fixed dataset split")
            train_docs, dev_docs, test_docs = fixed_split(tgt_docs, args.target_dataset)
        else:
            train_docs, dev_docs, test_docs = split_into_sets(tgt_docs, train_prop=0.7, dev_prop=0.15, test_prop=0.15)

        combined_train = src_docs + train_docs
        curr_tok2id, _ = extract_vocab(combined_train, lowercase=True, top_n=args.max_vocab_size)
        curr_tok2id = {tok: all_tok2id[tok] for tok in curr_tok2id}

        model = create_model_instance(args.model_name, tok2id=curr_tok2id)
        model.train(epochs=args.num_epochs, train_docs=combined_train, dev_docs=dev_docs)
        # Reload best checkpoint
        model = NoncontextualController.from_pretrained(model.path_model_dir)

        model.evaluate(test_docs)
        model.visualize()
