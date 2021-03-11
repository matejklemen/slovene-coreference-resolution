import logging
import sys

import numpy as np
import torch
from sklearn.model_selection import KFold

from contextual_model_elmo import ContextualControllerELMo, parser
from data import read_corpus
from utils import fixed_split, split_into_sets, KFoldStateCache

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
parser.add_argument("--source_dataset", type=str, default="coref149")
parser.add_argument("--target_dataset", type=str, default="senticoref")


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
        INNER_K, OUTER_K = 3, 10
        logging.info(f"Performing {OUTER_K}-fold (outer) and {INNER_K}-fold (inner) CV...")

        save_path = "cache_run_elmo_combined.json"
        if args.kfold_state_cache_path is None:
            train_test_folds = KFold(n_splits=OUTER_K, shuffle=True).split(tgt_docs)
            train_test_folds = [{
                "train_docs": [tgt_docs[_i].doc_id for _i in train_dev_index],
                "test_docs": [tgt_docs[_i].doc_id for _i in test_index]
            } for train_dev_index, test_index in train_test_folds]

            fold_cache = KFoldStateCache(script_name="run_elmo_combined.py",
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
            logging.info(f"Fold#{curr_fold_data['idx_fold']}...")

            best_metric, best_name = float("inf"), None
            for idx_inner_fold, (train_index, dev_index) in enumerate(KFold(n_splits=INNER_K).split(curr_train_dev_docs)):
                curr_train_docs = src_docs + [curr_train_dev_docs[_i] for _i in train_index]
                curr_dev_docs = [curr_train_dev_docs[_i] for _i in dev_index]

                curr_model = create_model_instance(
                    model_name=f"fold{curr_fold_data['idx_fold']}_{idx_inner_fold}"
                )
                dev_loss = curr_model.train(epochs=args.num_epochs, train_docs=curr_train_docs, dev_docs=curr_dev_docs)
                logging.info(f"Fold {curr_fold_data['idx_fold']}-{idx_inner_fold}: {dev_loss: .5f}")
                if dev_loss < best_metric:
                    best_metric = dev_loss
                    best_name = curr_model.path_model_dir

            logging.info(f"Best model: {best_name}, best loss: {best_metric: .5f}")
            curr_model = ContextualControllerELMo.from_pretrained(best_name)
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

        model = create_model_instance(args.model_name)
        model.train(epochs=args.num_epochs, train_docs=combined_train, dev_docs=dev_docs)
        # Reload best checkpoint
        model = ContextualControllerELMo.from_pretrained(model.path_model_dir)
        model.evaluate(test_docs)
        model.visualize()
