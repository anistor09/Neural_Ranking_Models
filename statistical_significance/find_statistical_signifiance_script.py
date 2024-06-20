import pyterrier as pt
from pathlib import Path
import os
from ranx import compare, Run, Qrels

# datasets_names = ['fiqa', 'nfcorpus', 'scifact', 'quora', 'hotpotqa', 'dbpedia', 'fever', 'msmarco-passage']

datasets_names = ['nfcorpus']

models = ["bge-base-en-v1.5",
          "bge-small-en-v1.5",
          "e5-base-unsupervised",
          "e5-base-v2",
          "e5-small-v2",
          "gte-base-en-v1.5",
          "nomic-embed-text-v1",
          "snowflake-arctic-embed-m",
          "snowflake-arctic-embed-xs",
          "tct_colbert_msmarco"]


def main():
    # Initialize PyTerrier
    if not pt.started():
        pt.init(tqdm="notebook")

    for dataset_name in datasets_names:

        path = os.path.abspath(os.getcwd()) + "/results/trec_runs/" + dataset_name + "/"

        runs = []
        for model_name in models:
            run_files = Run.from_file(Path(path + model_name + ".trec").resolve().as_posix())

            runs.append(run_files)

    if "msmarco" not in dataset_name:
        qrels = pt.get_dataset(f'irds:beir/{dataset_name}/test').get_qrels()
    else:
        qrels = pt.get_dataset('irds:msmarco-passage/trec-dl-2019').get_qrels()

    qrels = Qrels.from_df(qrels, q_id_col='qid', doc_id_col='docno', score_col='label')

    report = compare(qrels, runs, metrics=["mrr@10", "ndcg@10"], max_p=0.05,
                     stat_test='student', make_comparable=True)

    print(report)

    report.save(path + "/../../significance_reports/" + dataset_name + ".json")


if __name__ == '__main__':
    main()
