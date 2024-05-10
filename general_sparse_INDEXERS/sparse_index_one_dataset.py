from memory_profiler import profile
import pyterrier as pt


def index_one(prefix_dataset, dataset_name, model_name):
    if not pt.started():
        pt.init(tqdm="notebook")

    dataset = pt.get_dataset(prefix_dataset + dataset_name)

    index_path = "../" + model_name + "/sparse_indexes/sparse_index_" + model_name + "_" + dataset_name
    indexer = pt.IterDictIndexer(index_path)
    indexer.index(dataset.get_corpus_iter(), fields=["text"])


@profile
def main():
    dataset_name = "fiqa"
    model_name = "gte-base-en-v1.5"
    prefix_dataset = "irds:beir/"
    index_one(prefix_dataset, dataset_name, model_name)


if __name__ == '__main__':
    main()
