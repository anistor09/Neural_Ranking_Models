from memory_profiler import profile
import pyterrier as pt


def index_one(prefix_dataset, dataset_name, model_name, max_doc_id_length):
    if not pt.started():
        pt.init(tqdm="notebook")

    dataset = pt.get_dataset(prefix_dataset + dataset_name)

    index_path = "../sparse_indexes/sparse_index_" + model_name + "_" + dataset_name
    indexer = pt.IterDictIndexer(index_path, meta={'docno': max_doc_id_length})
    indexer.index(dataset.get_corpus_iter(), fields=["text"])


# @profile
def main():
    dataset_name = "scifact"
    model_name = "gte-base-en-v1.5"
    prefix_dataset = "irds:beir/"
    index_one(prefix_dataset, dataset_name, model_name)


if __name__ == '__main__':
    main()
