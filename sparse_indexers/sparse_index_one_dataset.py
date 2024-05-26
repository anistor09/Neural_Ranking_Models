from memory_profiler import profile
import pyterrier as pt
from general_dense_indexers.dense_index_one_dataset import get_dataset_name


def index_one(prefix_dataset, dataset_name, max_doc_id_length):
    if not pt.started():
        pt.init(tqdm="notebook")

    dataset = pt.get_dataset(prefix_dataset + dataset_name)

    index_path = "./sparse_indexes/sparse_index_" + get_dataset_name(prefix_dataset + dataset_name)
    indexer = pt.IterDictIndexer(index_path, meta={'docno': max_doc_id_length})
    indexer.index(dataset.get_corpus_iter(), fields=["text"])


@profile
def main():
    prefix_dataset = "irds:beir/"
    dataset_name = "dbpedia-entity"
    max_id_length = 200
    index_one(prefix_dataset, dataset_name, max_id_length)


if __name__ == '__main__':
    main()
