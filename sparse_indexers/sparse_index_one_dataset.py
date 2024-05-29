import pyterrier as pt
from general_dense_indexers.dense_index_one_dataset import get_dataset_name


def docs_iter(dataset):
    for d in dataset.get_corpus_iter():
        yield {"text": d["text"], "title": d["title"], "docno": d["docno"].encode("utf-8")}


def index_one(prefix_dataset, dataset_name, max_doc_id_length):
    if not pt.started():
        pt.init(tqdm="notebook")

    dataset = pt.get_dataset(prefix_dataset + dataset_name)

    index_path = "./sparse_indexes/sparse_index_" + get_dataset_name(prefix_dataset + dataset_name)
    indexer = pt.IterDictIndexer(index_path, meta={'docno': max_doc_id_length})

    if 'dbpedia' in dataset_name or 'fever' in dataset_name:
        indexer.index(docs_iter(dataset), fields=["text"])
    else:
        indexer.index(dataset.get_corpus_iter(), fields=["text"])


def main():
    prefix_dataset = "irds:beir/"
    dataset_name = "fever"
    max_id_length = 221
    index_one(prefix_dataset, dataset_name, max_id_length)


if __name__ == '__main__':
    main()
