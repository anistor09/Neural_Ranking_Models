import pyterrier as pt
from general_dense_indexers.dense_index_one_dataset import get_dataset_name


def docs_iter(dataset):
    """
       Generator to iterate over documents in a dataset, yielding each document as a dictionary.
       It specifically encodes the 'docno' field into a string format of the utf encoding as Pyterrier accepts only
       String format for  docno.
    """
    for d in dataset.get_corpus_iter():
        yield {'docno': str(d['docno'].encode('utf-8')), 'text': d['text']}


def index_one(prefix_dataset, dataset_name, max_doc_id_length):
    """
        Configures and executes the SPARSE indexing process for a given dataset using PyTerrier.
        Creates a sparse index for the documents by handling the specifics based on the dataset name. For DBPedia
        and Fever, as they used characters such as è, é, ê, ë in the doc ids and an
        additional transformer is used. The elements are transformed in their byte encoding before
        adding them in the sparse index because otherwise Pyterrier truncates them and the doc ids from the first
        (sparse) stage and the second (dense) stage will not be the same.
    """
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
    max_id_length = 227
    index_one(prefix_dataset, dataset_name, max_id_length)


if __name__ == '__main__':
    main()
