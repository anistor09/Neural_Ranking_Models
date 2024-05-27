import pyterrier as pt
from pathlib import Path
import torch
from fast_forward import OnDiskIndex, Mode, Indexer
import re


def docs_iter(dataset):
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"].encode("utf-8"), "text": d["text"]}


def format_name(text):
    return re.sub(r'[:/.-]', '_', text)


def get_dataset_name(text):
    parts = re.split('[:/-]', text)

    if len(parts) > 2:
        dataset_name = ""
        for i in range(2, len(parts)):
            dataset_name += parts[i] + "_"

        if dataset_name.endswith('_'):
            return dataset_name[:-1]
        return dataset_name

    else:
        print("No third element available")


def index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=768,
                     mode=Mode.MAXP):
    if not pt.started():
        pt.init(tqdm="notebook")

    dataset = pt.get_dataset(dataset_name)

    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not available")

    index_path = directory + "/dense_indexes/ffindex_" + get_dataset_name(dataset_name) + "_" + format_name(
        model_name) + ".h5"

    ff_index = OnDiskIndex(
        Path(index_path), dim=dim, query_encoder=q_encoder, mode=mode, max_id_length=max_id_length
    )

    ff_indexer = Indexer(ff_index, d_encoder, batch_size=batch_size)
    ff_indexer.index_dicts(docs_iter(dataset))


def main():
    dataset_name = "scifact"
    index_collection(dataset_name)


if __name__ == '__main__':
    main()
