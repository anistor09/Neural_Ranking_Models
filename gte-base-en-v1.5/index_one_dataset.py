import pyterrier as pt
from pathlib import Path
from gte_base_en_encoder import GTEBaseDocumentEncoder
import torch
from fast_forward import OnDiskIndex, Mode, Indexer


def docs_iter(dataset):
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"], "text": d["text"]}

# def docs_iter(dataset, limit=1000):
#     # Iterate over the documents and yield up to 'limit' documents
#     for count, d in enumerate(dataset.get_corpus_iter()):
#         if count >= limit:
#             break
#         yield {"doc_id": d["docno"], "text": d["text"]}


def format_filename(text):
    if '/' in text:
        return text.split('/')[0] + "_" + text.split('/')[1]
    else:
        return text


def index_collection(dataset_name):
    if not pt.started():
        pt.init(tqdm="notebook")

    prefix = "irds:beir/"

    full_path = prefix + dataset_name
    dataset = pt.get_dataset(full_path)

    if torch.cuda.is_available():
        print("daaa")

    q_encoder = GTEBaseDocumentEncoder("Alibaba-NLP/gte-base-en-v1.5")
    d_encoder = GTEBaseDocumentEncoder(
        "Alibaba-NLP/gte-base-en-v1.5",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_path = "indexes/ffindex_" + format_filename(dataset_name)+ "_gte-base-en-v1.5.h5"
    ff_index = OnDiskIndex(
        Path(index_path), dim=768, query_encoder=q_encoder, mode=Mode.MAXP, max_id_length=47
    )

    ff_indexer = Indexer(ff_index, d_encoder, batch_size=8)
    ff_indexer.index_dicts(docs_iter(dataset))


def main():
    dataset_name = "arguana"
    index_collection(dataset_name)


if __name__ == '__main__':
    main()
