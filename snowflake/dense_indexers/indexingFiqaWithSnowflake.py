import pyterrier as pt
from pathlib import Path
from snowflake_encoder import SnowFlakeDocumentEncoder, SnowFlakeQueryEncoder
import torch
from fast_forward import OnDiskIndex, Mode, Indexer


def docs_iter(dataset):
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"], "text": d["text"]}

# def docs_iter(dataset, limit=10):
#     # Iterate over the documents and yield up to 'limit' documents
#     for count, d in enumerate(dataset.get_corpus_iter()):
#         if count >= limit:
#             break
#         yield {"doc_id": d["docno"], "text": d["text"]}


def main():
    if not pt.started():
        pt.init(tqdm="notebook")

    dataset = pt.get_dataset("irds:beir/fiqa")

    if torch.cuda.is_available():
        print("daaa")

    q_encoder = SnowFlakeQueryEncoder("Snowflake/snowflake-arctic-embed-m")
    d_encoder = SnowFlakeDocumentEncoder(
        "Snowflake/snowflake-arctic-embed-m",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    ff_index = OnDiskIndex(
        Path("../ffindex_fiqa_snowflake_m.h5"), dim=768, query_encoder=q_encoder, mode=Mode.MAXP
    )

    ff_indexer = Indexer(ff_index, d_encoder, batch_size=8)
    ff_indexer.index_dicts(docs_iter(dataset))


if __name__ == '__main__':
    main()
