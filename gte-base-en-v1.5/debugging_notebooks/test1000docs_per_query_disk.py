from memory_profiler import profile
import pyterrier as pt
from pyterrier.measures import RR, nDCG, MAP
from gte_base_en_encoder import GTEBaseDocumentEncoder
import torch
from fast_forward import OnDiskIndex, Mode
from pathlib import Path
from fast_forward.util.pyterrier import FFScore
from fast_forward.util.pyterrier import FFInterpolate


def run_test():
    if not pt.started():
        pt.init(tqdm="notebook")
    index_path = "./sparse_index_fiqa"
    # Load index to memory
    index = pt.IndexFactory.of(index_path, memory=True)

    bm25 = pt.BatchRetrieve(index, wmodel="BM25")

    testset = pt.get_dataset("irds:beir/fiqa/test")

    q_encoder = GTEBaseDocumentEncoder("Alibaba-NLP/gte-base-en-v1.5")

    ff_index = OnDiskIndex.load(
        Path("./dense_index_fiqa_GTE-base/ffindex_fiqa_gte-base-en-v1.5.h5"), query_encoder=q_encoder, mode=Mode.MAXP
    ).to_memory()

    ff_score = FFScore(ff_index)
    ff_int = FFInterpolate(alpha=0.05)

    print(pt.Experiment(
        [~bm25 % 1000 >> ff_score >> ff_int],
        testset.get_topics(),
        testset.get_qrels(),
        eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
        names=["BM25 >> FF"]))


@profile
def main():
    run_test()


if __name__ == '__main__':
    main()
