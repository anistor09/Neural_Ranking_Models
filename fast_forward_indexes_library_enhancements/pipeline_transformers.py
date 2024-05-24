import pandas as pd
import pyterrier as pt

from fast_forward.index import Index
from fast_forward.ranking import Ranking


class FFInterpolateNormalized(pt.Transformer):
    """PyTerrier transformer that interpolates scores computed by `FFScore`."""

    def __init__(self, alpha: float) -> None:
        """Create an FFInterpolate transformer.

        Args:
            alpha (float): The interpolation parameter.
        """
        # attribute name needs to be exactly this for pyterrier.GridScan to work
        self.alpha = alpha
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate the scores for all query-document pairs in the data frame as
        `alpha * score_0 + (1 - alpha) * score`.

        Args:
            df (pd.DataFrame): The PyTerrier data frame.

        Returns:
            pd.DataFrame: A new data frame with the interpolated scores.
        """
        new_df = df[["qid", "docno", "query"]].copy()
        min_score_lexical = min(df["score_0"])
        max_score_lexical = max(df["score_0"])
        min_score_dense = min(df["score"])
        max_score_dense = max(df["score"])

        lexical_normalized = (df["score_0"] - min_score_lexical) / (max_score_lexical - min_score_lexical)
        sparse_normalized = (df["score"] - min_score_dense) / (max_score_dense - min_score_dense)

        new_df["score"] = self.alpha * lexical_normalized + (1 - self.alpha) * sparse_normalized
        return new_df
