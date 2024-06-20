from fast_forward import Ranking


def preprocess_data(first_stage_results):
    ranking = Ranking(
        first_stage_results.rename(columns={"qid": "q_id", "docno": "id"}),
        copy=False,
        is_sorted=True,
    )
    # get all unique queries and query IDs and map to unique numbers (0 to m)
    query_df = (
        ranking._df[["q_id", "query"]].drop_duplicates().reset_index(drop=True)
    )
    query_df["q_no"] = query_df.index

    # attach query numbers to data frame
    df = ranking._df.merge(query_df, on="q_id", suffixes=[None, "_"])

    # get all unique queries and query IDs and map to unique numbers (0 to m)
    query_df = (
        ranking._df[["q_id", "query"]].drop_duplicates().reset_index(drop=True)
    )
    query_df["q_no"] = query_df.index

    # attach query numbers to data frame
    df = ranking._df.merge(query_df, on="q_id", suffixes=[None, "_"])

    queries = list(query_df["query"])

    return df, queries


def get_vector_ids(df):
    # map doc/passage IDs to unique numbers (0 to n)
    id_df = df[["id"]].drop_duplicates().reset_index(drop=True)
    id_df["id_no"] = id_df.index

    vector_ids = id_df["id"].to_list()

    return vector_ids
