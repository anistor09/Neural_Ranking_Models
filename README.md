# The impact of the semantic matching within interpolation-based re-ranking

This README file supports my final bachelor thesis that can be found at: < input thesis link >. Before using the library,
it is strongly recomended to read the previously mentioned paper, alongside the Fast-Forward indexes paper (https://arxiv.org/abs/2311.01263). 

## Repository overview

This repository is structures as follows:

* /encoders directory stores the implementation of all HuggingFace models employed in my research

* /general_dense_indexers provides general scripts needed in the dense indexing of all the HuggingFace models,
making them easily reusable in each model's directory

* /bge /e5 /gte_base_en_v1_5 /nomic /snowflake /tct_colbert directories store:
  * the /dense_indexers directory (scripts for dense indexing), dense indexes (this is where the /dense_indexes will be stored if created within this codebase -> these directories
  are currently empty because of the file size constraint on GitHub but these available on a request basis)
  * the /experiments directory which stores Python scripts for conducting experiments (including alpha hyperparameter 
  (used within the interpolation between the dense and semantic scores) tuning)
  * /results stores the following files:
    * ranking_metrics_alpha.csv - Ranking results for the current model across all datasets for the tuned alpha hyperparameter (alpha value that yields best nDCG results)
    * alpha_search_dev.csv - Ranking results (nDCG@10 and RR@10) across all alpha values for the current model
    * latency_data.csv - Per query latency results for the complete semantic re-ranking across  FiQA, NFCorpus and SciFact datasets for the current model
    * query_encoding_latency_data.csv - Per query latency results for the query embedding within semantic re-ranking  across  FiQA, NFCorpus and SciFact datasets for the current model
    * vector_embedding_latency_data.csv - Per query latency results for the documet embedding retrieval within semantic re-ranking  across  FiQA, NFCorpus and SciFact datasets for the current model

* /experiment_utils directory includes the resources needed for running any experiment,
including all the steps from loading the sparse and dense indexes into memory to hyperparameter
tuning and storing results 

* /fast_forward_indexes_library_enhancements directory includes changes to the Fast-Forward indexes pipeline provided
here (https://github.com/mrjleo/fast-forward-indexes/ implementation of https://arxiv.org/abs/2311.01263) that were needed for compliance with all datasets needed, running
latency measurements per operation (query emnedding, documents embedding retrieval) and adding additional transformers to the Fast-Forward indexes pipeline 

* /results directory stores all experiments of our results

  * aggreagated_ranking_results.csv - ranking results (nDCG@10 and RR@10) for all models across all datasets
  * aggreagated_latency_results.csv - per query latency results for the second stage (semantic re-ranking) stage of the Fast-Forward indexes pipeline - only for
  FiQA, NFCorpus and SciFact datasets because of the limited resources on my local machine (16 G RAM) which prevented me to load larger indexes in memory
  
  * aggreagated_ranking_latency_results.csv - merged ranking and latency results for each <model, dataset> pair for which both are available
  * aggreagated_vector_embedding_latency_results.csv - per query latency for document vector embeddings retrieval 
  across FiQA, NFCorpus and SciFact datasets
  * aggreagated_query_encoding_latency_results.csv - per query latency for document vector embeddings retrieval across 
  FiQA, NFCorpus and SciFact datasets
  * aggreagated_latency_parts_results.csv - merged per query latency of the complete semantic re-ranking stage, per 
  query latency of query embedding and per query document embeddings retrieval latency across FiQA, NFCorpus and SciFact datasets
 
  * /plots stores all the figures used within my paper, created using the plot_latency_per_query_breakdown.ipynb 
    plot_latency_x_ndcg_per_dataset.ipynb Jupyter notebooks. Color blind friendy colors are used.
  
  * /signifiacne_reports stores all the data yielded from the significance pairwise tests. p < 0.05 is used for
    the statistical signifiance tests, which are conducted for the ranking nDCG@10 and RR@10 results, using the TREC files of each model 
  across all datasets.
 
  * /trec_runs TREC Files (experiments output) for the DBPedia_Entity, FiQA, NFCorpus, TREC-DL-passage-2019 and SciFact datasets.
  Data is also available for Fever and Quara datasets, available on request basis (they excedded the default GitHub file size).
  
  * aggregate_results_script.py provides solution for aggregating the latency and ranking results from all the model's directories
    alongside helper methods for further aggreagating the results and creating Latex tables
  
  * extract_signifiance_relations.py extract the signifiance relations between the models

* /sparse_indexers stores:
  * experiments_sparse_indexes.ipynb - Notebook in which the experiments for the first stage (sparse) retrieval  are conducted -> only ranking results (nDCG@10 and RR@10) are stored
  * sparse_index_one_dataset.py - script that creates the sparse index for any dataset (not dependent on the model used for retrieval (e.g., BM25))

* /statistical_significance stores:
  * find_statistical_signifiance_script.py script for running the statistical significance tests (p < 0.05) given the TREC runfiles
  * /all_models_and_datasets/save_trec_files.py script that runs all experiments and stores the TREC files only

* /example_supercomputer_scripts directory stores multiple bash scripts used within my research for running resource-intensive tasks
on the DelftBlue Supercomputer (TUDelft).

    The tasks include indexing collections of data in lower dimensional vector embeddings (GPU with CUDA cores needed for
faster computation),  running CPU-only jobs which require more RAM memory that can be found on a normal
machine, creating TREC files (output of our pipeline that is a proof that supports the experimental results) and
running hyperparameter tuning. For some datasets such as MS MARCO (TREC DL passage 2019 evaluation set), the model requires high memory
resources as the indexes are loaded in memory for faster evaluation.

## Running guidelines

The repository is configured for command line operation to handle multiple jobs that demand substantial computational resources.
I ran it on the DelftBlue SuperComputer. The scripts for the supercomputer are flexible and can be modified to execute any Python file within the repository.

A Python file can be executed from the command line using the following example:

```bash
python -m e5.dense_indexers.index_multiple_base
```

## General observations

* Further enhancements will be made to this library in the future to improve code quality.

* The latency measurements employs the timeit Python module that relies on
time.perf counter() function, which measures wallclock time and can be impacted by many different factors, such as
machine load. Therefore, it is strongly recommended to ensure the latency
tests are the main active tasks on the machine despite  the presence of 
essential system processes.

* Sparse and dense indexes, as well as the complete set of TREC runfiles are available on a request basis. The dense 
indexes size should be considered as the biggest (MS MARCO) ones reach as much as 25 GB.

 
