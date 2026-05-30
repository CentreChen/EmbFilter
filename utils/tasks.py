"""
Copied from Script for evaluating Jina Embedding Models on the MTEB benchmark.
https://github.com/embeddings-benchmark/mteb/blob/main/scripts/run_mteb_english.py
"""

TASK_LIST_STS = [
    "STS17",
    "SICK-R",
    "STSBenchmark", # used for ablation
    "BIOSSES",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS22",
]

TASK_LIST_CLASSIFICATION = [
    "Banking77Classification",
    "EmotionClassification",
    "MassiveIntentClassification",  # used for ablation
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "ImdbClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "BiorxivClusteringS2S",
    "MedrxivClusteringS2S",
    "TwentyNewsgroupsClustering", # used for ablation
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "MedrxivClusteringP2P",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions", # used for ablation
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "SciDocsRR",
    "StackOverflowDupQuestions", # used for ablation
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
]

TASK_LIST_RETRIEVAL = [
    "SciFact",
    "ArguAna",
    "NFCorpus", # used for ablation
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_SUMMARIZATION = [
    "SummEval",
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS  
    + TASK_LIST_SUMMARIZATION
)

ALL_TASKS = [
    ("STS", TASK_LIST_STS),
    ("Classification", TASK_LIST_CLASSIFICATION), 
    ("Clustering", TASK_LIST_CLUSTERING), 
    ("PairClassification", TASK_LIST_PAIR_CLASSIFICATION), 
    ("Reranking", TASK_LIST_RERANKING), 
    ("Retrieval", TASK_LIST_RETRIEVAL),
    ("Summarization", TASK_LIST_SUMMARIZATION),
]



if __name__ == '__main__':
    import mteb
    from mteb.tasks import *

    MTEB_MAIN_EN = mteb.get_benchmark("MTEB(eng, classic)")

    tasks = MTEB_MAIN_EN.tasks

    TASK_LIST_ = [task.metadata.name for task in tasks]

    assert len(TASK_LIST_) == len(TASK_LIST) and not (set(TASK_LIST_) - set(TASK_LIST))
    breakpoint()
