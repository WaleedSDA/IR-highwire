from __future__ import annotations

import pandas as pd


class EvaluationEngine:
    """
    Wraps pt.Experiment to evaluate rankers against Highwire TREC qrels.
    Attributes: qrels: HighwireQrels.
    Metrics: MAP, R-Prec, MRR, P@5, P@10, NDCG@10.
    """

    _DATASETS = [
        "irds:highwire/trec-genomics-2006",
        "irds:highwire/trec-genomics-2007",
    ]
    _METRICS = ["map", "ndcg_cut_10", "P_5", "P_10", "recip_rank", "Rprec"]

    def __init__(self, dataset_names: list[str] | None = None):
        import pyterrier as pt
        if not pt.started():
            pt.init()

        self._pt = pt
        names = dataset_names or self._DATASETS
        datasets = [pt.get_dataset(n) for n in names]

        # Merge topics and qrels from both years
        topics_frames = [d.get_topics() for d in datasets]
        qrels_frames = [d.get_qrels() for d in datasets]

        # Offset qids to avoid collisions between 2006 and 2007
        for i, frame in enumerate(topics_frames[1:], start=1):
            frame["qid"] = frame["qid"].astype(str).apply(lambda q: f"{i}_{q}")
        for i, frame in enumerate(qrels_frames[1:], start=1):
            frame["qid"] = frame["qid"].astype(str).apply(lambda q: f"{i}_{q}")

        self._topics = pd.concat(topics_frames, ignore_index=True)
        self._qrels = pd.concat(qrels_frames, ignore_index=True)

    @property
    def qrels(self) -> pd.DataFrame:
        return self._qrels

    @property
    def topics(self) -> pd.DataFrame:
        return self._topics

    def run_experiment(
        self,
        rankers: list,
        names: list[str] | None = None,
    ) -> pd.DataFrame:
        pipelines = [
            r.get_pipeline() if hasattr(r, "get_pipeline") else r
            for r in rankers
        ]
        names = names or [f"ranker_{i}" for i in range(len(rankers))]
        return self._pt.Experiment(
            pipelines,
            self._topics,
            self._qrels,
            eval_metrics=self._METRICS,
            names=names,
        )

    def compute_map(self, results: pd.DataFrame) -> float:
        ev = self._pt.Utils.evaluate(results, self._qrels, metrics=["map"])
        return float(ev["map"])

    def compute_ndcg(self, results: pd.DataFrame, k: int = 10) -> float:
        metric = f"ndcg_cut_{k}"
        ev = self._pt.Utils.evaluate(results, self._qrels, metrics=[metric])
        return float(ev[metric])

    def compute_p_at_k(self, results: pd.DataFrame, k: int = 10) -> float:
        metric = f"P_{k}"
        ev = self._pt.Utils.evaluate(results, self._qrels, metrics=[metric])
        return float(ev[metric])
