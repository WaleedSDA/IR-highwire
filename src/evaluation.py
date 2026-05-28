from __future__ import annotations

import logging
import time

import pandas as pd

_log = logging.getLogger(__name__)


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
        from .pt_initializer import init_pyterrier
        init_pyterrier()

        import pyterrier as pt


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
        feedbacks: list[tuple] | None = None,
        reranker=None,
    ) -> pd.DataFrame:
        """
        Build all enabled combinations and run pt.Experiment.

        feedbacks: list of (RelevanceFeedback, label_suffix) pairs, e.g.
                   [(bo1_obj, "Bo1"), (kl_obj, "KL")]

        For each first-stage ranker:
          <Ranker>
          <Ranker>+<FB>        for each feedback model
          <Ranker>+Neural      (if reranker given)
          <Ranker>+<FB>+Neural for each feedback model (if reranker given)
        """
        names = names or [f"ranker_{i}" for i in range(len(rankers))]
        neural_t = reranker.as_transformer() if reranker is not None else None
        fb_list = feedbacks or []

        pipelines: list = []
        pipeline_names: list[str] = []

        for ranker, label in zip(rankers, names):
            retriever = ranker.get_pipeline() if hasattr(ranker, "get_pipeline") else ranker

            pipelines.append(retriever)
            pipeline_names.append(label)

            for fb, fb_label in fb_list:
                pipelines.append(fb.get_pipeline(retriever))
                pipeline_names.append(f"{label}+{fb_label}")

            if neural_t is not None:
                pipelines.append(retriever >> neural_t)
                pipeline_names.append(f"{label}+Neural")

            for fb, fb_label in fb_list:
                if neural_t is not None:
                    pipelines.append(fb.get_pipeline(retriever) >> neural_t)
                    pipeline_names.append(f"{label}+{fb_label}+Neural")

        # Run each pipeline individually so we can log progress, then pass the
        # collected result DataFrames to pt.Experiment for metric computation.
        total = len(pipelines)
        n_topics = len(self._topics)
        _log.info("Evaluation starting — %d pipelines × %d topics", total, n_topics)

        result_frames: list[pd.DataFrame] = []
        for i, (pipe, name) in enumerate(zip(pipelines, pipeline_names), 1):
            _log.info("[%d/%d] Running: %s", i, total, name)
            t0 = time.perf_counter()
            result_frames.append(pipe.transform(self._topics.copy()))
            elapsed = time.perf_counter() - t0
            _log.info("[%d/%d] Done: %s  (%.1fs, %d remaining)", i, total, name, elapsed, total - i)

        _log.info("All pipelines done — computing metrics")
        df = self._pt.Experiment(
            result_frames,
            self._topics,
            self._qrels,
            eval_metrics=self._METRICS,
            names=pipeline_names,
        )
        _log.info("Evaluation complete")
        return df

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
