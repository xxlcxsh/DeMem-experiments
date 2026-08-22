from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Situation:
    id: int
    semantic_cluster: int
    text: str

@dataclass(frozen=True)
class EnvironmentConfig:
    num_semantic_clusters: int = 8
    num_decisions: int = 8
    feature_dim: int = 32
    mismatch: float = 0.0

    noise_std: float = 0.1


class SituationGenerator:
    def __init__(
        self,
        config: EnvironmentConfig,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.correct_action_dict = {}
        self.semantic_centers = self.rng.normal(
            size=(
                config.num_semantic_clusters,
                config.feature_dim,
            )
        )
        self.preferred_decisions = np.arange(
            config.num_semantic_clusters
        ) % config.num_decisions

        self._next_id = 0

    def generate(
        self,
        semantic_cluster: int | None = None,
    ) -> Situation:
        if semantic_cluster is None:
            semantic_cluster = int(
                self.rng.integers(
                    0,
                    self.config.num_semantic_clusters,
                )
            )

        decision_id = self._sample_decision(
            semantic_cluster
        )
        self.correct_action_dict[self._next_id] = decision_id

        text = self._generate_text(
            semantic_cluster
        )

        situation = Situation(
            id=self._next_id,
            semantic_cluster=semantic_cluster,
            text=text,
        )

        self._next_id += 1

        return situation
    
    def get_correct_action(self,situation):
        return self.correct_action_dict.pop([situation.id])
    
    def _sample_decision(
        self,
        semantic_cluster: int,
    ) -> int:
        preferred = self.preferred_decisions[
            semantic_cluster
        ]
        p_preferred = 1.0 - self.config.mismatch

        if self.rng.random() < p_preferred:
            return int(preferred)

        alternatives = [
            d
            for d in range(self.config.num_decisions)
            if d != preferred
        ]

        return int(self.rng.choice(alternatives))
    def _generate_text(self, semantic_cluster: int) -> str:
        ANCHORS = ['Corvus', 'Draco', 'Norma', 'Phoenix', 'Vega', 'Atlas', 'Orion', 'Altair']
        FILLERS = ['query', 'тикет', 'запрос', 'отчёт', 'вопрос', 'конфиг']
        TEMPLATES = ["На проект {anchor} пришёл {filler} от пользователя",
                      "В {anchor} появился новый {filler} от пользователя",
                      "{anchor} пришёл ещё один {filler}"]
        anchor = ANCHORS[semantic_cluster]
        return self.rng.choice(TEMPLATES).format(
        anchor=anchor, filler=self.rng.choice(FILLERS)
        )