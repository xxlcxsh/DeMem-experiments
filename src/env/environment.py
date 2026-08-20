from __future__ import annotations

from dataclasses import dataclass

from .generator import Situation, SituationGenerator


@dataclass(frozen=True)
class StepResult:
    action: int
    correct_action: int
    reward: float
    done: bool


class SyntheticEnvironment:
    def __init__(
        self,
        generator: SituationGenerator,
        max_steps: int = 1000,
    ) -> None:
        self.generator = generator
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self) -> Situation:
        self.step_count = 0
        return self.generator.generate()

    def step(
        self,
        situation: Situation,
        action: int,
    ) -> StepResult:
        self.step_count += 1

        reward = float(
            action == situation.correct_action
        )

        done = self.step_count >= self.max_steps

        return StepResult(
            situation=situation,
            action=action,
            reward=reward,
            done=done,
        )

    def sample(self) -> Situation:
        return self.generator.generate()