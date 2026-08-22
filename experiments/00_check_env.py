from collections import defaultdict
from src.env.generator import (
    EnvironmentConfig,
    SituationGenerator,
)


def run(mismatch: float, n: int = 10_000) -> None:
    config = EnvironmentConfig(
        mismatch=mismatch,
    )

    generator = SituationGenerator(
        config=config,
        seed=67,
    )

    counts = defaultdict(lambda: defaultdict(int))

    for _ in range(n):
        situation = generator.generate()
        # decision_id больше не хранится в Situation (это была утечка правильного
        # ответа в наблюдение агента) -- читаем его отдельно через generator,
        # как это делает SyntheticEnvironment.step()
        decision_id = generator.get_correct_action(situation)

        counts[
            situation.semantic_cluster
        ][decision_id] += 1

    print(f"\nMismatch = {mismatch}")

    for cluster, decisions in counts.items():
        total = sum(decisions.values())

        distribution = {
            decision: round(count / total, 3)
            for decision, count in decisions.items()
        }

        print(
            f"cluster={cluster}: "
            f"{distribution}"
        )


if __name__ == "__main__":
    for mismatch in (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ):
        run(mismatch)