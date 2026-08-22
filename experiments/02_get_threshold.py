"""
Подбор threshold для DecisionCentricMemory.

Два независимых способа посмотреть на один и тот же вопрос -- предпочтителен
результат части Б, часть А просто даёт быструю оценку разумного диапазона
перед тем, как гонять более дорогой сквозной прогон.

Часть А -- offline, без памяти и без агента: генерируем много ситуаций,
считаем попарную TF-IDF cosine similarity, размечаем пары как "тот же
semantic_cluster" (должны были бы считаться похожими) / "разный cluster"
(не должны). Текст в генераторе зависит только от cluster (не от mismatch и
не от decision), так что разметка не зависит от mismatch -- можно взять любое
значение. Дальше перебираем threshold и смотрим, какой лучше всего разделяет
два распределения (по F1 и по balanced accuracy, раз классы сильно
несбалансированы -- пар с одинаковым cluster на порядок меньше).

Часть Б -- online, по-настоящему: гоняем DecisionCentricMemory через
SyntheticEnvironment с заглушкой вместо агента (предсказываем действие
ближайшей записи в памяти, если она есть) на сетке threshold x mismatch,
смотрим на итоговую online accuracy, финальный размер памяти и суммарное
число конфликтов за весь прогон (не только по финальному снимку self.memory
-- эвикшн выбивает конфликтные записи, так что смотреть надо на счётчик,
накопленный по ходу прогона, а не на то, что осталось в памяти к концу).
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.env.generator import EnvironmentConfig, SituationGenerator
from src.env.environment import SyntheticEnvironment
from src.memory.demem import DecisionCentricMemory


# ---------------------------------------------------------------------------
# Часть А: offline-разделимость по распределению similarity
# ---------------------------------------------------------------------------

def part_a_distributional(n: int = 1500, seed: int = 67) -> float:
    config = EnvironmentConfig()
    generator = SituationGenerator(config=config, seed=seed)

    texts, clusters = [], []
    for _ in range(n):
        situation = generator.generate()
        texts.append(situation.text)
        clusters.append(situation.semantic_cluster)

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    sim = cosine_similarity(matrix, matrix)
    clusters = np.array(clusters)

    same_cluster = clusters[:, None] == clusters[None, :]
    off_diag = ~np.eye(n, dtype=bool)

    same_pairs = sim[same_cluster & off_diag]
    diff_pairs = sim[~same_cluster & off_diag]

    print("Часть А -- распределение similarity")
    print(f"  same-cluster пар: {len(same_pairs)}, "
          f"mean={same_pairs.mean():.3f}, min={same_pairs.min():.3f}, max={same_pairs.max():.3f}")
    print(f"  diff-cluster пар: {len(diff_pairs)}, "
          f"mean={diff_pairs.mean():.3f}, min={diff_pairs.min():.3f}, max={diff_pairs.max():.3f}")

    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold, best_f1 = None, -1.0
    print(f"\n  {'threshold':>9} | {'precision':>9} | {'recall':>7} | {'F1':>6} | {'balanced_acc':>12}")
    for t in thresholds:
        tp = (same_pairs >= t).sum()
        fn = (same_pairs < t).sum()
        fp = (diff_pairs >= t).sum()
        tn = (diff_pairs < t).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        balanced_acc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))

        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            marker = "  <-- лучший F1 пока что"
        print(f"  {t:>9.2f} | {precision:>9.3f} | {recall:>7.3f} | {f1:>6.3f} | {balanced_acc:>12.3f}{marker}")

    print(f"\n  Часть А рекомендует threshold ~ {best_threshold:.2f} (по F1)")
    return best_threshold


# ---------------------------------------------------------------------------
# Часть Б: сквозной прогон DecisionCentricMemory
# ---------------------------------------------------------------------------

def run_online(threshold: float, mismatch: float, capacity: int = 12,
               n_steps: int = 800, seed: int = 0) -> dict:
    gen = SituationGenerator(EnvironmentConfig(mismatch=mismatch), seed=seed)
    env = SyntheticEnvironment(gen, max_steps=n_steps)
    mem = DecisionCentricMemory(capacity=capacity, threshold=threshold)

    situation = env.reset()
    correct = 0
    total_conflicts = 0  # накопленное за весь прогон, не по финальному memory

    for _ in range(n_steps):
        retrieved = mem.retrieve(situation, topk=1)
        predicted = retrieved[0].action if retrieved else 0

        # независимо от внутренностей update() смотрим, был ли это конфликт
        # по тем же правилам -- чтобы не терять счёт из-за эвикшена
        if mem.memory:
            score, index = mem.get_situation_similarity_w_nearest_entry(situation)
            nearest_action = mem.memory[index].action
        else:
            score, nearest_action = 0.0, None

        step_result = env.step(situation, predicted)
        correct += step_result.reward

        if mem.memory and score >= threshold and nearest_action != step_result.correct_action:
            total_conflicts += 1

        mem.update(situation, predicted, step_result)
        situation = env.sample()

    return dict(
        online_acc=correct / n_steps,
        final_memory_size=len(mem.memory),
        total_conflicts=total_conflicts,
    )


def part_b_online(candidate_thresholds, mismatches=(0.0, 0.5, 1.0), n_steps=800):
    print("\nЧасть Б -- сквозной прогон (online accuracy / |memory| / конфликты за весь прогон)")
    header = f"{'threshold':>9} | " + " | ".join(f"mismatch={m:<4}" for m in mismatches)
    print(header)
    for t in candidate_thresholds:
        row = [f"{t:>9.2f}"]
        for m in mismatches:
            res = run_online(threshold=t, mismatch=m, n_steps=n_steps)
            row.append(f"acc={res['online_acc']:.2f} mem={res['final_memory_size']:>2d} "
                       f"conf={res['total_conflicts']:>3d}")
        print(" | ".join(row))


if __name__ == "__main__":
    best_from_a = part_a_distributional()

    # берём окрестность рекомендации части А плюс исходный 0.9 для сравнения,
    # чтобы явно увидеть разницу с тем, что было изначально
    candidates = sorted(set([0.9, round(best_from_a - 0.1, 2),
                              round(best_from_a, 2), round(best_from_a + 0.1, 2)]))
    candidates = [c for c in candidates if 0.0 < c < 1.0]

    part_b_online(candidates)