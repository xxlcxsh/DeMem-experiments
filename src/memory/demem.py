from src.memory.base import Memory, MemoryEntry
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass


@dataclass
class DememEntry(MemoryEntry):
    was_right: int = 0
    was_conflict: int = 0


class DecisionCentricMemory(Memory):
    def __init__(self, capacity, threshold: float = 0.5):
        super().__init__(capacity)
        self.vectorizer = TfidfVectorizer()
        self.threshold = threshold

    def get_situation_similarity_w_nearest_entry(self, situation):
        texts = [entry.text for entry in self.memory]
        matrix = self.vectorizer.fit_transform(texts)
        vec_query = self.vectorizer.transform([situation.text])
        scores = cosine_similarity(vec_query, matrix)[0]
        index = np.argsort(scores)[::-1][0]
        score = scores[index]
        return score, index

    def retrieve(self, situation, topk=1):
        if not self.memory:
            return []
        texts = [entry.text for entry in self.memory]
        matrix = self.vectorizer.fit_transform(texts)
        vec_query = self.vectorizer.transform([situation.text])
        scores = cosine_similarity(vec_query, matrix)[0]
        nearest_indices = np.argsort(scores)[::-1][:topk]
        return [self.memory[i] for i in nearest_indices]

    def update(self, situation, predicted_action, step_result):
        if self.memory:
            score, index = self.get_situation_similarity_w_nearest_entry(situation=situation)
            nearest_entry = self.memory[index]
            if score < self.threshold:
                self.memory.append(DememEntry(text=situation.text, action=step_result.correct_action))
            else:
                if nearest_entry.action == step_result.correct_action:
                    nearest_entry.was_right += 1
                else:
                    self.memory.append(DememEntry(text=situation.text, action=step_result.correct_action))
                    nearest_entry.was_conflict += 1
            if len(self.memory) > self.capacity:
                max_conflicts = -1
                for i in range(len(self.memory)):
                    if self.memory[i].was_conflict > max_conflicts:
                        max_conflicts = self.memory[i].was_conflict
                        candidates = []
                        candidates.append((i, self.memory[i].was_right))
                    elif self.memory[i].was_conflict == max_conflicts:
                        candidates.append((i, self.memory[i].was_right))
                candidates.sort(key=lambda x: x[1])
                get_out_index = candidates[0][0]
                self.memory.pop(get_out_index)
        else:
            self.memory.append(DememEntry(text=situation.text, action=step_result.correct_action))