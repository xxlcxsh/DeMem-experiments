from base import Memory, MemoryEntry
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
@dataclass
class DememEntry:
    entry: MemoryEntry
    similarity: float
class DecisionCentricMemory(Memory):
    def __init__(self,capacity):
        super().__init__(capacity)
        self.vectorizer = TfidfVectorizer()
    def retrieve(self, situation, topk=1):
        if not self.memory:
            return []
        texts = [entry.text for entry in self.memory]
        matrix = self.vectorizer.fit_transform(texts)
        vec_query = self.vectorizer.transform([situation.text])
        scores = cosine_similarity(vec_query,matrix)[0]
        nearest_indices = np.argsort(scores)[::-1][:topk]
        return [self.memory[i] for i in nearest_indices]

    def update(self, situation, predicted_action, step_result):
        threshold = 0.9
        if predicted_action != step_result.correct_action and similarity>=threshold:
            pass
        elif predicted_action != step_result.correct_action and similarity<threshold:
            self.memory.append(MemoryEntry(text=situation.text, action=step_result.correct_action))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)