from base import Memory,MemoryEntry
class RecentMemory(Memory):
    def retrieve(self, situation, topk = 5):
        return self.memory[-topk:]
    def update(self, situation, predicted_action, step_result):
        self.memory.append(MemoryEntry(text=situation.text, action=step_result.correct_action))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)