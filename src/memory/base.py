from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.env.generator import Situation
from src.env.environment import StepResult
@dataclass
class MemoryEntry:
    text: str
    action: int
class Memory(ABC):
    def __init__(self,capacity: int = 100):
        self.capacity = capacity
        self.memory: list[MemoryEntry] = []
    @abstractmethod
    def retrieve(self, situation: Situation, topk: int = 5) -> list[MemoryEntry]:
        pass
    @abstractmethod
    def update(self, situation: Situation, predicted_action: int, step_result: StepResult):
        pass