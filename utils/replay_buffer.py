import random
import torch
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition', 's, a, r, ns, d')
Batch = namedtuple('Batch', 's, a, r, ns, d')

MetaTransition = namedtuple('MetaTransition', 's, a, r, ns, d, z')
MetaBatch = namedtuple('MetaBatch', 's, a, r, ns, d, z')

class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.appendleft(transition)

    def ready_for(self, batch_size: int) -> bool:
        if len(self.memory) >= batch_size:
            return True
        return False
    
    def sample(self, batch_size: int) -> Batch:
        batch = random.sample(self.memory, batch_size)
        batch = Batch(*zip(*batch))
        s  = torch.tensor(np.array(batch.s), dtype  = torch.float).view(batch_size, -1)
        a  = torch.tensor(np.array(batch.a), dtype  = torch.float).view(batch_size, -1)  # continuous, multi-dim action
        r  = torch.tensor(np.array(batch.r), dtype  = torch.float).view(batch_size,  1)
        ns = torch.tensor(np.array(batch.ns), dtype = torch.float).view(batch_size, -1)
        d  = torch.tensor(np.array(batch.d), dtype  = torch.float).view(batch_size,  1)
        return Batch(s, a, r, ns, d)

    def __len__(self):
        return len(self.memory)

class MetaReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition: MetaTransition) -> None:
        self.memory.appendleft(transition)

    def ready_for(self, batch_size: int) -> bool:
        if len(self.memory) >= batch_size:
            return True
        return False

    def sample(self, batch_size: int) -> MetaBatch:
        batch = random.sample(self.memory, batch_size)
        batch = MetaBatch(*zip(*batch))
        s  = torch.tensor(np.array(batch.s), dtype  = torch.float).view(batch_size, -1)
        a  = torch.tensor(np.array(batch.a), dtype  = torch.float).view(batch_size, -1)  # continuous, multi-dim action
        r  = torch.tensor(np.array(batch.r), dtype  = torch.float).view(batch_size,  1)
        ns = torch.tensor(np.array(batch.ns), dtype = torch.float).view(batch_size, -1)
        d  = torch.tensor(np.array(batch.d), dtype  = torch.float).view(batch_size,  1)
        z  = torch.tensor(np.array(batch.z), dtype  = torch.float).view(batch_size, -1)
        return MetaBatch(s, a, r, ns, d, z)
    
    def __len__(self):
        return len(self.memory)
