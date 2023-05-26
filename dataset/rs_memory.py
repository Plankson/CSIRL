
from collections import deque
import numpy as np
import random
import torch


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class Memory(object):
    def __init__(self, memory_size: int, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


    def get_samples(self, batch_size, device):
        batch = self.sample(batch_size, False)

        # batch_state, batch_next_state, batch_action, batch_re_obs, batch_reward1, batch_reward2, batch_done = zip(*batch)
        batch_state, batch_next_state, batch_action, batch_re_obs, batch_reward1, batch_done = zip(*batch)

        # Scale obs for atari. TODO: Use flags
        if isinstance(batch_state[0], LazyFrames):
            # Use lazyframes for improved memory storage (same as original DQN)
            batch_state = np.array(batch_state) / 255.0
        if isinstance(batch_next_state[0], LazyFrames):
            batch_next_state = np.array(batch_next_state) / 255.0
        batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)

        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)
        batch_re_obs = torch.as_tensor(batch_re_obs, dtype=torch.float, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward1 = torch.as_tensor(batch_reward1, dtype=torch.float, device=device).unsqueeze(1)
        # batch_reward2 = torch.as_tensor(batch_reward2, dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)

        # return batch_state, batch_next_state, batch_action,batch_re_obs, batch_reward1, batch_reward2, batch_done
        return batch_state, batch_next_state, batch_action,batch_re_obs, batch_reward1, batch_done

