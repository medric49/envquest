# class DQNAgentMemory(AgentMemory):
#     def __init__(self, capacity: int, discount: float, n_step: int):
#         self.discount = discount
#         self.capacity = capacity
#         self.n_step = n_step
#
#         self.observations = None
#         self.actions = None
#         self.rewards = None
#         self.next_observations = None
#         self.next_step_terminal = None
#
#         self.first_step_indices = None
#         self.last_step_indices = None
#
#         self.initialize()
#
#     def initialize(self):
#         self.observations = deque(maxlen=self.capacity)
#         self.actions = deque(maxlen=self.capacity)
#         self.rewards = deque(maxlen=self.capacity)
#         self.next_observations = deque(maxlen=self.capacity)
#         self.next_step_terminal = deque(maxlen=self.capacity)
#
#         self.first_step_indices = deque(maxlen=self.capacity)
#         self.last_step_indices = deque(maxlen=self.capacity)
#
#     def push(self, timestep: TimeStep, next_timestep: TimeStep):
#         observation = timestep.observation
#         action = next_timestep.action
#         reward = next_timestep.reward
#         next_observation = next_timestep.observation
#         next_step_terminal = next_timestep.last() and not next_timestep.truncated
#
#         self.observations.append(observation)
#         self.actions.append(action)
#         self.rewards.append(reward)
#         self.next_observations.append(next_observation)
#         self.next_step_terminal.append(next_step_terminal)
#
#         self.first_step_indices.append(1 if timestep.first() else 0)
#         self.last_step_indices.append(1 if next_timestep.last() else 0)
#
#     def __len__(self):
#         return len(self.observations)
#
#     def sample(
#         self, size: int = None, recent=False, exclude_first_steps=False, exclude_last_steps=False, **kwargs
#     ) -> tuple[np.ndarray, ...]:
#         if size is None:
#             raise ValueError("size is required")
#
#         indices = np.arange(len(self), dtype=np.int32)
#         if exclude_first_steps:
#             indices = indices[~np.asarray(self.first_step_indices, dtype=np.bool)]
#         if exclude_last_steps:
#             indices = indices[~np.asarray(self.last_step_indices, dtype=np.bool)]
#         if not recent:
#             indices = np.random.choice(indices, size=indices.shape[0], replace=False)
#         indices = indices[-size:]
#
#         observations = np.stack(self.observations)[indices]
#         actions = np.stack(self.actions)[indices]
#         next_observations = np.stack(self.next_observations)[indices]
#         next_step_terminal = np.array(self.next_step_terminal, dtype=np.uint8)[indices]
#         rewards = np.stack(self.rewards)[indices]
#
#         return (
#             observations,
#             actions,
#             rewards,
#             next_observations,
#             next_step_terminal,
#         )
