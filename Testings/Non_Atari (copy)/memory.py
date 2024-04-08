from import_packages import *


class ReplayBuffer:
    def __init__(self, capacity, num_envs):
        self.capacity = capacity
        self.num_envs = num_envs
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return jax.random.choice(jax.random.PRNGKey(np.random.randint(low=0, high=100)), len(self.buffer), shape=(batch_size,), replace=False)

    def get_batch(self, indices):
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices])
        states = jnp.array(states).reshape((self.num_envs*len(indices), -1))
        actions = jnp.array(actions).reshape((self.num_envs*len(indices), -1))
        rewards = jnp.array(rewards).reshape((self.num_envs*len(indices), -1))
        next_states = jnp.array(next_states).reshape(
            (self.num_envs*len(indices), -1))
        dones = jnp.array(dones).reshape((self.num_envs*len(indices), -1))
        return states, actions, rewards, next_states, dones


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PERMemory:
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return (error+self.e)**self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []

        segment = self.tree.total()/n
        for i in range(n):
            a = segment*i
            b = segment*(i+1)
            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch

    def get_data(self, batch, n):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(n):
            states.append(batch[i][1][0])
            actions.append(batch[i][1][1])
            rewards.append(batch[i][1][2])
            next_states.append(batch[i][1][3])
            dones.append(batch[i][1][4])
        states = jnp.asarray(states)
        actions = jnp.asarray(actions)
        rewards = jnp.asarray(rewards)
        next_states = jnp.asarray(next_states)
        dones = jnp.asarray(dones)
        return states, actions, rewards, next_states, dones

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
