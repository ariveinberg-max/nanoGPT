"""The learning machinery behind a unit.

A unit is not scripted. It carries a small policy network and improves it from
its own lived experience: every tick it observes its internal state and the
world, samples an intent, and is rewarded by how well that intent served its
drives. Over a few simulated weeks a unit that began flailing settles into a
daily rhythm nobody wrote down.

REINFORCE with a moving-average baseline and an entropy bonus. numpy only --
the units have to keep running on whatever hardware the lab has.
"""

import numpy as np

INTENTS = (
    "sleep",
    "eat",
    "work",
    "socialize",
    "wander",
    "rest",
    "errand",
    "seek",
)
N_INTENTS = len(INTENTS)


class Policy:
    """A two-layer tanh MLP mapping an observation to a distribution over intents."""

    def __init__(self, n_obs, n_hidden=24, n_act=N_INTENTS, rng=None):
        self.rng = rng or np.random.default_rng()
        self.w1 = self.rng.normal(0, 1.0 / np.sqrt(n_obs), (n_obs, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.w2 = self.rng.normal(0, 1.0 / np.sqrt(n_hidden), (n_hidden, n_act))
        self.b2 = np.zeros(n_act)

    def forward(self, obs):
        h = np.tanh(obs @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        logits -= logits.max()
        exp = np.exp(logits)
        return h, exp / exp.sum()

    def act(self, obs, temperature=1.0):
        h, probs = self.forward(obs)
        if temperature != 1.0:
            p = probs ** (1.0 / temperature)
            probs = p / p.sum()
        a = int(self.rng.choice(len(probs), p=probs))
        return a, (obs, h, probs, a)


class Learner:
    """Accumulates experience and folds it back into the policy."""

    def __init__(self, policy, lr=0.05, entropy=0.035, entropy_floor=0.004,
                 anneal=120, horizon=48, gamma=0.97):
        self.policy = policy
        self.lr = lr
        self.entropy0 = entropy
        self.entropy_floor = entropy_floor
        self.anneal = anneal
        self.updates = 0
        self.entropy = entropy
        self.horizon = horizon
        self.gamma = gamma
        self.baseline = 0.0
        self.trace = []
        self.reward_history = []

    def record(self, step, reward):
        self.trace.append((step, reward))
        self.reward_history.append(reward)
        if len(self.reward_history) > 4096:
            del self.reward_history[:2048]
        if len(self.trace) >= self.horizon:
            self.learn()

    def learn(self):
        if not self.trace:
            return 0.0
        rewards = np.array([r for _, r in self.trace], dtype=float)
        # discounted return-to-go, so an intent gets credit for what followed it
        returns = np.zeros_like(rewards)
        acc = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            acc = rewards[i] + self.gamma * acc
            returns[i] = acc
        self.baseline = 0.95 * self.baseline + 0.05 * returns.mean()
        advantage = returns - self.baseline
        scale = advantage.std()
        if scale > 1e-6:
            advantage = advantage / scale

        # curiosity is expensive once a unit knows its way around; anneal it
        t = min(1.0, self.updates / self.anneal)
        self.entropy = self.entropy0 + t * (self.entropy_floor - self.entropy0)
        self.updates += 1

        p = self.policy
        gw1 = np.zeros_like(p.w1); gb1 = np.zeros_like(p.b1)
        gw2 = np.zeros_like(p.w2); gb2 = np.zeros_like(p.b2)
        for (obs, h, probs, a), adv in zip((s for s, _ in self.trace), advantage):
            dlogits = -probs * adv
            dlogits[a] += adv
            # entropy bonus keeps a unit curious instead of locking in early
            logp = np.log(probs + 1e-9)
            ent_grad = -probs * (logp + (probs * logp).sum())
            dlogits += self.entropy * ent_grad

            gw2 += np.outer(h, dlogits)
            gb2 += dlogits
            dh = (p.w2 @ dlogits) * (1 - h * h)
            gw1 += np.outer(obs, dh)
            gb1 += dh

        n = len(self.trace)
        p.w1 += self.lr * gw1 / n
        p.b1 += self.lr * gb1 / n
        p.w2 += self.lr * gw2 / n
        p.b2 += self.lr * gb2 / n
        self.trace.clear()
        return float(rewards.mean())

    def recent_reward(self, window=512):
        if not self.reward_history:
            return 0.0
        return float(np.mean(self.reward_history[-window:]))
