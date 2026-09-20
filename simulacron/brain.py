"""The learning machinery behind a unit.

A unit is not scripted. It carries a small policy network and improves it from
its own lived experience: every tick it observes its internal state and the
world, samples an intent, and is rewarded by how well that intent served its
drives. Over a few simulated weeks a unit that began flailing settles into a
daily rhythm nobody wrote down.

Actor-critic: a policy head over intents and a value head that predicts how
well the unit is about to do from where it stands. The value head is what makes
this work at all. A unit's reward is dominated by slow-moving state -- how
hungry it is, what is in its pocket -- and against a single moving-average
baseline that drift drowns the part that depends on the choice just made. With
a baseline that reads the situation, what is left is the news: was this hour
worth it, given where I was standing? numpy only -- the units have to keep
running on whatever hardware the lab has.
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
    "rob",
)
N_INTENTS = len(INTENTS)


class Policy:
    """A tanh MLP with two heads: intents to choose, and how good this looks."""

    def __init__(self, n_obs, n_hidden=24, n_act=N_INTENTS, rng=None):
        self.rng = rng or np.random.default_rng()
        self.w1 = self.rng.normal(0, 1.0 / np.sqrt(n_obs), (n_obs, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.w2 = self.rng.normal(0, 1.0 / np.sqrt(n_hidden), (n_hidden, n_act))
        self.b2 = np.zeros(n_act)
        self.w3 = np.zeros(n_hidden)      # value head
        self.b3 = 0.0

    def forward(self, obs):
        h = np.tanh(obs @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        logits -= logits.max()
        exp = np.exp(logits)
        return h, exp / exp.sum(), float(h @ self.w3 + self.b3)

    def value(self, obs):
        return self.forward(obs)[2]

    def act(self, obs, temperature=1.0):
        h, probs, v = self.forward(obs)
        if temperature != 1.0:
            p = probs ** (1.0 / temperature)
            probs = p / p.sum()
        a = int(self.rng.choice(len(probs), p=probs))
        return a, (obs, h, probs, a, v)


class Learner:
    """Accumulates experience and folds it back into the policy."""

    def __init__(self, policy, lr=0.05, entropy=0.035, entropy_floor=0.004,
                 anneal=120, horizon=48, gamma=0.97, value_lr=0.30, share=1):
        self.policy = policy
        # `share` is how many learners sit on these weights. Their steps add
        # up, so each one has to be that much smaller -- twenty-four units
        # taking full-size steps on one substrate diverged the value head
        # into NaN within a few hundred days.
        self.lr = lr / share
        self.value_lr = value_lr / share
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
        values = np.array([rec[4] for rec, _ in self.trace], dtype=float)
        returns = np.zeros_like(rewards)
        # bootstrap the tail off the last state's value rather than pretending
        # the world ends in two days, which biased every late step downward
        acc = values[-1]
        for i in range(len(rewards) - 1, -1, -1):
            acc = rewards[i] + self.gamma * acc
            returns[i] = acc

        advantage = returns - values
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
        gw3 = np.zeros_like(p.w3); gb3 = 0.0
        for (obs, h, probs, a, v), adv, ret in zip(
                (rec for rec, _ in self.trace), advantage, returns):
            dlogits = -probs * adv
            dlogits[a] += adv
            # entropy bonus keeps a unit curious instead of locking in early
            logp = np.log(probs + 1e-9)
            ent_grad = -probs * (logp + (probs * logp).sum())
            dlogits += self.entropy * ent_grad

            gw2 += np.outer(h, dlogits)
            gb2 += dlogits
            dh = (p.w2 @ dlogits) * (1 - h * h)

            # The value head regresses on the realised return, with its own
            # much larger step. Folded into the policy's learning rate it was
            # ten times too slow to track returns of this magnitude: the
            # baseline sat near zero, so the advantage the policy saw was
            # dominated by how the unit's day happened to be going rather than
            # by the choice it had just made, and units never learned to look
            # at the clock at all.
            dv = float(np.clip(ret - v, -20.0, 20.0))
            gw3 += dv * h
            gb3 += dv
            dh += 0.1 * dv * p.w3 * (1 - h * h)

            gw1 += np.outer(obs, dh)
            gb1 += dh

        n = len(self.trace)
        p.w1 += self.lr * gw1 / n
        p.b1 += self.lr * gb1 / n
        p.w2 += self.lr * gw2 / n
        p.b2 += self.lr * gb2 / n
        p.w3 += self.value_lr * gw3 / n
        p.b3 += self.value_lr * gb3 / n
        self.baseline = float(returns.mean())
        self.trace.clear()
        return float(rewards.mean())

    def recent_reward(self, window=512):
        if not self.reward_history:
            return 0.0
        return float(np.mean(self.reward_history[-window:]))
