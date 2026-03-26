import os
import numpy as np
import pandas as pd
import pickle


def pfsp(win_rates, weighting="squared"):
    """Prioritized Fictitious Self-Play opponent sampling."""
    win_rates = [min(i, 0.95) for i in win_rates]
    weightings = {
        "variance": lambda x: x * (1 - x),
        "linear": lambda x: 1 - x,
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared": lambda x: (1 - x) ** 2,
    }
    fn = weightings[weighting]
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm


def kbsp(win_rates, k=5):
    """K-Best opponent sampling: prioritize weakest opponents."""
    sorted_wr = sorted(win_rates)
    if len(sorted_wr) < k:
        baseline_val = sorted_wr[-1]
    else:
        baseline_val = sorted_wr[k - 1]

    probs = np.asarray(np.asarray(win_rates) <= baseline_val, np.float32)
    norm = probs.sum()
    if norm == 0:
        probs = np.ones_like(win_rates)
        norm = probs.sum()
    else:
        norm_rest = float(probs.sum()) * 0.15
        z_cnt = sum(1 for p in probs if p == 0)
        for i in range(len(probs)):
            if probs[i] == 0 and z_cnt > 0:
                probs[i] = norm_rest / float(z_cnt)
        norm = probs.sum()

    return probs / norm


class WinrateTracker():
    def __init__(self, nmin=500, nmax=500):
        self.n = 0
        self.v = 0.5
        self.nmin = nmin
        self.nmax = nmax

    def update(self, v):
        self.n += 1
        self.clp_n = np.clip(self.n, self.nmin, self.nmax)
        self.v = self.v * (self.clp_n - 1) / self.clp_n + v / self.clp_n


class League():
    """
    League training manager for competitive self-play.

    Maintains a pool of historical policy weights and tracks win rates
    against each archived opponent. Uses PFSP (Prioritized Fictitious
    Self-Play) to sample opponents that the current agent struggles against.

    Key concepts:
    - policy_id == -1 means self-play (current agent vs itself)
    - Other policy_ids are archived snapshots
    - Opponents are resampled every N games per environment
    - New weights are archived when win rate exceeds threshold
    """
    def __init__(self, initial_weight=None, n=500, last_num=1000,
                 kbest=5, output_dir=None, selfplay_ratio=0.3):
        self.weights_dic = {}
        self.current_pid = -1
        self.pids = []
        self.winrates = None
        self.n = n
        self.last_num = last_num
        self.output_dir = output_dir
        self.kbest = kbest
        if initial_weight is not None:
            self.add_weight(initial_weight)
        self._loaded = False
        self.selfplay_ratio = selfplay_ratio
        self.current_point = -1

    def has_policy(self, policy_id):
        return policy_id in self.weights_dic

    def weight_number(self):
        return len(self.pids)

    def get_all_weights_dic(self):
        return self.weights_dic

    def get_all_policy_ids(self):
        return self.pids

    def get_latest_policy_id(self):
        return self.pids[-1]

    def get_weight(self, policy_id):
        return self.weights_dic[policy_id]

    def select_opponent(self, ret_weight=False):
        probs = pfsp([i.v for i in self.winrates[-self.last_num:]])
        policy_id = np.random.choice(self.pids[-self.last_num:], p=probs)
        if ret_weight:
            return policy_id, self.get_weight(policy_id)
        else:
            return policy_id

    def select_opponent_batch(self, n):
        """Select a batch of opponents, mixing archived policies and self-play."""
        probs = pfsp([i.v for i in self.winrates[-self.last_num:]])
        # Sparse sampling for large leagues
        if len(probs) > 64:
            for i in range(len(probs)):
                if i != 0 and i != len(probs) - 1 and i % (len(probs) // 5) != 0:
                    probs[i] = 0
        # Add self-play probability
        p = np.insert(probs * (1 - self.selfplay_ratio),
                      len(probs),
                      [np.sum(probs) * self.selfplay_ratio])
        p = p / p.sum()
        policy_ids = np.random.choice(
            self.pids[-self.last_num:] + [-1],
            size=(n,), p=p)
        return policy_ids

    def initized(self):
        return len(self.pids) > 0

    def initize_if_possible(self, new_weight):
        if not self.initized():
            self.add_weight(new_weight)

    def save_weight(self, new_weight):
        """Save weight snapshot to disk."""
        if self.output_dir:
            os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
            self.current_point += 1
            fname = os.path.join(self.output_dir, 'models',
                                 'interval_{}.pkl'.format(self.current_point))
            with open(fname, 'wb') as whdl:
                pickle.dump(new_weight, whdl)

    def add_weight(self, new_weight, dump=False):
        """Add a new policy weight to the league archive."""
        self.current_pid += 1
        self.pids.append(self.current_pid)
        self.weights_dic[self.current_pid] = new_weight
        n = self.n
        if self.winrates is None:
            self.winrates = [WinrateTracker(n, n) for _ in self.pids]
        else:
            old_winrates = self.winrates
            self.winrates = [WinrateTracker(n, n) for _ in self.pids]
            for i in range(min(len(self.winrates), len(old_winrates))):
                self.winrates[i].v = old_winrates[i].v
        self.selfplay_winrate = WinrateTracker(n, n)

        if self.output_dir and dump:
            os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
            fname = os.path.join(self.output_dir, 'models',
                                 'league_{}.pkl'.format(self.current_pid))
            with open(fname, 'wb') as whdl:
                pickle.dump(new_weight, whdl)

    def set_winrates(self, winrates):
        assert len(winrates) == len(self.winrates)
        for i in range(len(winrates)):
            self.winrates[i].v = winrates[i]

    def update_result(self, policy_id, result, selfplay=False):
        """Update win rate tracker after a game result."""
        if selfplay:
            self.selfplay_winrate.update(result)
        else:
            self.winrates[policy_id].update(result)

    def winrate_all_match(self, winrate):
        """Check if win rate exceeds threshold against all recent opponents."""
        if len(self.winrates) > 7:
            winrate_processed = []
            for i in range(len(self.winrates)):
                if i != 0 and i != len(self.winrates) - 1 and i % (len(self.winrates) // 5) != 0:
                    pass
                else:
                    winrate_processed.append(self.winrates[i])
        else:
            winrate_processed = self.winrates
        return np.all([i.v > winrate for i in winrate_processed[-self.last_num:]])

    def get_statics_table(self, dump=True):
        """Generate a summary table of win rates vs all opponents."""
        names = ["self-play"] + ["c_" + str(i) for i in self.pids]
        winrates = [self.selfplay_winrate.v] + [i.v for i in self.winrates]
        nums = [self.selfplay_winrate.n] + [i.n for i in self.winrates]
        table = pd.DataFrame({
            "oppo": names,
            "winrate": winrates,
            "matches": nums,
        }).T

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            table.to_csv(os.path.join(self.output_dir, "winrates.csv"),
                         header=False, index=False)
        return table
