"""Colored noise implementations for Stable Baselines3"""
from enum import Enum

import numpy as np
import torch as th
from pink.cnrl import ColoredNoiseProcess
from stable_baselines3.common.distributions import DiagGaussianDistribution


class NoiseColors(float, Enum):
    WHITE = 0.0
    PINK = 1.0
    RED = 2.0


class ColoredNoiseDist(DiagGaussianDistribution):
    def __init__(self, beta, seq_len, action_dim=None, rng=None, action_low=None, action_high=None):
        """Gaussian colored noise distribution for using colored action noise with stochastic policies.

        This class implements a non-squashed distribution suitable for
        PPO, but not SAC. The class keeps an internal storage of
        pre-generated pink-noise and allows for the use of batched
        actions, as required for vectorized environments. Changing the
        number of vectorized environments (requested actions) will
        lead to new internal generators.

        Parameters
        ----------
        beta : float or array_like
            Exponent(s) of colored noise power-law spectra. If it is a single float, then `action_dim` has to be
            specified and the noise will be sampled in a vectorized manner for each action dimension. If it is
            array_like, then it specifies one beta for each action dimension. This allows different betas for different
            action dimensions, but sampling might be slower for high-dimensional action spaces.
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int, optional
            Dimensionality of the action space. If passed, `beta` has to be a single float and the noise will be
            sampled in a vectorized manner for each action dimension.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        epsilon : float, optional, by default 1e-6
            A small value to avoid NaN due to numerical imprecision.

        """
        assert (action_dim is not None) == np.isscalar(
            beta
        ), "`action_dim` has to be specified if and only if `beta` is a scalar."

        assert (action_low is None and action_high is None) or (
            action_low is not None and action_high is not None
        ), "set either none of action_low and action_high, or both."
        self.action_low = -1.0 if action_low is None else action_low
        self.action_high = 1.0 if action_high is None else action_high
        self.action_halfspan = (self.action_high - self.action_low) / 2.0
        self.action_middle = (self.action_high + self.action_low) / 2.0
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.rng = rng
        self._gens = {}
        if np.isscalar(beta):
            self.beta = beta
            super().__init__(self.action_dim)
        else:
            self.beta = np.asarray(beta)
            super().__init__(len(self.beta))

    def get_gen(self, n_envs, device="cpu"):
        if n_envs not in self._gens:
            if np.isscalar(self.beta):
                gen = ColoredNoiseProcess(beta=self.beta, size=(n_envs, self.action_dim, self.seq_len), rng=self.rng)
            else:
                gen = [ColoredNoiseProcess(beta=b, size=(n_envs, self.seq_len), rng=self.rng) for b in self.beta]
            self._gens[n_envs] = gen
        return self._gens[n_envs]

    def sample(self) -> th.Tensor:
        device = self.distribution.mean.device
        n_envs, action_dim = self.distribution.batch_shape
        gen = self.get_gen(n_envs, device=device)
        if np.isscalar(self.beta):
            cn_sample = th.tensor(gen.sample(), dtype=th.float32, device=device)
        else:
            cn_sample = th.tensor([cnp.sample() for cnp in gen], dtype=th.float32, device=device)
        self.gaussian_actions = self.distribution.mean + self.distribution.stddev * cn_sample
        return self.gaussian_actions * self.action_halfspan + self.action_middle

    def __repr__(self) -> str:
        return f"ColoredNoiseDist(beta={self.beta})"
