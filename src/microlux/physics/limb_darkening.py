"""
Limb darkening profiles for extended source integration.

Defines an abstract interface and a linear law implementation with cumulative
profile used to integrate annuli efficiently.
"""

import abc

from jax import numpy as jnp


class AbstractLimbDarkening:
    """
    Abstract base class for limb darkening models.

    Required methods:
    - `profile(r)`: Intensity at radius `r` (normalized to unit mean).
    - `cumulative_profile(r)`: Integrated intensity from center to radius `r`.
    """

    @abc.abstractmethod
    def profile(self, r: jnp.ndarray) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def cumulative_profile(self, r: jnp.ndarray) -> jnp.ndarray:
        pass


class LinearLimbDarkening(AbstractLimbDarkening):
    """
    Linear limb darkening model normalized to unit total flux.
    """

    a: float

    def __init__(self, a: float):
        self.a = a

    def profile(self, r: jnp.ndarray) -> jnp.ndarray:
        """Surface brightness at radius `r` with unit mean normalization."""
        return 1 / (1 - self.a / 3) * (1 - self.a * (1 - jnp.sqrt(1 - r**2)))

    def cumulative_profile(self, r: jnp.ndarray) -> jnp.ndarray:
        """Cumulative flux from center to radius `r` (0→1)."""
        mu = jnp.sqrt(1 - r**2)

        res = (self.a * (mu**2 - 2 / 3 * mu**3) - mu**2) / (1 - self.a / 3) + 1
        return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r = jnp.linspace(0, 1, 100)
    ld = LinearLimbDarkening(1)
    plt.plot(r, ld.profile(r) / ld.cumulative_profile(1), label="limb fun")
    plt.plot(r, ld.cumulative_profile(r), label="cumulative limb fun")
    plt.legend()
    plt.show()
