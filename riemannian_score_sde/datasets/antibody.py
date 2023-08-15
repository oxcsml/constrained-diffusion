import jax
import jax.numpy as np


class Polytope:
    def __init__(self, npz, scale=None, rng=None, batch_size=64):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        self.rng = rng
        self.data = dict(np.load(npz))
        self.data["data"] = self.data["r"][:, 1:-1] if "walk" in npz else self.data["r"]
        self.batch_size = batch_size

    def __len__(self):
        return self.data["data"].shape[0]

    def __getitem__(self, idx):
        return self.data["data"][idx], None  # , self.data['seq'][idx]

    def __next__(self):
        self.rng, next_rng = jax.random.split(self.rng)

        idx = jax.random.choice(next_rng, len(self), shape=(self.batch_size,))

        return self[idx]

    def get_all(self):
        return self.data["data"], None  # , self.data['seq']


class PolytopeTorus:
    def __init__(self, npz, scale=None, rng=None, batch_size=64):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        self.rng = rng
        self.data = dict(np.load(npz))
        self.data["polytope"] = self.data["r"]
        self.data["torus"] = np.stack(
            [np.cos(self.data["tau"]), np.sin(self.data["tau"])], axis=-1
        ).reshape(-1, 2 * self.data["tau"].shape[1])

        self.batch_size = batch_size

    def __len__(self):
        return self.data["polytope"].shape[0]

    def __getitem__(self, idx):
        return [
            self.data["polytope"][idx],
            self.data["torus"][idx],
        ], None  # , self.data['seq'][idx]

    def __next__(self):
        self.rng, next_rng = jax.random.split(self.rng)

        idx = jax.random.choice(next_rng, len(self), shape=(self.batch_size,))

        return self[idx]

    def get_all(self):
        return [self.data["polytope"], self.data["torus"]], None  # , self.data['seq']


class LoopTorus:
    def __init__(self, npz, scale=None, rng=None, batch_size=64):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        self.rng = rng
        self.data = dict(np.load(npz))
        self.data["polytope"] = self.data["r"]
        self.data["torus"] = np.stack(
            [np.cos(self.data["tau"]), np.sin(self.data["tau"])], axis=-1
        ).reshape(-1, 2 * self.data["tau"].shape[1])

        self.batch_size = batch_size

    def __len__(self):
        return self.data["polytope"].shape[0]

    def __getitem__(self, idx):
        return self.data["torus"][idx], None

    def __next__(self):
        self.rng, next_rng = jax.random.split(self.rng)

        idx = jax.random.choice(next_rng, len(self), shape=(self.batch_size,))

        return self[idx]

    def get_all(self):
        return self.data["torus"], None  # , self.data['seq']


class Antibody:
    def __init__(self, scale=None, n_links=16, rng=None, batch_size=64):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        self.rng = rng
        self.n_links = n_links
        self.data = dict(
            np.load(
                f"/data/ziz/not-backed-up/fishman/score-sde/data/walk.0.{n_links}.npz"
            )
        )
        self.data["data"] = np.hstack(
            [
                self.data["r"][:, 1:-1],
                np.stack(
                    [np.cos(self.data["tau"]), np.sin(self.data["tau"])], axis=-1
                ).reshape(-1, 2 * self.data["tau"].shape[1]),
            ]
        )
        self.data["seq"] = self.data["seq"].astype(float)
        self.batch_size = batch_size

    def __len__(self):
        return self.data["data"].shape[0]

    def __getitem__(self, idx):
        return self.data["data"][idx], self.data["seq"][idx]

    def __next__(self):
        self.rng, next_rng = jax.random.split(self.rng)

        idx = jax.random.choice(next_rng, len(self), shape=(self.batch_size,))

        return self[idx]

    def get_all(self):
        return self.data["data"], self.data["seq"]
