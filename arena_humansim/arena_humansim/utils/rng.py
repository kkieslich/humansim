import hashlib

import numpy as np


def derive_seed(base_seed: int, name: str) -> int:
    digest = hashlib.blake2b(f"{base_seed}/{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class RNG:
    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._substreams: dict[str, np.random.Generator] = {}
        self._agent_keys: dict[int, str] = {}
        #: the substream names each agent has drawn from, so removing one agent never touches another's (keys are free
        #: text since `bind_agent`: a prefix match on `agent_ped_` would also hit `agent_ped_2_...`)
        self._agent_streams: dict[int, set[str]] = {}

    def get_substream(self, name: str) -> np.random.Generator:
        if name not in self._substreams:
            self._substreams[name] = np.random.default_rng(derive_seed(self._seed, name))
        return self._substreams[name]

    def bind_agent(self, agent_id: int, key: str) -> None:
        """Key the agent's substreams on a stable identity instead of its id. A key another live agent holds gets the
        id appended, so two agents never share a stream."""
        if any(k == key and aid != agent_id for aid, k in self._agent_keys.items()):
            key = f"{key}#{agent_id}"
        self._agent_keys[agent_id] = key

    def _agent_prefix(self, agent_id: int) -> str:
        return f"agent_{self._agent_keys.get(agent_id, agent_id)}_"

    def get_agent_substream(self, agent_id: int, module_name: str) -> np.random.Generator:
        name = self._agent_prefix(agent_id) + module_name
        self._agent_streams.setdefault(agent_id, set()).add(name)
        return self.get_substream(name)

    def remove_agent_substreams(self, agent_id: int) -> None:
        for name in self._agent_streams.pop(agent_id, ()):
            self._substreams.pop(name, None)
        self._agent_keys.pop(agent_id, None)

    def reset(self, seed: int | None = None):
        self._seed = seed if seed is not None else self._seed
        self._rng = np.random.default_rng(self._seed)
        self._substreams.clear()
        self._agent_keys.clear()
        self._agent_streams.clear()

    @property
    def seed(self) -> int:
        return self._seed
