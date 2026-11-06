from __future__ import annotations

import numpy as np

from arena_humansim.utils.rng import RNG


def test_same_seed_identical_substream(rng_pair: tuple[RNG, RNG]) -> None:
    a, b = rng_pair
    sa = a.get_substream("planner")
    sb = b.get_substream("planner")
    va = sa.random(10)
    vb = sb.random(10)
    assert np.array_equal(va, vb)


def test_distinct_substream_names_independent() -> None:
    r = RNG(42)
    s1 = r.get_substream("name_a")
    s2 = r.get_substream("name_b")
    v1 = s1.random(10)
    v2 = s2.random(10)
    assert not np.array_equal(v1, v2)


def test_same_substream_name_returns_same_generator() -> None:
    r = RNG(42)
    s1 = r.get_substream("foo")
    s2 = r.get_substream("foo")
    assert s1 is s2


def test_substream_order_independent_of_first_access() -> None:
    r1 = RNG(42)
    r2 = RNG(42)
    v1 = r1.get_substream("a").random(5)
    _ = r2.get_substream("b").random(5)
    v1_again = r2.get_substream("a").random(5)
    assert np.array_equal(v1, v1_again)


def test_reset_restores_state() -> None:
    r = RNG(42)
    s = r.get_substream("x")
    first = s.random(5)
    r.reset()
    s2 = r.get_substream("x")
    second = s2.random(5)
    assert np.array_equal(first, second)


def test_reset_with_new_seed_changes_stream() -> None:
    r = RNG(42)
    first = r.get_substream("x").random(5)
    r.reset(seed=7)
    second = r.get_substream("x").random(5)
    assert r.seed == 7
    assert not np.array_equal(first, second)


def test_agent_substream_namespaced() -> None:
    r = RNG(42)
    a = r.get_agent_substream(1, "planner")
    b = r.get_substream("agent_1_planner")
    assert a is b


def test_remove_agent_substreams() -> None:
    r = RNG(42)
    r.get_agent_substream(1, "planner")
    r.get_agent_substream(1, "perception")
    r.get_agent_substream(2, "planner")
    r.remove_agent_substreams(1)
    assert "agent_1_planner" not in r._substreams
    assert "agent_1_perception" not in r._substreams
    assert "agent_2_planner" in r._substreams


def test_seed_property() -> None:
    r = RNG(123)
    assert r.seed == 123


def test_removing_an_agent_leaves_streams_of_a_key_it_prefixes() -> None:
    r = RNG(42)
    r.bind_agent(1, "ped")
    r.bind_agent(2, "ped_2")
    kept = r.get_agent_substream(2, "planner")
    r.get_agent_substream(1, "planner")
    r.remove_agent_substreams(1)
    assert r.get_agent_substream(2, "planner") is kept


def test_two_live_agents_with_one_key_draw_from_their_own_streams() -> None:
    r = RNG(42)
    r.bind_agent(1, "ped")
    r.bind_agent(2, "ped")
    assert r.get_agent_substream(1, "planner") is not r.get_agent_substream(2, "planner")


def test_a_key_freed_by_a_removed_agent_is_taken_whole() -> None:
    r = RNG(42)
    r.bind_agent(1, "ped")
    first = r.get_agent_substream(1, "planner").integers(1 << 30)
    r.remove_agent_substreams(1)
    r.bind_agent(7, "ped")
    assert r.get_agent_substream(7, "planner").integers(1 << 30) == first
