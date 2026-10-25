# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The fanout rerun's finding (holders_full): a new container put straight into a holder that already lists FANOUT_CAP or more places.

Marc, 2026-10-13 ("fix #1 to plain containers"; scenes do not exist, it is all containers): a new container is counted by what this pulse put
into it, not only by the scene bit. Stonemarch 1553: a leatherworker's cottage, a holder and not a scene, went straight into Millhaven (27 places)
and nothing looked at it. There was no test of this rule before (it was built and watched in play)."""

from __future__ import annotations

from types import SimpleNamespace

from smritilila.memory.record import Record
from smritilila.smrti.pulse import FANOUT_CAP, holders_full


class _Kosha:
    """Just enough of Kosha for holders_full: which holders list how many places, and which entities carry the scene bit."""

    def __init__(self, width: dict[str, int], scenes: set[str], not_places: set[str] = frozenset()) -> None:
        self.width, self.scenes, self.not_places = width, scenes, set(not_places)

    def is_place(self, eid, t, depth=3):
        return eid not in self.not_places

    def is_scene(self, eid):
        return eid in self.scenes

    def holds(self, holder, t):
        return [f"{holder}-{i}" for i in range(self.width.get(holder, 0))]

    def label(self, eid):
        return eid


def _run(records, minted, width, scenes=(), not_places=("jakal", "guard")):
    b = SimpleNamespace(minted=set(minted), earth="earth")
    return holders_full(_Kosha(width, set(scenes), set(not_places)), b, records, 10)


def _declared(entity, holder):
    return Record(f"d-{entity}", "kosha", entity, holder, 10, "p", "declared")


def _moved(entity, into):
    return Record(f"m-{entity}", "kosha", entity, into, 10, "p", "moved")


FULL = {"village": FANOUT_CAP + 19}


def test_a_new_container_that_is_not_a_scene_but_has_someone_in_it_is_counted() -> None:
    """The lane at Stonemarch 1558 as a plain container: the actor walks into it this pulse, so it holds something."""
    found = _run([_declared("lane", "village"), _moved("jakal", "lane")], {"lane"}, FULL)
    assert found == {"village": ("village", FANOUT_CAP + 19, ["lane"])}


def test_a_new_thing_with_nothing_in_it_is_not_a_container() -> None:
    assert _run([_declared("coin", "village")], {"coin"}, FULL) == {}


def test_the_scene_bit_still_counts_an_empty_new_place() -> None:
    assert _run([_declared("room", "village")], {"room"}, FULL, scenes={"room"}) != {}


def test_a_wrapper_is_the_group_only_if_it_gathers_two_or_more_places() -> None:
    """Stonemarch 1553 and 1543: a cottage round its porch, the Cellars round the vaults. The holder gained one child each time, as without
    the wrapper, so a wrapper of one is a flat addition. A wrapper that gathers two places stands for them in the width (one where there
    would have been two), and so does one that takes the holder's existing places in."""
    cottage = [_declared("cottage", "village"), _declared("porch", "cottage"), _moved("jakal", "porch")]
    assert _run(cottage, {"cottage", "porch"}, FULL) == {"village": ("village", FANOUT_CAP + 19, ["cottage"])}      # round one new place
    two = [_declared("row", "village"), _declared("shop", "row"), _declared("smithy", "row"), _moved("jakal", "shop")]
    assert _run(two, {"row", "shop", "smithy"}, FULL, scenes={"smithy"}) == {}                                       # round two new places (the empty one carries the bit)
    regroup = [_declared("wing", "village"), _moved("old-0", "wing"), _moved("old-1", "wing")]
    assert _run(regroup, {"wing"}, FULL) == {}                                                                       # two existing places taken in
    mixed = [_declared("wing", "village"), _declared("hall", "wing"), _moved("old-0", "wing"), _moved("jakal", "hall")]
    assert _run(mixed, {"wing", "hall"}, FULL) == {}                                                                 # one new place and one existing
    people = [_declared("wing", "village"), _declared("guard", "wing"), _moved("jakal", "wing")]
    assert _run(people, {"wing", "guard"}, FULL, not_places=("jakal", "guard")) != {}                                # people are not places: counted
    one_old = [_declared("wing", "village"), _declared("hall", "wing"), _moved("jakal", "hall")]
    assert _run(one_old, {"wing", "hall"}, FULL) != {}                                                               # one new place alone: counted


def test_a_holder_below_the_cap_is_never_full() -> None:
    assert _run([_declared("lane", "village"), _moved("jakal", "lane")], {"lane"}, {"village": FANOUT_CAP - 1}) == {}


def test_a_container_an_existing_thing_is_moved_into_is_not_new() -> None:
    """Moving the actor into a container that already existed adds no child to the full holder."""
    assert _run([_moved("jakal", "inn")], set(), FULL) == {}
