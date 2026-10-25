# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""No scene tag (SMRITILILA_NO_SCENES=1, kosha.no_scenes; the default of every launcher since 2026-10-19, Marc: "turn it on for all worlds and for world creation").

A container is a shared moment (a scene) once a perceiver has stood directly in it, and a perceiver never is one. A perceiver is anything that has an experience tuple (v5 §4), so
when the Narrator voices a place -- a perceiver tuple for the shop beside its scene row, which it does in about one pulse in a hundred (Stonemarch 16 of 2306, Crossing 6 of 597,
FAFO 3 of 96) -- the place perceives from then on, and by the rule it is no longer a scene: the people in it share the scene outside, and a move into it is refused (`not_a_scene`). Met on
2026-10-19 by building a world through the builder and playing in it (notes/lived_later_offline_2026-10-19/create_sim.py): the butcher's shop was voiced in Arthur's world turn and
became a perceiver. I took it for a bug and guarded against it; Marc: "that's not a bug. It will go quiet if it's not interacted with as it should." The guard is out; these tests
pin what he ruled, and what the tag still does.
"""

from __future__ import annotations

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.smrti.kosha import Kosha
from smritilila.smrti.pulse import run_smrti_pulse

VIEW = {"darsana": None, "becomings": {}, "world_data": {}}


def _world(tagged: bool):
    """A town where Bea stands, with a shop Arthur stands in, a loft nobody has stood in, and a phone Arthur carries. Tagged: the town, the shop and the loft carry the bit."""
    bit = "scene" if tagged else None
    ents = {
        "town": Entity("town", ["the town"], 0, "genesis", avarana=bit),
        "bea": Entity("bea", ["Bea"], 0, "genesis", kosha="town"),
        "shop": Entity("shop", ["the shop"], 0, "genesis", avarana=bit, kosha="town"),
        "loft": Entity("loft", ["the loft"], 0, "genesis", avarana=bit, kosha="shop"),
        "art": Entity("art", ["Arthur"], 0, "genesis", kosha="shop"),
        "phone": Entity("phone", ["his phone"], 0, "genesis", kosha="art"),
    }
    mems = [Memory("m1", "art", frozenset(), "I opened the shop.", "PRIMARY", 4, "shop", 1, "p1", None),
            Memory("m2", "phone", frozenset(), "I rode in his apron pocket.", "PRIMARY", 4, "shop", 1, "p1", None),
            Memory("m3", "bea", frozenset(), "I crossed the square.", "PRIMARY", 4, "town", 1, "p1", None)]
    return ents, mems


def _voice_the_shop(monkeypatch, tagged: bool):
    """One pulse by Arthur in which the Narrator voices the shop as a perceiver, as well as writing its scene row: (result, canon, memories, events)."""
    ents, mems = _world(tagged)
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **kw: events.append((e, kw)))
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **kw: None)

    class C:
        def complete_with_tool(self, system, user, **kw):
            return {"narration": "You sweep the sawdust toward the door.",
                    "perceiver_tuples": [{"perceiver": "art", "text": "I swept the sawdust toward the door.", "salience": "ordinary"},
                                         {"perceiver": "shop", "text": "The shop stays quiet as the butcher works.", "salience": "ordinary"}],
                    "scene_tuples": [{"scene": "shop", "text": "The butcher sweeps the shop floor in the quiet of the morning.", "present": ["art"]}],
                    "moves": [], "suggested_next_action": ""}
    canon = Maya(world_time=3, entities=dict(ents))
    r = run_smrti_pulse(canon, "art", "I sweep up.", 3, "p3", mems, C(), None, view=VIEW)
    return r, canon, mems, events


def test_a_container_is_a_scene_once_a_perceiver_has_stood_in_it_and_a_perceiver_never_is(monkeypatch) -> None:
    monkeypatch.setenv("SMRITILILA_NO_SCENES", "1")
    ents, mems = _world(tagged=False)
    k = Kosha(Maya(world_time=3, entities=ents), mems, [])
    assert k.is_scene("shop") and k.is_scene("town")                     # Arthur and the phone stood in the shop, Bea in the town, with no bit on either
    assert not k.is_scene("loft")                                        # nobody has stood in this
    assert not k.is_scene("art") and not k.is_scene("phone") and not k.is_scene("bea")   # Arthur holds the phone, and holds is not a scene: a perceiver is none
    assert k.scene_of("art", 3) == "shop" and k.scene_of("phone", 3) == "shop"   # the phone is where its holder is
    assert set(k.occupants("shop", 3)) == {"art", "phone"}
    monkeypatch.delenv("SMRITILILA_NO_SCENES")                                # the same records read with the tag: no bit anywhere, so the shop is no scene
    k2 = Kosha(Maya(world_time=3, entities=_world(tagged=False)[0]), mems, [])
    assert not k2.is_scene("shop")


def test_with_the_tag_a_scene_voiced_as_a_perceiver_is_dropped_and_stays_a_scene(monkeypatch) -> None:
    """The tag says what is a scene, so a tuple for one is its impersonal record's business (v5 §31): dropped and counted, nothing written for it as a perceiver."""
    monkeypatch.delenv("SMRITILILA_NO_SCENES", raising=False)
    r, canon, mems, events = _voice_the_shop(monkeypatch, tagged=True)
    assert [kw["scene"] for e, kw in events if e == "smrti.scene_voiced_dropped"] == ["shop"]
    assert not [m for m in r.memories_written if m.entity_id == "shop" and m.tier == "PRIMARY"]
    k = Kosha(canon, mems + r.memories_written, r.records_written)
    assert k.is_scene("shop") and not k.is_perceiver("shop") and k.scene_of("art", 4) == "shop"


def test_without_the_tag_a_place_the_narrator_voices_is_a_perceiver_from_then_on(monkeypatch) -> None:
    """Marc, 2026-10-19: "that's not a bug." A perceiver is whatever has an experience tuple, so the voiced shop perceives and is no scene (a perceiver never is one under the rule);
    Arthur, still in it, shares the scene outside; nothing is dropped, because there is no scene there to veto the tuple. A guard that dropped it would use 'scene' as a kind
    to overrule a perceiver, which is what the derived rule exists to be rid of."""
    monkeypatch.setenv("SMRITILILA_NO_SCENES", "1")
    r, canon, mems, events = _voice_the_shop(monkeypatch, tagged=False)
    assert not [e for e, kw in events if e == "smrti.scene_voiced_dropped"]
    assert [m for m in r.memories_written if m.entity_id == "shop" and m.tier == "PRIMARY"]      # the shop's own tuple stands
    k = Kosha(canon, mems + r.memories_written, r.records_written)
    assert k.is_perceiver("shop") and not k.is_scene("shop")
    assert k.scene_of("art", 4) == "town"                                                       # Arthur shares the scene outside, where he was in the shop's
