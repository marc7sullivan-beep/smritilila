# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""v3 genesis (v5 §39): a world begins as tuples.
Its name: sṛṣṭi (सृष्टि), creation (v5 §39).

A world file names its places (each a scene, or a region that holds
scenes, with one impersonal first tuple), its people (each with a
few first tuples in their own perspective, standing in a place), who is
playable, and the setting. The seeder writes entities, memories and
records (what holds what, names, clauses, natures) and nothing else: no
geometry. The builder writes tuples and those records; v2's relations
are gone.

    PYTHONPATH=src .venv/Scripts/python -m smritilila.smrti.genesis worlds/vadhu_crossing.json postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_vadhu

Refuses a database that already holds entities.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import func, select

from smritilila.graph.entity import Entity
from smritilila.ids import new_id
from smritilila.llm.embedder import SentenceTransformerEmbedder
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.persistence.engine import make_engine, make_session_factory
from smritilila.persistence.models import EntityRow
from smritilila.persistence.repository import save_memories, save_records, save_session_config, upsert_entities
from smritilila.smrti.narrator import salience_of, use_of
from smritilila.smrti.log import NOWHERE
from smritilila.smrti.pulse import mnemonic_id


from smritilila.smrti.kala import ITIHASA  # noqa: E402  Entity.origin of the world at large
ITIHASA_LABEL = "the world at large"


def plant_chronicle(bhuvana: str, embed) -> tuple[Entity, Memory, Record]:
    """Itihāsa (इतिहास, 'thus it was'; 2026-09-20): the world at large as a
    perceiver of its own -- standing nowhere, impersonal like a place's
    stream, drawn on far ticks like any far actor.
    Its name: bhūmi (भूमि), earth, ground (v5 §14).
    Its first tuple is
    THE YEAR of the bhuvana; its turns chronicle what moves in the age.
    Whoness only (s0): this id has a stream. It declares its name, as every
    planted entity does (v5 §16/§39)."""
    head = bhuvana.split("THE REGIONS", 1)[0].strip() or bhuvana[:1200]
    e = Entity(mnemonic_id(ITIHASA_LABEL, new_id()), [ITIHASA_LABEL], 0, ITIHASA)
    m = Memory(new_id(), e.id, frozenset(), head, "PRIMARY", 3, NOWHERE, 0, "genesis", embed(head))
    return e, m, Record(new_id(), "name", e.id, ITIHASA_LABEL, 0, "genesis")


def plant_person(label: str, tuples: list, place: str, now: int, pulse_id: str, embed, *, eid: str | None = None,
                 backdate: bool = False, minute: int | None = None) -> tuple[Entity, list[Memory], Record]:
    """A person planted with their first tuples, standing in `place` -- one
    way for genesis, an expansion and a newcomer's arrival. `tuples`: plain
    strings (an authored seed: the last weightier) or {"text", "weight"}
    (the Narrator's word, s2). Placed by a record, as a pulse places (v4
    v5 §10), not by the entity's field. `backdate`: genesis only -- the first
    tuples run up to now (-20, -10, 0), before anything; in a live world
    they are all at `now`, in order, never on ticks others have lived.
    `minute`: kāla -- where the person's time starts (v5 §20, Marc 2026-09-25:
    a newcomer joins at the current time of the place they start in)."""
    eid = eid or mnemonic_id(label, new_id())
    person = Entity(eid, [label], now, "genesis")
    n = len(tuples)
    prefix = new_id()[:24]  # ids that sort in order within one tick (the store reads by tick, then id)
    memories = []
    for i, t in enumerate(tuples):
        text = t if isinstance(t, str) else t["text"]
        sal = (3 if i == n - 1 else 4) if isinstance(t, str) else salience_of(t.get("weight"), default=4)
        tick = now - 10 * (n - 1 - i) if backdate else now
        use = use_of(t.get("use")) if isinstance(t, dict) and t.get("use") else None  # the Narrator's word; an authored string has none
        memories.append(Memory(f"{prefix}{i:08x}", eid, frozenset(), text, "PRIMARY", sal, place, tick, pulse_id, embed(text),
                               use=use, began=minute, ended=minute))
    return person, memories, Record(new_id(), "kosha", eid, place, now, pulse_id, note="declared")


def place_records(entities: list[Entity], world_time: int, pulse_id: str) -> list[Record]:
    """A planted place's first container, as a record (v5 §10); the entity's
    field is kept only as the old fallback."""
    return [Record(new_id(), "kosha", e.id, e.kosha, world_time, pulse_id, note="declared") for e in entities if e.kosha]


def _place_entity(eid: str, p: dict, world_time: int, ids: dict[str, str]) -> Entity:
    """A seed place as an entity. v4 (v5 §11, §39): `scene` true is a scene;
    false is a region -- a container with no scene bit and no kind; `in`
    is its first container, fixed at birth. A row with no scene bit (v3's
    `kind`, an authored world file) is a scene, as every v3 place was (v5 Appendix B)."""
    into = ids.get(p.get("in") or "")
    return Entity(eid, [p["label"]], world_time, "genesis", avarana="scene" if p.get("scene", True) else None, kosha=into)


def name_records(entities: list[Entity], names: dict[str, str], world_time: int, pulse_id: str) -> list[Record]:
    """v5 §16/§39: every entity planted declares its name -- the seed's `name`,
    else its whole label -- so no v4-born world is ever named by the v3
    capital-letter derivation. names: entity id -> declared name."""
    out = []
    for e in entities:
        name = (names.get(e.id) or (e.labels[0] if e.labels else "")).strip()
        if name:
            out.append(Record(new_id(), "name", e.id, name, world_time, pulse_id))
    return out


def clause_records(places: dict[str, dict], ids: dict[str, str], world_time: int, pulse_id: str) -> list[Record]:
    """v5 §7: a seed place whose physics differs from its world carries a
    clause that holds WITHIN it -- written once, at planting, reviewed as
    the seed is. Settled first (settle_clauses): a law is written where it
    holds, not wherever it was described."""
    laws = settle_clauses(places)
    return [Record(new_id(), "clause", ids[k], text, world_time, pulse_id, note=f"within:{k}")
            for k, text in laws.items() if k in ids]


def prakriti_records(places: dict[str, dict], ids: dict[str, str], world_time: int, pulse_id: str) -> list[Record]:
    """v5 §15: a seed place whose NATURE differs from what holds it bears it,
    holding within it -- a time rate, closed to strangers, places that
    perceive, perception that reaches. Written once, at planting, as the
    seed said; read from the tree (Kosha.rate, seed, capable)."""
    # settled as laws are (settle): a nature every place directly in a region carries is the region's, and one its
    # container already carries is inherited, not repeated -- a copied rate would compose with itself
    natures = settle({k: json.dumps(p["prakriti"], sort_keys=True) for k, p in places.items() if p.get("prakriti")}, places)
    out: list[Record] = []
    for k, said in natures.items():
        nature, eid = json.loads(said), ids.get(k)
        if eid is None:
            continue
        if nature.get("rate"):
            out.append(Record(new_id(), "prakriti", eid, json.dumps({"rate": nature["rate"]}), world_time, pulse_id, note="time"))
        if nature.get("closed"):
            out.append(Record(new_id(), "prakriti", eid, json.dumps({"arrivals": False}), world_time, pulse_id, note="pacing"))
        for cap in ("perceives", "reaches"):
            if nature.get(cap):
                out.append(Record(new_id(), "capability", eid, cap, world_time, pulse_id, note="within:on"))
    return out


def settle_clauses(places: dict[str, dict]) -> dict[str, str]:
    """Where each seed law holds (the sample world, 2026-09-30: an island's
    law written on each of its scenes, never on the island, so a place made
    later inside the island stood under none, and nested scenes showed it
    twice). A law every place directly in a region carries -- two or more,
    word for word -- is the region's; a clause a place's container already
    carries is inherited, not repeated. One place carrying a law keeps it:
    the engine cannot tell a region's law from that place's own (the chapel
    in the village), so the seed is asked to write a region's law on the
    region. Within the seed's own tree; text compared, nothing read."""
    return settle({k: str(p["clause"]).strip() for k, p in places.items() if str(p.get("clause") or "").strip()}, places)


def settle(laws: dict[str, str], places: dict[str, dict]) -> dict[str, str]:
    """settle_clauses' rule for any value a seed place carries (a law; v5 §15, its
    nature): {place key: value}, each moved to where it holds."""
    laws = dict(laws)
    parent = {k: p.get("in") for k, p in places.items() if p.get("in") in places}
    kids: dict[str, list[str]] = {}
    for k, up in parent.items():
        kids.setdefault(up, []).append(k)
    changed = True
    while changed:  # bottom up, until nothing moves: an island's scenes, then the island's own region
        changed = False
        for region, under in kids.items():
            texts = {laws.get(k) for k in under}
            if region not in laws and len(under) >= 2 and len(texts) == 1 and None not in texts:
                laws[region] = texts.pop()
                for k in under:
                    del laws[k]
                changed = True

    def inherited(k: str) -> str | None:
        up, seen = parent.get(k), set()
        while up is not None and up not in seen:
            seen.add(up)
            if up in laws:
                return laws[up]
            up = parent.get(up)
        return None
    return {k: text for k, text in laws.items() if inherited(k) != text}


def plant_more(new: dict, known_place_ids: dict[str, str], embed, world_time: int, pulse_id: str = "vistara",
               earth: str | None = None) -> tuple[list[Entity], list[Memory], list[Record]]:
    """Vistāra: plant an expansion into a world that exists. New places
    and people get opaque ids; a person may stand in a known place (by
    the key the Narrator was shown) or a new one. First tuples land at
    the world's clock: a place's at now, a person's at now, in order."""
    ids: dict[str, str] = dict(known_place_ids)
    entities: list[Entity] = []
    memories: list[Memory] = []
    for key, p in new["places"].items():
        ids[key] = mnemonic_id(p["label"], new_id())
    for key, p in new["places"].items():
        entities.append(_place_entity(ids[key], p, world_time, ids))  # v5 §39: what it is in, stated at birth
        memories.append(Memory(new_id(), ids[key], frozenset(), p["tuple"], "AMBIENT", 5, ids[key], world_time, pulse_id, embed(p["tuple"])))
    placed = place_records(entities, world_time, pulse_id)
    if earth is not None:  # what the ring puts in nothing is in the earth (Marc, 2026-09-30)
        placed += [Record(new_id(), "kosha", e.id, earth, world_time, pulse_id, note="earth") for e in entities if not e.kosha]
    for key, person in new["people"].items():
        ids[key] = mnemonic_id(person["label"], new_id())
        e, mems, rec = plant_person(person["label"], person["tuples"], ids[person["place"]], world_time, pulse_id, embed, eid=ids[key])
        entities.append(e)
        memories += mems
        placed.append(rec)
    names = {ids[k]: v.get("name", "") for part in ("places", "people") for k, v in new[part].items() if k in ids}
    return entities, memories, (placed + name_records(entities, names, world_time, pulse_id)
                                + clause_records(new["places"], ids, world_time, pulse_id)
                                + prakriti_records(new["places"], ids, world_time, pulse_id))


def build(spec: dict, embed, bhuvana: str | None = None) -> tuple[list[Entity], list[Memory], dict]:
    """Entities and first tuples from a world spec. Ids are opaque to the
    Narrator by contract but copyable (pulse.py's mnemonic_id: the label's
    words plus a short tail -- 2026-09-21, until then the whole genesis
    cast was 32-hex and the boundary handle map had to cover for it);
    the spec's own keys are only for cross-reference inside the file."""
    ids: dict[str, str] = {}
    entities: list[Entity] = []
    memories: list[Memory] = []
    for key, p in spec["places"].items():
        ids[key] = mnemonic_id(p["label"], new_id())
    for key, p in spec["places"].items():
        # v5 §39: a world states its tree at birth -- `in`, the key of the
        # place (or region) this one is in; read as its first container
        entities.append(_place_entity(ids[key], p, 0, ids))
    placed = place_records(entities, 0, "genesis")
    for key, p in spec["places"].items():
        memories.append(Memory(new_id(), ids[key], frozenset(), p["tuple"], "AMBIENT", 5, ids[key], 0, "genesis", embed(p["tuple"])))
    for key, person in spec["people"].items():
        ids[key] = mnemonic_id(person["label"], new_id())
        # the last first-tuple is "now" (tick 0); a plain string takes an authored seed's weights (the last weightier)
        e, mems, rec = plant_person(person["label"], person["tuples"], ids[person["place"]], 0, "genesis", embed,
                                    eid=ids[key], backdate=True)
        entities.append(e)
        memories += mems
        placed.append(rec)
    # Earth (Marc, 2026-09-30): the world at large, the root beneath every place, holding all that can
    # come from it; a perceiver from birth -- its first memory the bhuvana, or the setting in a world that
    # keeps no age. Every place the seed put in nothing is in it.
    earth, earth_memory, earth_name = plant_chronicle(bhuvana or spec.get("setting") or spec.get("name") or "a world", embed)
    place_ids = {ids[k] for k in spec["places"]}
    placed += [Record(new_id(), "kosha", e.id, earth.id, 0, "genesis", note="earth") for e in entities if e.id in place_ids and not e.kosha]
    entities.append(earth)
    memories.append(earth_memory)
    config = {
        "earth": earth.id,
        "scene_owner_entity_id": ids[spec["scene_owner"]] if spec["scene_owner"] else "",  # a brief's world: no one yet
        "entry_point_location_ids": [ids[k] for k in spec["entry_points"]],
        "playable_entity_ids": [ids[k] for k in spec["playable"]],
        "setting_description": spec.get("setting"),
        # v5 §16/§39: every planted entity declares its name
        "records": placed + [earth_name] + name_records([e for e in entities if e.id != earth.id],
                                                         {ids[k]: v.get("name", "") for part in ("places", "people")
                                                          for k, v in spec[part].items()}, 0, "genesis")
                   + clause_records(spec["places"], ids, 0, "genesis")
                   + prakriti_records(spec["places"], ids, 0, "genesis"),
        # v5 §7-§8: the world's own view, as the seed wrote it (None: the default)
        "view": spec.get("view") or {"darsana": None, "becomings": {}, "world_data": {}},
    }
    return entities, memories, config


async def seed(spec_path: str, dsn: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    embedder = SentenceTransformerEmbedder()
    entities, memories, config = build(spec, embedder.embed)
    engine = make_engine(dsn)
    factory = make_session_factory(engine)
    async with factory() as session:
        existing = (await session.execute(select(func.count(EntityRow.id)))).scalar()
        if existing:
            raise SystemExit(f"refusing: {dsn} already holds {existing} entities")
        await upsert_entities(session, entities)
        await save_memories(session, memories)
        await save_records(session, config["records"])
        await save_session_config(
            session, config["scene_owner_entity_id"], config["entry_point_location_ids"],
            config["playable_entity_ids"], config["setting_description"],
        )
        await session.commit()
    await engine.dispose()
    print(f"seeded {spec['name'].encode('ascii', 'replace').decode()}: {len(entities)} entities, {len(memories)} first tuples; player = {config['scene_owner_entity_id']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("dsn")
    a = ap.parse_args()
    asyncio.run(seed(a.spec, a.dsn))


if __name__ == "__main__":
    main()
