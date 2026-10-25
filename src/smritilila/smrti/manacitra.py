# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Mānacitra (मानचित्र): the builder reads a map. Māna, "measure", and citra, "picture": the measured picture -- the ground as a drawing
of it, which is not the ground. Marc, 2026-10-21: the world builder should grow a world from a map, every named place on it, roads,
mountains, rivers, forests and swamps.

A map is shown to a model as a picture (llm.client.Picture), read the way a person reads one -- the great areas from the whole sheet, then
every name from each part of it at full size -- and what is read is made into a pratiṣṭhā seed (smrti/pratistha.py): places only, each with
its container and an impersonal first tuple. It is proposed, proved on a fork, reviewed and planted as any pratiṣṭhā is; nothing here
writes to a world. The engine's part stays nirguṇa (v5 §1): it keeps no coordinates and no kinds. What a map gives a world is words in
tuples -- "beside Oldford, to its north, on the road to the ford" -- which is all the tree ever was: a cue to retrieval, not a map.

  read    -- one model call for the great areas (the whole sheet), then one per part (a grid, overlapping so that a name cut by one part's
             edge lies whole in the next): every printed name, exactly as printed, with where it lies and what the map shows of it.
  merge   -- deterministic: the parts' readings made one list; a name cut by an edge and never read whole is dropped and said; a name the
             world already holds is not made again (it is where the new places are filed); of two readings a letter apart, the one read oftener stands over one read once, else both are kept and flagged.
  spec    -- deterministic: keys, containers resolved (a name, an id, else the earth), cycles broken.
Every place the map gives is a REGION, never a scene (2026-10-21, from kosha.neighbours): a scene's neighbours are the scenes in the same container, whole and
uncapped, so thirty scenes planted in one realm would each be a neighbour of every scene already there, in every prompt. A region is not a
neighbour, and play already mints a scene inside a region when someone arrives (Stonemarch's lair, under its hills). The tree shows the new
places to whoever stands beneath their holder; that is all the cost, and it is one line a place.
The seed is bounded by MAX_MAP_PLACES, not by pratiṣṭhā's one-thing cap (its spec carries `map`)."""
from __future__ import annotations

import io
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from smritilila.llm.client import Picture
from smritilila.smrti.bija import _key

MAX_MAP_PLACES = 400            # one map's seed: every name on a sheet, never a world's history
READ_ATTEMPTS = 3               # a part that cannot be read after this many tries fails the whole read: a silent hole in "every name" is worse
_STOP = frozenset({"the", "of", "a", "an", "and", "village", "hamlet", "town", "city"})
_MIMES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class MapReadFailed(RuntimeError):
    """A part of the map could not be read; nothing is proposed from a partial reading."""


# -- the picture ---------------------------------------------------------------------------------------------------------------------------

def _pillow():
    try:
        from PIL import Image
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError("reading a map needs Pillow (pip install pillow)") from exc
    return Image


def load_picture(path: str | Path) -> Picture:
    """The file as a Picture a model takes: png, jpeg and webp as they are; any other (a gif, a bmp) made a png."""
    p = Path(path)
    data = p.read_bytes()
    mime = _MIMES.get(p.suffix.lower())
    if mime:
        return Picture(data, mime, p.name)
    Image = _pillow()
    im = Image.open(io.BytesIO(data))
    im.seek(0)
    out = io.BytesIO()
    im.convert("RGB").save(out, "PNG")
    return Picture(out.getvalue(), "image/png", p.stem + ".png")


def tile_boxes(width: int, height: int, cols: int, rows: int, overlap: float) -> list[tuple[str, tuple[int, int, int, int]]]:
    """(name, (left, top, right, bottom)) for a grid of cols x rows parts that cover the sheet and overlap each neighbour by `overlap` of a
    part's own size -- so a name no wider than that lies whole in at least one of two neighbouring parts."""
    cols, rows = max(1, cols), max(1, rows)
    tw = width / (cols - (cols - 1) * overlap)
    th = height / (rows - (rows - 1) * overlap)
    out = []
    for r in range(rows):
        for c in range(cols):
            left = min(round(c * tw * (1 - overlap)), max(0, width - round(tw)))
            top = min(round(r * th * (1 - overlap)), max(0, height - round(th)))
            out.append((f"r{r + 1}c{c + 1}", (left, top, min(width, left + round(tw)), min(height, top + round(th)))))
    return out


def tiles_of(picture: Picture, cols: int = 3, rows: int = 3, overlap: float = 0.2) -> list[tuple[str, Picture]]:
    Image = _pillow()
    im = Image.open(io.BytesIO(picture.data)).convert("RGB")
    stem = Path(picture.name).stem or "map"
    parts = []
    for name, box in tile_boxes(im.width, im.height, cols, rows, overlap):
        buf = io.BytesIO()
        im.crop(box).save(buf, "PNG")
        parts.append((name, Picture(buf.getvalue(), "image/png", f"{stem} {name}")))
    return parts


# -- names ---------------------------------------------------------------------------------------------------------------------------------

def norm(name: str) -> str:
    """A printed name as it is compared: no accents, no case, no punctuation, one space."""
    plain = "".join(ch for ch in unicodedata.normalize("NFKD", str(name)) if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]+", " ", plain.casefold()).strip()


def bare(name: str) -> tuple[str, ...]:
    """A name's words without the words that only say what kind of place it is ('Village of Oldford' and 'Oldford' bear the same)."""
    return tuple(w for w in norm(name).split() if w not in _STOP)


def squash(name: str) -> str:
    """A name as two readings of it are compared: norm without its spaces ('Iron Wood' and 'IRONWOOD' are one name)."""
    return norm(name).replace(" ", "")


def match_known(name: str, known: Sequence[tuple[str, str, str]], *, exact_only: bool = False) -> list[str]:
    """The ids of the places the world already holds that this printed name IS: the same words once the kind-words are dropped; failing that --
    for a name of two words or more, and only among regions -- the opening words of a longer label ('the Kron Hills' for 'Kron Hills Northern
    Verge'). One word is never a prefix (a settlement called Kron is not the hills called Kron), and a scene is never taken for what it lies on
    (the river is not the wharf on it)."""
    me = bare(name)
    if not me:
        return []
    same = [eid for eid, label, _ in known if bare(label) == me]
    if same or exact_only or len(me) < 2:
        return same
    return [eid for eid, label, kind in known if kind == "region" and bare(label)[:len(me)] == me]


def is_acronym_of(name: str, label: str) -> bool:
    """A name of three to six letters that is the initials of three words or more that run together in the label ('ToEE' for 'Ruins of the
    Temple of the old faith'): the same place, abbreviated as a map abbreviates."""
    me = squash(name)
    words = norm(label).split()
    if not 3 <= len(me) <= 6 or len(me) != len(squash(re.sub(r"[^A-Za-z]", "", str(name)))) or " " in norm(name):
        return False
    return any("".join(w[0] for w in words[i:i + len(me)]) == me for i in range(len(words) - len(me) + 1))


def _edit_distance_one(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


# -- what the model is asked ---------------------------------------------------------------------------------------------------------------

AREAS_SYSTEM = """You read a MAP for the builder of a world kept as memories. The map is a picture of the ground; the world is what its people hold of it. You are shown the whole sheet. List every great AREA whose name is printed across it in lettering larger than a town's: a realm, kingdom, diocese, march or other domain; a territory, holdings or free lands ('Territory of X', 'Holdings of Y', 'Free Lands of Z'); a range of hills or mountains; a forest or wood; a plain, moor or other open country; a swamp, fen or marsh; a bay, sea or coast; a river or lake named in large lettering. Name each exactly as printed, in the spelling printed -- never correct, translate or complete a name, never invent one. A name you cannot read whole (cut by the edge of the picture, or hidden) is left out. Say what each is in a word or two, and give an impersonal account of it in one or two sentences: where it lies on the sheet, and which printed places lie beside or within it. Say only what the map shows. Do not list towns, castles, roads or small features: they are read later, part by part."""

AREAS_SCHEMA = {
    "type": "object",
    "properties": {"areas": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string", "description": "exactly as printed"},
        "kind": {"type": "string", "description": "what it is, in a word or two"},
        "known": {"type": "string", "description": "the id of the place the world already holds that this IS (the same place), else empty"},
        "tuple": {"type": "string", "description": "an impersonal account, one or two sentences, of what the map shows of it"},
    }, "required": ["name", "kind", "known", "tuple"]}}},
    "required": ["areas"],
}

FEATURES_SYSTEM = """You read ONE PART of a map for the builder of a world kept as memories. The map is a picture of the ground; the world is what its people hold of it. You are shown that part at full size, with what is already known: the great areas read from the whole sheet, and the places the world already holds (by id). List EVERY name printed within this part: each settlement (city, town, village, hamlet), castle, keep, tower, abbey, ruin, lair, camp, ford, bridge, pass, crossroads, mine and other marked site; each road, trail and way; each river, stream, lake and spring; each hill, mountain, forest, swamp, moor and other named piece of country that is not one of the great areas. Name every one exactly as printed, in the spelling printed -- never correct, translate or complete a name, never invent one. A name cut off by the edge of this part is marked `partial`: it will be read whole in the neighbouring part. A name inside a coat of arms or a banner is not a place. Where two things bear one name (a road and the town it leads to), list each.
For each name give: `in` -- the name of the great area it lies within (as listed below), or the name of another place printed in this part that holds it (a territory, a forest, a range), or, if it lies within a place the world already holds, that place's id; empty only if you cannot tell. `known` -- the id of the place the world already holds that this printed name IS (the same place), else empty. `tuple` -- an impersonal account, one or two sentences, of what the map shows of it: what it is, which printed places it lies beside and in which direction, and where its roads and rivers lead from it. Say only what the map shows."""

FEATURES_SCHEMA = {
    "type": "object",
    "properties": {"features": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string", "description": "exactly as printed"},
        "kind": {"type": "string", "description": "what it is, in a word or two"},
        "in": {"type": "string"}, "known": {"type": "string"},
        "partial": {"type": "boolean"},
        "tuple": {"type": "string"},
    }, "required": ["name", "kind", "in", "known", "partial", "tuple"]}}},
    "required": ["features"],
}


def _known_lines(known: Sequence[tuple[str, str, str]]) -> str:
    return "\n".join(f"  {eid}: {label}" for eid, label, _kind in known) or "  (none)"


def _position(name: str, cols: int, rows: int) -> str:
    r, c = int(name[1:name.index("c")]), int(name[name.index("c") + 1:])
    row = "the top" if r == 1 else "the bottom" if r == rows else "the middle"
    col = "the left" if c == 1 else "the right" if c == cols else "the middle"
    return f"row {r} of {rows} ({row}), column {c} of {cols} ({col})"


def read_areas(client, source: str | None, picture: Picture, known: Sequence[tuple[str, str, str]]) -> list[dict]:
    user = "\n".join([
        f"THIS WORLD'S SOURCE: {source.strip()}" if source and source.strip() else "THIS WORLD'S SOURCE: as the map shows it",
        "North is up unless the map says otherwise.",
        "PLACES THE WORLD ALREADY HOLDS (id: label):", _known_lines(known),
    ])
    out = _ask(client, AREAS_SYSTEM, user, "read_areas", "The great areas named on the map.", AREAS_SCHEMA, picture, "areas")
    return [a for a in out if isinstance(a, dict) and str(a.get("name") or "").strip()]


def read_tile(client, source: str | None, tile: str, picture: Picture, cols: int, rows: int, areas: Sequence[dict],
              known: Sequence[tuple[str, str, str]]) -> list[dict]:
    user = "\n".join([
        f"THIS WORLD'S SOURCE: {source.strip()}" if source and source.strip() else "THIS WORLD'S SOURCE: as the map shows it",
        f"THIS PART OF THE MAP: {_position(tile, cols, rows)}. North is up unless the map says otherwise. Its neighbours overlap it a little.",
        "THE GREAT AREAS (read from the whole sheet):", *(f"  {a['name']} ({a.get('kind', '')})" for a in areas),
        "PLACES THE WORLD ALREADY HOLDS (id: label):", _known_lines(known),
    ])
    out = _ask(client, FEATURES_SYSTEM, user, "read_part", "Every name printed in this part of the map.", FEATURES_SCHEMA, picture, "features")
    return [f for f in out if isinstance(f, dict) and str(f.get("name") or "").strip()]


def _ask(client, system, user, tool, description, schema, picture: Picture, field: str) -> list:
    last: Exception | None = None
    for _ in range(READ_ATTEMPTS):
        try:
            out = client.complete_with_tool(system, user, tool_name=tool, tool_description=description, input_schema=schema,
                                            max_tokens=16384, images=[replace(picture, fine=True)])   # small print: read at the finest
            rows = out.get(field) if isinstance(out, dict) else None
            if isinstance(rows, list):
                return rows
            last = RuntimeError(f"no `{field}` list in the answer")
        except Exception as exc:  # noqa: BLE001 -- a model's failure of any kind is tried again, then said
            last = exc
    raise MapReadFailed(f"{tool}: {last!r}"[:400])


# -- merge and spec ------------------------------------------------------------------------------------------------------------------------

def build_spec(areas: Sequence[dict], parts: dict[str, list[dict]], known: Sequence[tuple[str, str, str]], taken_ids: set[str],
               earth: str, image: str = "", min_votes: int = 1) -> tuple[dict, dict]:
    """The readings made one pratiṣṭhā seed: (spec, report). `known`: (id, label, kind) of every place the world holds; `taken_ids`: every id
    in the world (a new key must not be one); `earth`: the id of the world at large, where what lies in no area is filed. `areas` may hold one
    area once for each time it was read. `min_votes`: how many readings a name needs (as an area or in a part) to be made a place: a wrong name
    in a world that is only ever added to cannot be taken back, so a name read once among several readings is listed, not made.

    Every decision here is deterministic and listed in the report, so a god reading the proposal sees what was left out and why."""
    known_ids = {k[0] for k in known}
    label_of = {k[0]: k[1] for k in known}
    report: dict = {"image": image, "areas_read": len({squash(a["name"]) for a in areas}), "parts": {t: len(rows) for t, rows in parts.items()},
                    "already_held": [], "ambiguous": [], "claimed_held_but_not": [], "cut_and_never_whole": [], "misreads_resolved": [],
                    "read_too_seldom": [], "possible_misreads": [], "unplaced": [], "cycles_broken": 0, "dropped_over_cap": 0, "min_votes": min_votes}

    def held_by(row: dict) -> tuple[list[str], str]:
        """The places the world holds that this name IS, and by which rule: the same words; the model's word for it, if it shares a word with
        the place it names or abbreviates it (a model that says 'Irondelve' is 'the gnome quarry' says too much); the opening words of a region's label."""
        exact = match_known(row["name"], known, exact_only=True)
        if exact:
            return exact, "the same words"
        said = str(row.get("known") or "").strip()
        if said in known_ids:
            if set(bare(row["name"])) & set(bare(label_of[said])):
                return [said], "the model's word"
            if is_acronym_of(row["name"], label_of[said]):
                return [said], "the model's word, an abbreviation of it"
            report["claimed_held_but_not"].append({"name": row["name"].strip(), "claimed": said, "label": label_of[said]})
        opening = match_known(row["name"], known)
        return (opening, "the opening words of a region") if opening else ([], "")

    # 1. every reading of a name, parts in order: the whole readings, then the ones the model marked cut by an edge
    whole: dict[str, list[dict]] = {}
    cut: dict[str, str] = {}
    for tile in sorted(parts):
        for row in parts[tile]:
            s = squash(row["name"])
            if not s:
                continue
            if row.get("partial"):
                cut.setdefault(s, row["name"].strip())
            else:
                whole.setdefault(s, []).append(row)
    # a cut reading that is a piece of a whole one is that name (the mark is the model's and is not always right either way); what is left of
    # one, and is long enough to have been a name, is said and left out -- a name is never completed
    report["cut_and_never_whole"] = sorted(name for s, name in cut.items()
                                           if s not in whole and len(s) >= 5 and not any(s in w for w in whole))

    # 2. two readings a letter apart: the one read in more parts stands, if the other was read once; else both stand, flagged below
    for a in sorted(whole):
        for b in sorted(whole):
            if a < b and a in whole and b in whole and len(a) >= 6 and len(b) >= 6 and _edit_distance_one(a, b):
                more, less = (a, b) if len(whole[a]) > len(whole[b]) else (b, a)
                if len(whole[more]) >= 2 and len(whole[less]) == 1:
                    report["misreads_resolved"].append({"kept": whole[more][0]["name"].strip(), "dropped": whole[less][0]["name"].strip()})
                    del whole[less]

    # 3. one row for a name: the fullest account, a container from any reading that gave one
    chosen: dict[str, dict] = {}
    for s, rows in whole.items():
        best = max(rows, key=lambda r: len(str(r.get("tuple") or "")))
        container = next((str(r.get("in") or "").strip() for r in rows if str(r.get("in") or "").strip()), "")
        chosen[s] = {**best, "in": container, "name": str(best["name"]).strip()}

    # 4. the great areas first, as regions in the earth; then what is already held is set aside, the rest are places
    out: list[dict] = []
    names: dict[str, str] = {}          # a squashed name -> the key that stands for it
    held_names: dict[str, str] = {}     # ... or the id of the place the world holds
    used = set(taken_ids)

    def new_key(name: str) -> str:
        k = _key(name)
        while k in used:
            k += "_"
        used.add(k)
        return k

    def take(row: dict, inside: str) -> None:
        s = squash(row["name"])
        if not s or s in names or s in held_names:
            return
        ids, by = held_by(row)
        if ids:
            held_names[s] = ids[0]
            report["already_held"].append({"name": str(row["name"]).strip(), "ids": ids, "by": by})
            if len(ids) > 1:
                report["ambiguous"].append({"name": str(row["name"]).strip(), "ids": ids})
            return
        name = str(row["name"]).strip()
        k = new_key(name)
        names[s] = k
        out.append({"key": k, "label": name, "name": name, "in": inside, "scene": False, "tuple": str(row.get("tuple") or "").strip(),
                    "_kind": str(row.get("kind") or "")})

    votes: dict[str, int] = {}
    for a in areas:
        votes[squash(a["name"])] = votes.get(squash(a["name"]), 0) + 1
    for s, rows in whole.items():
        votes[s] = votes.get(s, 0) + len(rows)
    seldom: dict[str, str] = {}

    def enough(row: dict) -> bool:
        if votes.get(squash(row["name"]), 0) >= min_votes:
            return True
        seldom.setdefault(squash(row["name"]), str(row["name"]).strip())
        return False

    area_rows: dict[str, dict] = {}
    for a in areas:                                               # one row for an area read in several passes: the fullest account
        s = squash(a["name"])
        if s and (s not in area_rows or len(str(a.get("tuple") or "")) > len(str(area_rows[s].get("tuple") or ""))):
            area_rows[s] = a
    for a in area_rows.values():
        if enough(a):
            take(a, earth)
    for row in chosen.values():
        if enough(row):
            take(row, row["in"])
    report["read_too_seldom"] = sorted(seldom.values())

    # 5. containers: an id the world holds, or a name read, or else the earth
    def resolve(inside: str) -> str:
        if not inside:
            return ""
        if inside in known_ids or inside in names.values():
            return inside
        s = squash(inside)
        return names.get(s) or held_names.get(s) or next(iter(match_known(inside, known)), "")
    by_key = {p["key"]: p for p in out}
    areas_keys = {names[s] for s in area_rows if s in names}
    for p in out:
        if p["key"] in areas_keys:
            continue                                  # a great area is filed in the earth, as it was made
        got = resolve(p["in"])
        if not got or got == p["key"]:
            report["unplaced"].append(p["label"])
            got = earth
        p["in"] = got

    # 6. a loop of containers is broken at the place that closed it
    for p in out:
        seen, up = {p["key"]}, p["in"]
        while up in by_key:
            if up in seen:
                p["in"] = earth
                report["cycles_broken"] += 1
                break
            seen.add(up)
            up = by_key[up]["in"]

    # 7. two readings a letter apart that both stood are flagged, both kept: a misreading cannot be told from two places by the words alone
    labels = sorted({p["label"] for p in out})
    flat = {lab: squash(lab) for lab in labels}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            if len(flat[a]) >= 6 and len(flat[b]) >= 6 and _edit_distance_one(flat[a], flat[b]):
                report["possible_misreads"].append([a, b])

    if len(out) > MAX_MAP_PLACES:
        report["dropped_over_cap"] = len(out) - MAX_MAP_PLACES
        out = out[:MAX_MAP_PLACES]
    kinds = {p["key"]: p.pop("_kind") for p in out}
    report["places"] = len(out)
    report["kinds"] = {k: v for k, v in sorted(_count(kinds.values()).items(), key=lambda kv: -kv[1])[:20]}
    spec = {"title": f"The map{f' {image}' if image else ''}", "places": out, "people": [], "things": [],
            "why": "every name printed on the map, read part by part and filed where the map has it", "map": {"image": image, "parts": len(parts)}}
    return spec, report


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        k = norm(x) or "?"
        out[k] = out.get(k, 0) + 1
    return out


def read_map(client, source: str | None, picture: Picture, known: Sequence[tuple[str, str, str]], taken_ids: set[str], earth: str,
             *, cols: int = 3, rows: int = 3, overlap: float = 0.2, passes: int = 3, min_votes: int | None = None,
             workers: int = 6) -> tuple[dict, dict, dict]:
    """The whole read: the great areas, every part, and the seed made of them. (spec, report, readings): `readings` is what the model said,
    unedited -- {"areas": [...], "parts": {tile: [...]}} -- kept with the proposal so a god can see what each part read.

    A model's reading of small print wanders: the same sheet read twice gives two spellings of one name. So the sheet is read `passes` times,
    each its own sample, and a name that stands once against a neighbour spelling read more often is dropped, and a name read only once among
    several passes is listed and not made (`min_votes`, two when the sheet is read more than once; build_spec). Model calls: passes x (one
    for the areas, then cols x rows)."""
    passes = max(1, passes)
    min_votes = (2 if passes >= 2 else 1) if min_votes is None else min_votes
    with ThreadPoolExecutor(max(1, workers)) as pool:
        area_sets = list(pool.map(lambda _: read_areas(client, source, picture, known), range(passes)))
        areas, seen = [], set()
        for a in (a for each in area_sets for a in each):
            if squash(a["name"]) not in seen:                       # the areas of every pass, each once
                seen.add(squash(a["name"]))
                areas.append(a)
        parts_list = tiles_of(picture, cols, rows, overlap)
        jobs = [(f"{name}#{p + 1}" if p else name, name, tile) for p in range(passes) for name, tile in parts_list]
        results = list(pool.map(lambda j: read_tile(client, source, j[1], j[2], cols, rows, areas, known), jobs))
    parts = {label: rows_ for (label, _, _), rows_ in zip(jobs, results)}
    raw_areas = [a for each in area_sets for a in each]             # each time an area was read: a vote for it
    spec, report = build_spec(raw_areas, parts, known, taken_ids, earth, picture.name, min_votes)
    report["passes"] = passes
    return spec, report, {"areas": raw_areas, "parts": parts}
