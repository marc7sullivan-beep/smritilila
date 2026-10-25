# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The HTTP API's shapes: every request and response body the page and the agents exchange with a world's server.

Pydantic models only, and two small helpers: one builds an answer telling nothing yet, one names an entity. No engine, no world: what is
said between the page and the server, not what the server does. Moved out of main.py unchanged (2026-10-05, Marc:
"let's go get that main.py in shape"); main re-exported every name until c3e89a4; it no longer does, so a caller names this module.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class GenesisRequest(BaseModel):
    self_description: str = ""
    # v3 (bija, 2026-09-20): the world's setting in the player's own
    # words, and how far the seed reaches beyond what the description
    # names -- here | around | wide.
    setting: str | None = None
    breadth: Literal["here", "around", "wide"] = "here"
    # The new world screen (Marc, 2026-09-24): a world from its creator's
    # brief, with no player -- sent instead of a self_description. Only
    # name, source and where_when are required; the builder fills the rest.
    name: str | None = None
    source: str | None = None  # history, a book or a film -- which (Marc, 2026-09-24: always up front)
    where_when: str | None = None
    begins: str | None = None
    different: str | None = None   # what is different here, from the source or from ordinary life (2026-10-23; it and `feel` replaced days, granted and who)
    feel: str | None = None        # what play should feel like: tone, danger, pace, how hard

    def brief(self) -> dict[str, str]:
        return {"source": self.source or "", "where_when": self.where_when or "", "begins": self.begins or "", "different": self.different or "",
                "feel": self.feel or ""}


class GenesisResponse(BaseModel):
    player_entity_id: str
    narration: str
    # Every perceiver stream genesis actually produced, not just the
    # player's own -- a self-description can mint more than one referent
    # (a location, a carried object...), and each may get its own
    # perceiver stream. Without this, a client has no way to show
    # anything but the player's line even when the backend generated more.
    # v3 fills only the player's own opening ({} for a world born from a
    # brief, which has no player).
    narrations: dict[str, str]


class EntryPointOut(BaseModel):
    location_id: str


class JoinRequest(BaseModel):
    entry_location_id: str = ""
    self_description: str
    # or, in the player's own words, where they start (Marc, 2026-10-01): a listed scene if it is one, else a new one
    start_description: str = ""


class JoinResponse(BaseModel):
    player_entity_id: str
    narration: str
    narrations: dict[str, str]  # v3: the newcomer's own opening only, as GenesisResponse's


class PulseRequest(BaseModel):
    action_text: str
    # Which of world.playable_entity_ids this pulse acts as. Omitted
    # (None) keeps the original single-character behavior: acts as
    # world.scene_owner_entity_id. Supplied, it must already be one of
    # playable_entity_ids (submit_pulse 422s otherwise) -- an id a caller
    # merely guesses at is refused, not silently minted into a new actor.
    entity_id: str | None = None
    # The page's own id for this action (Marc, 2026-09-27: "Load failed", "Failed to fetch"): a send that fails
    # is retried with the same id, and a retry of an action already received joins it instead of running it again.
    client_id: str | None = None


class PulseResponse(BaseModel):
    # v5 §22: set when the pulse is held for the player's consent -- {"kind",
    # "prompt"} and, for a long action, "minutes" and "to" (sangama.confirm_of);
    # nothing was written, and POST /pulses/confirm answers it
    confirm: dict[str, Any] | None = None
    narrations: dict[str, str]
    promoted_entity_ids: list[str]
    # From the acting player's OWN pulse only (never the scheduler's own
    # activation pulse) -- the web UI's one-click "suggested action"
    # button, smrti/narrator.py's SUGGESTED NEXT ACTION.
    suggested_next_action: str
    # A second, one-click suggestion grounded in the player's own memory
    # instead of this pulse's immediate scene -- smrti/narrator.py's own
    # suggested_next_action_from_memory, same player-pulse-only scoping.
    suggested_next_action_from_memory: str
    # v3 sara: the narration's pith for the acting player's own pulse,
    # keyed like `narrations`; the UI showed it first and folded the full
    # narration beneath until 2026-10-19, and now shows the whole. Empty for activations.
    sara: dict[str, str] = {}
    # v3 s19: place tuples written where the acting player stood while
    # they were not acting -- what they witnessed of others' turns.
    # [{place, text, said, said_by, world_time}], keyed by the acting player.
    witnessed: dict[str, list[dict[str, Any]]] = {}
    # ... and beside it, for the same player: the quoted line, what is
    # waiting, and the weight (salience) of their own tuple this pulse
    sara_detail: dict[str, dict[str, Any]] = {}
    # saṅgama: set when the action joined a round that is still open -- the round's state (RoundOut);
    # nothing was written, and the telling arrives at GET /rounds/{player}
    round: dict[str, Any] | None = None


def _told_nothing(**fields) -> PulseResponse:
    """An answer that tells nothing yet: a question held for consent, an action waiting in its round, a jump declined."""
    return PulseResponse(narrations={}, promoted_entity_ids=[], suggested_next_action="", suggested_next_action_from_memory="",
                         **fields)


class RoundMember(BaseModel):
    id: str
    name: str
    acted: bool  # whether they have acted -- never what


class RoundOut(BaseModel):
    """Saṅgama, as one player's page sees it: who is in the round with them
    and who has acted (never what), whether it is being written, whether it
    waits on their consent, and -- once written -- their own telling."""
    scene: str | None
    members: list[RoundMember]
    open: bool  # someone has acted and the round has not yet been written
    acted: bool  # this player has
    running: bool
    confirm: dict[str, Any] | None = None  # v5 §36, in a round: the jump each who acted must agree to
    waiting_on: list[str] = []  # ... and who has not yet agreed
    declined_by: str | None = None
    go_in: float = 0  # seconds until Go may close the round: GO_QUIET after the last keystroke of anyone yet to act
    told: PulseResponse | None = None


class EntityOut(BaseModel):
    id: str
    labels: list[str]
    created_at_world_time: int
    # graph/entity.py's Origin. Nothing reads its shape now: the web UI's
    # kshetraOf was the last reader, and went with v2's relations
    # (index.html says so). Defaulted so nothing else that builds an
    # EntityOut has to change.
    origin: str = "genesis"


class WorldSnapshot(BaseModel):
    world_time: int
    scene_owner_entity_id: str
    # Every entity id a POST /pulses caller may currently act as -- lets a
    # client (the web UI's character switcher) know who's switchable
    # without guessing from entity labels alone.
    playable_entity_ids: list[str]
    entities: list[EntityOut]


class WorldExistsResponse(BaseModel):
    exists: bool
    # Empty when exists is False. entity_id -> labels, for a client to
    # render a "continue as..." character picker without a second round
    # trip to GET /world.
    playable_entities: dict[str, list[str]] = {}


class WorldInfo(BaseModel):
    name: str
    # Which registered world's own database is the one currently loaded
    # -- never the database_url itself, an internal connection detail
    # the client has no use for and shouldn't see.
    active: bool
    port: int | None = None  # an author's world: the port of its own server
    description: str | None = None  # its one line, drawn from its root
    url: str | None = None  # its own public address: choosing it is going there (pinned servers, 2026-10-01)


class WorldSelectRequest(BaseModel):
    name: str


class BackupNowResponse(BaseModel):
    world_name: str
    backup_path: str


class AbsorbResponse(BaseModel):
    kept: str
    gone: str
    label: str
    alias: str
    tuples: int
    filed: int
    presence: int
    records: int = 0     # v4 records re-pointed: the gone id's own
    contained: int = 0   # ... and containment records naming it as the container


class WorldView(BaseModel):
    """v5 §7-§8: a world's own view -- the darśana (None: the default),
    the library of becomings, and the world data (pacing, capabilities)."""
    darsana: str | None = None
    becomings: dict[str, str] = {}
    world_data: dict = {}


class BhuvanaResponse(BaseModel):
    bhuvana: str | None
    chronicle: list[str] = []  # itihasa: what has moved in the age since birth, latest last


class BhuvanaRequest(BaseModel):
    # Both optional: the world's own setting and its player's earliest
    # tuples stand in for what a birth would have been given.
    self_description: str | None = None
    setting: str | None = None


class DescribeResponse(BaseModel):
    description: str | None


class ExpandResponse(BaseModel):
    places: list[str]
    people: list[str]


class LocationOut(BaseModel):
    location_id: str
    label: str
    memories: int  # tuples ever formed there, by anyone: its traffic
    players_here: list[str]


class PlayerOut(BaseModel):
    """v5 §37: a player, whether the world may act for them, whether they are here."""
    entity_id: str
    label: str
    autonomous: bool
    online: bool
    waiting: bool = False  # 2026-10-01: in the world, no turn of their own until touched


class MelanaOut(BaseModel):
    """Melana (मेलन), coming together (Marc, 2026-10-16): an online player, and where and when they stand -- to every visitor,
    so a player can see who is here and sync with them. `at` and `at_text` are None in a world with no clock."""
    entity_id: str
    label: str
    scene_id: str | None = None
    scene: str | None = None
    at: int | None = None
    at_text: str | None = None
    online: bool = True          # False for a player listed though offline: the character `as` names, or one the owner asked for with `own` (GET /world/sync): not yet logged off


class AutonomyRequest(BaseModel):
    allow: bool  # allowed is waiting -- in the world, no turn of their own until touched (v4.58)
    wait: bool = True  # kept for old pages; allowed always waits


class PresenceRequest(BaseModel):
    online: bool


class RetireRequest(BaseModel):
    confirm: str  # the character's name, typed: retiring cannot be undone


class BecomingRequest(BaseModel):
    key: str
    clause: str
    why: str = ""


class ContainmentRequest(BaseModel):
    into: str
    why: str = ""


class HeldOut(BaseModel):
    entity_id: str
    label: str
    clauses: list[str]       # what holds for them -- their measure, their powers
    first: list[str]         # their first memories: who they say they are


class MeOut(BaseModel):
    owner: bool  # the page offers what only an owner may do (switching worlds) to an owner alone
    builder: bool = False  # an owner, or a registered user the owner has enabled for the world builder (2026-10-06): the page offers them the builder


class WorldCreateRequest(BaseModel):
    name: str


class WorldCreateResponse(BaseModel):
    name: str
    database: str
    port: int | None = None  # an author's world: the port of its own server


class InviteRequest(BaseModel):
    email: str


class EntityRecap(BaseModel):
    location_id: str | None
    last_narration: str | None


class EntityMemoryOut(BaseModel):
    world_time: int
    experience: str
    salience: int


class EntityMemories(BaseModel):
    """A perceiver's own recent tuples, oldest first -- what the web UI's
    "everyone else" card shows in v3. The card used to show only the
    narrations of autonomous turns received since the page loaded, so a
    character whose every tuple was written in the player's pulses
    (Ravenmoor, 2026-09-19: Hartley) showed "(nothing remembered yet)"
    with six memories in the store."""
    memories: list[EntityMemoryOut]


class EntityScene(BaseModel):
    """Who shares an entity's scene, by the engine's own rule
    (WorldSession.here, over containment) -- so a client never needs a
    copy of the rule. `present` is the other perceivers in the entity's
    scene (Kosha.occupants: whoever has an experience tuple of their own, a
    scene only where it has the capability "perceives"), bar players out of
    the world; `adjacent` was one location over, and is always empty now."""
    location_id: str | None
    present: list[str]
    adjacent: list[str]


class AnusmrtiOut(BaseModel):
    world_time: int
    experience: str
    weight: str      # the tuple's weight, in the Narrator's own words (indelible .. habitual)
    use: str         # key | useful | none
    why: str         # the retrieval rule that brought it (own_salient, own_recent, ...)


class Anusmrti(BaseModel):
    """Anusmṛti, recollection: a being's own tuples as the world's retrieval
    selects them for its own turn (smrti/pulse.py's anusmrti), oldest first."""
    memories: list[AnusmrtiOut]


class WitnessedResponse(BaseModel):
    witnessed: list[dict[str, Any]]  # [{place, text, said, said_by, world_time}], oldest first


class ConfirmRequest(BaseModel):
    entity_id: str
    proceed: bool


class PratisthaRequest(BaseModel):
    focus: str | None = None  # a god naming what to build; none: the builder chooses, by importance and nearness
    secret: bool = False      # kept from every god outside the secret, and everything it plants born secret


class ManacitraRequest(BaseModel):
    image: str                # a file in this world's maps folder (maps/<world>/), by name: a picture of the ground to read
    cols: int = 3             # the sheet is read in cols x rows overlapping parts, each at full size
    rows: int = 3
    passes: int = 3           # ... and the whole of it this many times, so that a spelling read once loses to the one read more often
    secret: bool = False


class ProposalOut(BaseModel):
    id: str
    status: str
    title: str
    summary: list[str]
    errors: list[str]
    warnings: list[str]
    proof: dict
    survey: list[dict] = []
    report: dict = {}         # what a map's reading left out and why (smrti/manacitra.py); empty for any other proposal


def _labelled(canon, eid: str) -> str:
    """An entity's first label, as an answer shows it (its id if it has none)."""
    e = canon.entities.get(eid)
    return e.labels[0] if e and e.labels else eid
