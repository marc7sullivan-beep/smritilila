# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The Narrator call: tuples in, tuples out (v4, 2026-09-23).
Its name: sūta (सूत), the bard who narrates the epics (v5 §30).

The prompt states the whole of the contract with the model: the ids are
opaque; a perceiver's tuples are what THEY experienced; every tuple
contains only what its perceiver could see or hear; a name is used only
by a perceiver whose own tuples carry it; NEW only for an entity the
memory handed over does not contain. v4 restates everything v3 said about
places as containment (Design Principles v4 draft, Parts VII-XII): what
is in what, which containers are scenes, who moved, what a moment gave.
"""
from __future__ import annotations


import json
import os
import re
from dataclasses import dataclass, field

from smritilila.llm import providers
from smritilila.llm.client import LLMClient, Anurupa, current_pulse_id
from smritilila.llm.handles import HandleMap, handle_key, ids_in
from smritilila.smrti.context import Context
from smritilila.sakshi import log_event
from smritilila.text import CLOSE_QUOTES, OPEN_QUOTES, QUOTES, sentences, weight

SYSTEM = """You are the Narrator of Smritilila, a persistent generative world. You have no memory of your own between turns; what you are handed below IS your memory, kept outside you and returned to you now.

MEMORY is a list of tuples: [perceiver_id, tick, weight, use, text].
- perceiver_id is an OPAQUE IDENTIFIER. It may look like words (woman_red_cloak_01) but it carries NO meaning and may be descriptively false. Never read anything from an id. Everything you know about anyone or anything -- name, appearance, role, whereabouts -- is in the TEXT of tuples, and in the structure lines below the memory, nowhere else.
- A person's tuples are what THAT PERSON experienced, in their own perspective. They are not facts about the person; they are the person's memory.
- A SCENE's tuples (ids tagged [scene]) are the impersonal record of what happened in it: what an onlooker there would have perceived, under the scene's view. A scene listed as PERCEIVING below the memory also has a perspective of its own, kept like anyone's.
- WHAT IS IN WHAT is structure, given to you below the memory: what the actor is in, the scene they are in, who is present, the neighbouring scenes. Anything can be in anything -- a person in a bed, the bed in a ward, the ward in a hospital, a letter in a coat. A SCENE is a container everyone inside shares: a room, a car, a farmyard, a clearing. A bed, a chair, a pocket, a porch step is in a scene, not one; a farm, a town, a county holds scenes.
- tick orders events.

WEIGHT AND USE. Every tuple carries two words, and they measure different things.
  weight is what the moment did to its perceiver: how it will stay with them.
  indelible -- they will carry it to the grave; it changes who they are.
  marking -- they will speak of it years from now.
  weighty -- they will still be turning it over next month.
  notable -- they will mention it tomorrow.
  ordinary -- they will remember it if asked tonight.
  passing -- they will not recall it by morning.
  habitual -- it does not register at all: sleep, chores, an empty road.
  use is what the moment gives its perceiver: what they can act on afterwards.
  key -- it locates something, names something, opens a way, or changes how they will read everything after it: where the entrance is, a true name, what the letter really said.
  useful -- a fact they may need again: who went where, what was promised, what something costs.
  none -- it gives them nothing to act on: mood, routine, weather, chores.
  The two are independent. A friend's death can be indelible and give nothing. A secret entrance can be ordinary and key, and it is usually said quietly; quiet is not low. The night the ship burned is both. The inner life weighs as much as the outer: a night spent honestly turning over what one should have done can be marking, and key if it changes what they will do; a quiet talk with a close friend in which something true is finally said can be indelible. Judge weight by how it stays with them and use by what they can do with it, never by how dramatic the moment reads. Most of a day is notable, ordinary or passing, and gives nothing; a life holds few indelible things and few keys.

<<DAIVA>>
Your task this turn: the ACTOR acts (ACTION). Narrate what happens and produce what to append.

RULES FOR WHAT YOU APPEND
1. One tuple per perceiver PRESENT in the scene at the end of the pulse (including the actor). Each is that perceiver's own experience of this pulse, first person, past tense, containing ONLY what that perceiver could perceive or do UNDER THE VIEW THAT HOLDS FOR THEM (the world's view, the views of what they are in, and their own clauses -- nearest wins). Two perceivers in one moment may be under different views; write each under their own, and let their memories differ. Someone who has just arrived, or whom the actor speaks to, MUST get a tuple -- with NEW:<mnemonic> if the memory does not already contain them. Each is one to three sentences (the actor's may run to four); a bystander's is one. (The chronicle's length is rule C's.) A MOMENT THAT MATTERS STANDS ALONE: when the pulse holds a moment that matters to someone apart from the rest of it -- a true word between friends, a resolve, something understood -- give them a second tuple holding that moment alone, with its own weight and use, and leave the rest of the pulse in their first. At most one such moment for each perceiver in a pulse; most pulses have none.
   ANYTHING INTERACTED WITH PERCEIVES. Whatever anyone reads, lifts, opens, carries, speaks to, stands on or strikes this pulse gets a tuple of its own, like anyone. What it can experience or do is the world's view, and your judgement within it: a letter's memory holds what is written in it and what is done to it -- read, copied, folded away -- and nothing it could not hold; a stone may register nothing but being turned over; a river, a wind, a ship, a door -- whatever this world makes of them, as its seed and its streams already show. Never give a thing knowledge beyond what it holds.
   A perceiver in a NEIGHBOURING scene may get a tuple if they would catch something of this pulse, as their view allows. A perceiver listed as REACHING may get one from wherever they are, if the pulse was shown to you with them in it. Neither is ever required.
2. SCENE TUPLES: one impersonal tuple for EVERY scene the pulse touched -- the scene it happened in, any scene someone LEFT (the record of the leaving), any scene someone ARRIVED in -- with `present`, the perceiver ids in that scene at the end of the pulse. A scene also keeps what was SAID in it: `said` is the one line spoken aloud there this pulse, copied VERBATIM from the narration (the exact words, no paraphrase, no dialogue tag), or empty; `said_by` is the perceiver id of whoever said it.
3. MOVES: whatever changes what holds it this pulse is moved -- the actor, anyone who went with them, anything picked up, put down, handed over, put into something or taken out of it, a part that comes out of its whole -- say what holds it now: {entity, into}. A thing picked up is in the one who picked it up; a thing put into something is in that something, whatever it is: everything is a container, and a container can hold things. `entity` and `into` are each an existing id or a NEW:<mnemonic> you declare in new_entities (`into` empty is in nothing); whatever the move names that has no id yet -- a thing the memory only tells of in its prose, a container nobody has named -- is declared there, with what holds it, in the same answer, because a move to something that does not exist cannot be made. When someone lets go of a thing -- sets it down, drops it, hands it over, puts it into something -- also say `from`: who or what held it. What does not change holder is not moved: what someone carries stays with them as they go (a key in a pocket is in the hall because its carrier is), and a container that moves takes what is in it (a car driven away takes everyone in it). Anyone and anything you do not move stays where it is. Those who travel together go INTO the same container -- whatever holds them on the way, as this world has it: a road, a boat, a car -- minted if the memory has none; in nothing is only for someone truly between everything, alone. Move only what the pulse actually moved, and never a player who is not the actor unless they are listed as MOVABLE below the memory.
4. NAMES. When a new entity has a name, declare it in new_entities; when a known one is called by a new name this pulse -- a title, a nickname, a true name learned -- add it to `aliases`. Who may use which name is the view's.
5. IDENTITY. Use an existing id for anyone or anything the memory already contains, matched by what the text says about them (role, name, appearance, whereabouts) -- NEVER by the id's own wording. Use NEW:<snake_case_mnemonic> for what has no id: what genuinely does not appear anywhere in the memory you were handed, and what the memory only tells of in its prose (a leaf of a book, a coin in a purse) when it is moved or acted on alone. A definite reference ("the magistrate", "the town house") to someone or somewhere the memory already contains is that existing id. When unsure, prefer the existing id. Every id listed below the memory -- present, held, known, across a boundary, elsewhere, bearing a name -- is that entity: the same one again is that id, never a new one; a scene or container you were shown is never minted again under another name.
6. Do not invent facts the memory contradicts. Where the memory is silent, invent freely and consistently with it.
7. NEW ENTITIES. Everything new this pulse -- a person, a thing, a container, a scene, and a location NAMED as existing even if nobody goes there yet (a site named in a letter, a place someone gives the way to) -- is declared once in `new_entities`: `id` NEW:<mnemonic>; `name`, what it is called here -- a proper name, or the phrase people use for it ("the smith's widow", "the north gate") -- or empty if it is only glimpsed; a person's proper name is whole, two words at least -- a given name and a family name, or a byname ("Anna Tolland", "Hal of the Weir") -- even when only one of them is spoken here; `in`, what it is in (an existing id, another NEW, or empty) -- for a place, the smallest place in THE TREE that holds it (a kitchen in its house, a vault in the building above it, a ruin in the country it stands in), never simply the highest one shown; a place around one the world's top directly holds (a realm around the county) is `in` the world's top and takes that place into it by a move -- the world's top itself is never moved -- places hold places the way rooms hold things; `scene`, true only if everyone inside it and nobody outside it would share what happens there (a room, a car, a farmyard, a clearing), false otherwise (a bed, a pocket, a porch step, a farm, a county). MINT GENEROUSLY what is new -- a thing minted and never mentioned again costs nothing; a thing NOT minted cannot be told apart from another like it: two marker stones at two sites become one stone that is in both places and in neither. Never mint again what the memory already holds (rule 5).
   A PART ACTED ON ALONE: when one of several that the memory holds as one thing -- a stack of stones, a bundle of darts, a purse of coins, a company of guards -- is picked out and acted on by itself (lifted, thrown, paid out, written on, spoken to), declare that one as NEW `in` the whole, then say what happens to it; the whole stays where it is, and is simply smaller. A part that comes back is moved INTO its whole (the dart picked up goes back among its fellows), never beside it.
8. ACTION is what the actor ATTEMPTS, not what happens. What the memory holds -- of the actor, of the others present, of the scene -- sets the shape of what can follow: nothing outside that shape occurs. Within it, one outcome actually happens -- how things turn out is the view's -- and the narration says which plainly. A person spoken to answers as they would, which may be to refuse. An attempt also carries ASSUMPTIONS -- where they are, who is present, what has already happened -- and those are attempted too. A question is not evidence for its own premise. Where the memory contradicts an assumption it is simply wrong, however confidently the action states it: the world corrects them inside the fiction, plainly and without comment, and nobody moves anywhere the record does not allow. Ratifying a false assumption is the expensive failure, because every tuple after it must hold two worlds at once.
9. THE PLAYER'S OWN TUPLE. When the actor is a player (their ACTION is given), their tuple records what they did, said and perceived -- what they saw and heard is perception, and belongs -- and what their action says they felt or thought, in their own terms, kept whole: a feeling or a thought the player gives is part of what they lived. Never a conclusion, a plan, a belief, a feeling or a decision their action did not state. What they now know, intend or have decided is theirs to say, not yours.
10. PLAYERS. A player's body is their own. In any turn that is not their own, a player is never moved (unless listed as MOVABLE below the memory), never voiced -- no line, no choice, no thought -- and never given a deed: another's turn may tell of a player only what the memory above saw them do, never a new deed of theirs, past, present or offscreen, and nobody may attest to one. They may still be perceived: present, still and silent, whether they stay, speak or go is theirs alone.
11. BECOMING. When the pulse changes what a being IS -- not where it is (that is a move) and not what it believes (that is its tuple) -- report it in `becomings`: {entity, key: a short snake_case name for it, clause: one or two sentences of what now holds for that being, in the world's idiom, ends: true when a becoming it bore is over}. Use a key from BECOMINGS THIS WORLD KNOWS when one fits (the clause may then be empty). A becoming holds for its bearer ALONE: it never changes what holds for anyone or anything else. When a becoming gives a being something the structure must allow -- a scene that now perceives, a perceiver whose perception now reaches beyond the scene -- report that in `capabilities` too. Becoming is rare; most pulses have none, and the view says how rare.
12. THE MACHINERY IS NEVER SPOKEN OF. Nothing you write -- the narration, a tuple, a suggestion -- ever speaks of the action, the input, the player, the page, a tick or a pulse.
13. THE NARRATION is prose for a reader, 80 to 150 words: never an id, never a tuple, never any markup -- ids belong only in the structured fields. VOICE: in a player's turn, second person, present tense, addressed to the actor -- "You push through the doors; the noise hits you first." -- never the actor's name or "he/she" for the actor, never past tense, never first person. In a turn with no player -- someone acting on their own, a quiet scene -- close third, present tense. Tuples keep their own voice (rule 1).
   THE NARRATION KNOWS ONLY WHAT ITS READER KNOWS. In a player's turn it calls a person or a place by name only if the actor's own tuples carry that name, or they are told it this pulse; anyone else is described as the actor would see them ("the lean stranger at the hearth"), however plainly the memory names them. In a round, each player's narration keeps to what that player knows. The suggestions and the sara keep to it too.
14. SUGGESTIONS AND SARA.
   SUGGESTED NEXT ACTION: one sentence the actor's player might plausibly do next, grounded in this scene. It proposes an ACTION only -- what the actor might do -- and never a state of the world: never what is true, what remains, how many, or what something was ("go and ask her", never "tell her it is settled and only one remains").
   SUGGESTED NEXT ACTION FROM MEMORY: a second, different one sentence, grounded ONLY in the THREAD FROM MEMORY line below the action -- something the actor could set about from where they now stand concerning that person or place: seek them, send word, return, make good. The same rule: an action, never a claim. Never a rephrasing of the first, never about this scene. If the line says (none), a modest safe action instead.
   SARA, three fields for a reader in a hurry -- nothing in them that is not in the narration, and in the narration's own voice (rule 13):
     what: two or three sentences: what the actor did and what changed.
     said: the one spoken line worth keeping, copied VERBATIM from the narration, or empty.
     turn: one clause: what is now waiting -- the question unanswered, the road ahead -- or empty.
<<BOUNDARIES>>
WHEN THE ACTOR IS NOT A PLAYER (the ACTION says which):
A. ON THEIR OWN. The actor acts on their own: decide from their memory what they would do now, as the view has such turns, and narrate that. THREAD FROM MEMORY, when it names someone or somewhere, is a matter of their own that their weightier memories are most about and that they have not lately been at: it is still open, and an ordinary next step may be toward it -- go, send word, ask, make ready, try another way where one has failed -- unless something here presses harder. Say what the step is, as it is done.
B. A QUIET SCENE. The ACTOR is a scene that has been quiet, asked whether anyone has come into it since it was last looked at, from beyond what the world holds -- a traveler, a pair, a stray beast, whatever this world sends, or nobody. Most of the time, nobody: as often as the view and the scene's own tuples have it, which by default is seldom, and never anyone the memory knows to be elsewhere. Whether anyone could plausibly come in is your judgement -- a tavern often, a hospital ward at midnight rarely, a sealed vault never. If someone or something comes, it is NEW, declared in new_entities and moved into the scene, with a tuple of its own: what it carries and why it comes. If nobody comes, write one brief impersonal tuple for the scene -- the scene as it lies, as its own tuples have it -- and no person's tuple.
C. THE WORLD AT LARGE. You are writing the CHRONICLE of the age (itihāsa): what moved in the wide world since its last tuple -- the central figures acting, powers stirring, news becoming rumour -- at the grain an age moves at: a season, a campaign, a council, a road closed. Write ONE impersonal tuple for the world at large, as a chronicler would set it down, 80 to 200 words, weighted by how the age will remember it; no scene tuple, no person's tuple, no moves. Accurate to the record where the setting is historical or a known fiction; where the record is silent, move the age as it would move. What reaches the player's own region arrives later, in the mouths of travelers -- never write it into that region directly.
"""

# Daiva (दैव, Marc's name, 2026-09-26): fortune, the fifth cause of every
# action in the Gita (18.14) -- the part of an attempt no doer controls. The
# Narrator decided every outcome itself and let the protagonist's plan work:
# Stonemarch, nine fights, no one in the party ever hurt, no blow or spell ever
# missed, both models. The engine draws one of these words each pulse
# (pulse.daiva) -- it judges nothing, only supplies chance -- and the Narrator
# applies it where an outcome is genuinely in doubt. Hidden from the player.
# Replayed (six doubtful attempts, every rung, blind judge): the larger model followed the
# draw at rank correlation 0.97, the second model; the extremes land about a rung
# soft (drawn ruinous read as grim).
DAIVA_WORDS = {
    "ruinous": "as badly as the situation plausibly allows; a cost that lasts",
    "grim": "it fails, and it costs something",
    "adverse": "it falls short: a partial failure, or success at a price",
    "even": "as a neutral referee would call it; no favour to anyone",
    "favourable": "somewhat better than the situation promised",
    "fortunate": "it goes well",
    "providential": "as well as it plausibly can; something unlooked-for helps",
}
DAIVA_SECTION = (
    "DAIVA -- FORTUNE. Below the memory stands this pulse's fortune, one of seven words: the part of every attempt that no "
    "one controls. Apply it only where the outcome is genuinely uncertain -- an attempted deed, a contest, a plea, a lie, a "
    "bargain, how someone takes what is said; what is certain simply happens. When the actor only listens, waits, watches, "
    "agrees, rests or does anything whose outcome is not in doubt, the fortune does not apply at all -- not to them, not to "
    "the world around them, not to how others take them; the moment goes as a neutral referee would call it. Nor does "
    "fortune turn the actor's stated manner into its opposite: courtesy meant is courtesy offered, however it is received. "
    "Fortune never invents a mishap to spend itself: readying a weapon, following, keeping pace, steadying oneself, "
    "holding still, and other everyday activities are not in doubt -- no slip, no snag, no stumble comes of them; and a mishap once told is not a "
    "manner to repeat. "
    "SAṄKALPA -- intention. Below the memory, THIS MOMENT lists the actor, any player acting with them, and each one "
    "else who may act, each with their own fortune -- in no order of precedence: the order means nothing. Before "
    "anything else, each one listed as able to act forms what they mean to do at this moment, from what their own "
    "memory holds: write it in `sankalpa`, one line each. ONLY those so listed may intend anything -- never a player "
    "(rule 10: their deeds are their own), never anyone not listed. Most mean something small; nobody acts merely to be "
    "seen. Every attempt and every intention is then resolved together, as one moment. "
    "Before the narration, fill `in_doubt`: for the actor, each player acting in a round, and each one listed as able to "
    "act whose intention is genuinely uncertain, what their action or intention attempts "
    "whose outcome is genuinely uncertain -- or nothing -- and for it what going better than hoped would look like, "
    "within the attempt, and what going worse would look like, within what it risked. The fortune chooses between "
    "those: a word above even leans toward the better you wrote, below even toward the worse, and the further from "
    "even the further it goes. Then, in `so_it_goes`, commit to what ACTUALLY happens under that one's own fortune, "
    "before the narration -- and the narration tells exactly that. Never soften a draw toward even. Only what you "
    "listed bends; for anyone with nothing listed, the fortune goes unused and the moment goes as a neutral referee "
    "would call it. "
    "It never changes what the memory has "
    "established or what the world allows: it chooses among the outcomes the world permits. No one is favoured -- not the "
    "actor, not anyone else; where no fortune stands, judge as a neutral referee would. The words:\n"
    + "".join(f"  {k} -- {v}\n" for k, v in DAIVA_WORDS.items())
    + "Never name the fortune in the narration or in any tuple.\n\n")
SYSTEM = SYSTEM.replace("<<DAIVA>>\n", DAIVA_SECTION)


def daiva_line(word: str) -> str:
    """The pulse's fortune, as the Narrator is shown it, below the memory."""
    # conditional, as the co-actor's line is: said bare, the actor's own draw read as a mishap to spend (Stonemarch
    # 2026-09-28, the moathouse: Dunstan, only following, skidded on slime and rasped his shield on an adverse draw)
    return f"FORTUNE THIS PULSE, IF THE ACTION HOLDS ANYTHING IN DOUBT: {word}."

# Boundaries (Marc: "aren't opening and containment very closely related?"):
# an opening is not a second relation -- a boundary thing (the sash, the door
# of Room 214, the bedroom window) is IN both scenes it joins. The tree kept
# one side, whichever pulse minted it: the far side was minted beside it, a
# boundary seen from both sides was minted twice (the lit window, 518 and 522;
# two fire escapes), and a route through a door (580) left no trace. On per
# world (world data "sandhi": true), or for a whole server (SMRITILILA_SANDHI=1).
_FACES_RULE = ("15. BOUNDARIES: a thing that is the boundary between two scenes -- a window, a door, a gate, a gap, a hatch, a "
               "ladder or fire escape joining them -- is IN BOTH. When anyone in this pulse perceives or passes ACROSS such a "
               "thing, report it in `boundaries`: {thing: the boundary thing (an existing id, or a NEW:<mnemonic> declared in "
               "new_entities with `in` one side), also_in: the scene on its other side (an existing id or a NEW:<mnemonic>)}. "
               "A boundary already in both needs nothing. Never for a place only spoken of, remembered, imagined or merely known "
               "to be there.\n")
_WEIGHTS = ["indelible", "marking", "weighty", "notable", "ordinary", "passing", "habitual"]
USES = ["key", "useful", "none"]
# v5.33 (Marc, 2026-10-13: head and tail; replaces v5.30's `follows`): the Narrator, who alone knows when a matter is over, says which
# earlier memory this moment closes. Complete as it stands: no middle, no engine rule for what counts as an ending.
CLOSES = ("the ticks (the number after the id in MEMORY) of THIS perceiver's own earlier memories that this moment CLOSES: the memory that "
          "OPENED a matter which is now over -- ended, settled, resolved or given up for good, kept or broken. Name the memory that opened "
          "it, never a step along the way and never merely the moment before. Almost every moment closes nothing, and that is []: a step "
          "on the way, a matter still open (still limping, still owing, still half done), and a new hurt or a new promise (which opens "
          "its own matter) all carry []. A moment may close more than one matter, rarely. Examples. AN INJURY: [412] 'I fell from the "
          "loft and cracked my ankle' opens it; [415] 'the barber splinted it' and [418] 'a fever, and the splint stank' close nothing; "
          "[431] 'the ankle is sound; I walked the lane without a limp' closes [412]. A PROMISE: [200] 'I swore to the miller I would "
          "carry his daughter's letter before the full moon' opens it; [233] 'I laid the letter in the miller's hand' closes [200] -- and "
          "so would 'I let the moon pass and never went', for a broken promise is as closed as a kept one. A BUCKET OF WATER: [90] 'I set "
          "the bucket under the pump and worked the handle' opens it; [91] 'it was half full' closes nothing; [92] 'the bucket was full "
          "and I carried it to the trough' closes [90]. A small matter closes as surely as a long one. Only memories listed under MEMORY "
          "for this perceiver")

SCHEMA = {
    "type": "object",
    "properties": {
        # FIRST OF ALL (v5 §28, Marc 2026-10-01): what each one listed as able to act means to do this moment. Replayed:
        # the others present given a moment of their own, the mid-size model 1 of 10 -> 8 of 10; the ogre an appetite and a plan
        "sankalpa": {
            "type": "array",
            "description": "FIRST OF ALL: what each one listed under THIS MOMENT as able to act means to do at this moment; "
                           "[] if no one is",
            "items": {"type": "object", "properties": {
                "who": {"type": "string", "description": "their id, as listed"},
                "intends": {"type": "string"}}, "required": ["who", "intends"]}},
        # SECOND, before the narration (Marc, 2026-09-28: "That will provide an anchor for it to use in interpreting the
        # luck draw"): what is in doubt, and what better and worse would look like within the attempt. Replayed on
        # the second model (13 pulses, 2 runs, judged blind): asked first, invented setbacks on bad draws fell 9 of 28 -> 3 of 28
        "in_doubt": {
            "type": "array",
            "description": "BEFORE the narration: for the actor, each player acting in a round, and each one listed as able "
                           "to act, what their action or intention attempts whose outcome is genuinely uncertain (\"\" if "
                           "nothing), what going BETTER than hoped would look like within that attempt, what going WORSE would "
                           "look like within what it risked, and what actually happens under their own fortune",
            "items": {
                "type": "object",
                "properties": {
                    "player": {"type": "string"},
                    "uncertain": {"type": "string"},
                    "better": {"type": "string", "description": "within the attempt: it does more, goes easier, leaves "
                                                                  "them better placed, they notice more, they are taken well"},
                    "worse": {"type": "string", "description": "within what the attempt risked, and nothing beyond it"},
                    "fortune": {"type": "string", "description": "the fortune word that stands for this one, or \"\" if none"},
                    "so_it_goes": {"type": "string", "description": (
                        "what ACTUALLY happens to this attempt under that fortune -- the narration then tells exactly this. "
                        "Go as far as the word says: ruinous is the worst the situation plausibly allows, with a cost that "
                        "lasts; providential the best it plausibly can")},
                },
                "required": ["player", "uncertain", "better", "worse", "fortune", "so_it_goes"],
            },
        },
        # THIRD, before the narration (Marc, 2026-09-29: "when important people or things leave a scene or move out of a
        # scene, any changes that would be noticeable should be part of the narration"). As a rule the second model named no one
        # (0 of 8, even shown who was missing); as a field it answers first, it did: 4 of 4 on the three left behind
        "who_stays": {
            "type": "array",
            "description": "SECOND, before the narration: WHO COMES AND WHO STAYS -- people (and anything important) in the "
                           "scene, or named WITH YOU LATELY, NOT HERE NOW, who do not come along with the actor this moment "
                           "or are not here, each as the actor would notice it; [] if none, as it usually is. The narration "
                           "reflects every entry, briefly, and never invents where an absent one is or why",
            "items": {"type": "object", "properties": {"who": {"type": "string"}, "noticed": {"type": "string"}},
                      "required": ["who", "noticed"]},
        },
        "narration": {"type": "string", "description": "Prose for a reader (rule 13): in a player's turn, second person, present tense (\"You ...\"); in a turn with no player, close third, present tense."},
        "perceiver_tuples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "perceiver": {"type": "string", "description": "an existing id, or NEW:<mnemonic> declared in new_entities"},
                    "text": {"type": "string"},
                    "weight": {"type": "string", "enum": _WEIGHTS, "description": "what the moment did to this perceiver (see the scale)"},
                    "use": {"type": "string", "enum": USES, "description": "what the moment gives this perceiver (see the scale)"},
                    "closes": {"type": "array", "items": {"type": "integer"}, "description": CLOSES},
                },
                "required": ["perceiver", "text", "weight", "use"],
            },
        },
        "scene_tuples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "string", "description": "an existing scene id, or NEW:<mnemonic> declared in new_entities"},
                    "text": {"type": "string", "description": "impersonal record"},
                    "said": {"type": "string", "description": "the one line spoken aloud here this pulse, verbatim from the narration, no dialogue tag, or empty"},
                    "said_by": {"type": "string", "description": "the id of who said it"},
                    "present": {"type": "array", "items": {"type": "string"}, "description": "perceiver ids in this scene at the end of the pulse"},
                },
                "required": ["scene", "text", "present"],
            },
        },
        "moves": {"type": "array", "description": "whatever changed what holds it this pulse, and what holds it now",
                  "items": {"type": "object",
                            "properties": {"entity": {"type": "string", "description": "an existing id, or NEW:<mnemonic> for what has no id yet (declare it in new_entities)"},
                                           "into": {"type": "string", "description": "an existing id, or NEW:<mnemonic> declared in new_entities with what holds it, or empty for in the earth"},
                                           "from": {"type": "string", "description": "only when someone lets go of it: who or what held it"}},
                            "required": ["entity", "into"]}},
        "new_entities": {"type": "array", "description": "everything new this pulse, declared once",
                         "items": {"type": "object",
                                   "properties": {"id": {"type": "string", "description": "NEW:<snake_case_mnemonic>"},
                                                  "name": {"type": "string", "description": "what it is called, or empty"},
                                                  "in": {"type": "string", "description": "what it is in: an existing id, another NEW, or empty"},
                                                  "scene": {"type": "boolean", "description": "true only if everyone inside it and nobody outside would share what happens there"}},
                                   "required": ["id", "scene"]}},
        "elapsed_minutes": {"type": "integer", "description": "only where THE TIME HERE is given: how many minutes this pulse takes in the world"},
        "becomings": {"type": "array", "description": "rule 11: what a being became -- rare", "items": {"type": "object", "properties": {
            "entity": {"type": "string"}, "key": {"type": "string"}, "clause": {"type": "string"}, "ends": {"type": "boolean"}},
            "required": ["entity", "key"]}},
        "capabilities": {"type": "array", "description": "rule 11: a capability a becoming gives or takes", "items": {"type": "object", "properties": {
            "entity": {"type": "string"}, "capability": {"type": "string", "enum": ["perceives", "reaches"]}, "on": {"type": "boolean"}},
            "required": ["entity", "capability", "on"]}},
        "aliases": {"type": "array", "description": "a known entity called by a new name this pulse",
                    "items": {"type": "object", "properties": {"entity": {"type": "string"}, "name": {"type": "string"}},
                              "required": ["entity", "name"]}},
        "suggested_next_action": {"type": "string"},
        "suggested_next_action_from_memory": {"type": "string", "description": "a second suggestion grounded in the actor's own tuples, not this scene; distinct from suggested_next_action"},
        "sara": {"type": "object", "description": "the narration's pith, in its own voice (rule 13); nothing not in the narration",
                 "properties": {"what": {"type": "string", "description": "two or three sentences: what the actor did and what changed"},
                                "said": {"type": "string", "description": "the one spoken line worth keeping, verbatim from the narration, or empty"},
                                "turn": {"type": "string", "description": "one clause: what is now waiting, or empty"}},
                 "required": ["what", "said", "turn"]},
    },
    "required": ["sankalpa", "in_doubt", "who_stays", "narration", "perceiver_tuples", "scene_tuples", "moves", "suggested_next_action", "suggested_next_action_from_memory", "sara"],
}

_SCHEMA_FACES = json.loads(json.dumps(SCHEMA))
_SCHEMA_FACES["properties"]["boundaries"] = {
    "type": "array", "description": "rule 15: every boundary thing anyone perceived or passed across, with the scene on its other side; "
                                    "[] when nobody did",
    "items": {"type": "object", "properties": {"thing": {"type": "string"}, "also_in": {"type": "string"}}, "required": ["thing", "also_in"]}}
_SCHEMA_FACES["properties"]["perceiver_tuples"]["items"]["required"] = SCHEMA["properties"]["perceiver_tuples"]["items"]["required"] + ["closes"]  # v5.33: the field is answered, [] if none
_SCHEMA_FACES["required"] = _SCHEMA_FACES["required"] + ["boundaries"]  # optional, the second model skipped it: 2 of 6 (live, collide)

# Saṅgama (संगम, "confluence" -- Marc's name, 2026-09-27): players in one
# scene act as one round, and one pulse tells it to each of them. The rule
# is shown below the memory only in a round (the system prompt stays one
# cacheable text); the schema carries `round_narrations` only then.
ROUND_NARRATIONS = {
    "type": "array", "description": "a round (SAṄGAMA below the memory): one narration for EVERY player of the round other than the ACTOR",
    "items": {"type": "object", "properties": {
        "player": {"type": "string", "description": "the player's id"},
        "narration": {"type": "string", "description": "the same moment told to this player: second person, present tense (rule 13)"},
        "suggested_next_action": {"type": "string", "description": "one sentence this player might plausibly do next (rule 14)"}},
        "required": ["player", "narration", "suggested_next_action"]}}


def round_schema(schema: dict) -> dict:
    s = json.loads(json.dumps(schema))
    s["properties"]["round_narrations"] = ROUND_NARRATIONS
    s["required"] = s["required"] + ["round_narrations"]
    return s


# Marc, 2026-10-13 (Stonemarch 1637-1658, Jakal's spellbook: leaves stitched in, none ever declared, the book holding nothing): "Everything is a
# container. A container can hold things. If the narrative says something merged with or goes into something that should happen." With
# SMRITILILA_HOLDS=1 rule 7 says a thing that holds things is declared with them, and the prompt says what each thing the actor holds holds
# (pulse.fill_holds, shown in PRESENT). No verbs are listed: writing and binding were ruled too specific.
HOLDS_RULE = (" WHAT HOLDS THINGS IS DECLARED WITH THEM: everything is a container, and a container can hold things. When you declare something that "
              "holds things (a book with its spells, a purse with its coins), declare each thing it holds as NEW too, held by it. A thing that goes into "
              "another, or merges with it, is held by it from then on; if it is new, declare it NEW, held by what it went into.")
HOLDS_ANCHOR = "never beside it."


def holds_on() -> bool:
    return os.environ.get("SMRITILILA_HOLDS") == "1"


def holds_rules(system: str) -> str:
    """The contract's rule 7 with HOLDS_RULE after its part-and-whole lines. Said once; a prompt without the anchor is left as it was."""
    if HOLDS_RULE in system or system.count(HOLDS_ANCHOR) != 1:
        return system
    i = system.index(HOLDS_ANCHOR) + len(HOLDS_ANCHOR)
    return system[:i] + HOLDS_RULE + system[i:]


def round_lines(acting: dict[str, str], still: list[str]) -> list[str]:
    """What the Narrator is told of a round, below the memory: who else acts
    at this moment and what each attempts, who is in it and does not act."""
    lines = ["SAṄGAMA -- THIS PULSE IS A ROUND. The players in THIS MOMENT act AT THE SAME MOMENT as the ACTOR. Every ACTION in it is an "
             "attempt (rule 8), and they are resolved TOGETHER: they may help, hinder or cross one another, and the outcome is one "
             "moment, the same for all. A player who ACTS in the round is an actor of this pulse as the ACTOR is -- give them their "
             "own tuple (rule 9), move them as their own action takes them, let them say their own words, and nothing more than "
             "their action states. A player IN THE ROUND WHO DOES NOT ACT is still theirs (rule 10): never moved, never voiced, "
             "given no deed -- but their tuple is written: what they perceived. In `round_narrations` give EVERY player of the "
             "round other than the ACTOR their own narration of this moment (rule 13: second person, present tense, addressed "
             "to them): what they did, saw and heard, the others' spoken words as spoken, and the same outcome the ACTOR's "
             "narration tells. `narration` stays the ACTOR's own.",
             # a round holds the world's own turns in its scene (v5 §36), so in back-to-back rounds no one else there ever
             # had one: in the moathouse, ~30 rounds, the companions took none and acted only when ordered (Marc,
             # 2026-09-28). Replayed on the ogre round (the second model, 3 runs): Elmo shot and Calmert shielded or struck
             # 3 of 3, unasked; live, both stood by. Opponents too
             # v5 §28 (Marc, 2026-10-01): who acts besides the players is THIS MOMENT's list, each with a fortune of their own --
             # v4.35's "fortune is the players' alone" withdrawn
             "THE OTHERS HERE ACT TOO: each one listed in THIS MOMENT as able to act takes this moment as they would, from "
             "what their memory holds (their saṅkalpa) -- in a fight they fight; whoever tends, tends; whoever keeps watch, "
             "watches -- needing no one's asking; their deeds are resolved with the players' in the same moment, each under "
             "their own fortune."]
    for pid, action in acting.items():
        lines.append(f"ALSO ACTING IN THIS ROUND: {pid} -- ACTION: {action} (in THIS MOMENT)")
    if still:
        lines.append("IN THE ROUND, NOT ACTING: " + ", ".join(still))
    return lines


def moment_lines(entries: list[tuple[str, str | None, str | None]], seed: str) -> list[str]:
    """v5 §28: THIS MOMENT -- the actor, each player acting, and each one who may
    act, (id, their action or None, their fortune or None), in an order
    drawn from the moment itself (Marc, 2026-10-01: "actions listed first
    will naturally have a higher priority for the narrator so the order
    should be random") -- seeded, so a rerun keeps it; chosen by no one."""
    import hashlib
    import random
    order = sorted(entries, key=lambda e: e[0])  # from a canonical order: never the order pressed, or handed in
    random.Random(int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)).shuffle(order)
    lines = ["THIS MOMENT, IN NO ORDER OF PRECEDENCE -- every attempt and every intention below is resolved together:"]
    for who, action, word in order:
        luck = f" -- FORTUNE, IF IT HOLDS ANYTHING IN DOUBT: {word}" if word else ""
        lines.append(f"- {who} -- ACTION: {action}{luck}" if action is not None else f"- {who} -- may act (their saṅkalpa){luck}")
    return lines


def round_daiva_line(pid: str, word: str) -> str:
    """A co-actor's own fortune, for their own attempt."""
    # not "ATTEMPT": said so, every co-actor's action read as an attempt (Stonemarch 2026-09-28, the fen: readying a
    # dart, keeping pace and steadying a foot drew slips, snags and a torn sleeve in turn -- Jakal made clumsy)
    return f"FORTUNE FOR {pid}, IF THEIR ACTION HOLDS ANYTHING IN DOUBT: {word}."

NARRATOR_MAX_TOKENS = 16000  # the whole answer and whatever the model thinks first
MIN_TUPLE_TEXT = 24  # shorter than this (smritilila.text.weight: script-weighted) is a placeholder, not a memory

# The weight scale (smritilila/memory/salience.py): the Narrator chooses a word; the store keeps the number.
from smritilila.memory.salience import SALIENCE_NAMES, SALIENCE_WORDS, salience_of  # noqa: E402,F401


def use_of(value: object) -> str:
    """v5 §3: key | useful | none; anything else is none."""
    v = str(value or "").strip().lower()
    return v if v in USES else "none"


DEFAULT_VIEW = """THE DEFAULT VIEW -- what this world is taken to be unless THIS WORLD'S OWN VIEW, or a view below the memory, says otherwise:
- Perception is by the senses: a perceiver takes in what they could see, hear, smell or touch, and does what they could do. Minds are private: another's thoughts, intentions or unspoken reasons never appear in someone else's tuple.
- Most people are not watching closely; a bystander's tuple is brief and mundane.
- Across a scene's boundary only what carries reaches: low voices through a wall, a door heard below.
- Places do not perceive: a scene's record is an onlooker's with no interior. The dead, the absent and the far-off do not perceive; the living do.
- A NAME is knowledge: a perceiver uses a name only if their OWN tuples contain it, or they are told it this pulse; otherwise they refer to someone by what they can perceive.
- The world does not yield to assertion: a name the actor uses that the world does not hold is the character's own confusion, told inside the world.
- An autonomous turn is an ordinary, in-character next step, not a dramatic turn. Arrivals into a quiet scene are seldom.
- How things turn out: not always the outcome attempted, nor the likeliest, nor the one the story would prefer. An attempt may succeed, half succeed, fail, cost something, or spectacularly and destructively backfire. A question pressed on someone is answered, refused, lied about or deflected -- and said -- in the turn it is pressed; "has not yet answered" is not an outcome. Hesitation may colour the telling; it does not take the turn.
- Becoming is rare."""


def source_line(source: str | None) -> str:
    """The world's SOURCE, up front (Marc, 2026-09-24: "we should always
    include the source up front. historical or fictional, that's what
    grounds the narrator in the correct context"). Ravenmoor's view never
    said it was Ravenmoor's. Empty when a world has none."""
    if not source or not source.strip():
        return ""
    return (f"THIS WORLD'S SOURCE: {source.strip()}\nEverything below is grounded in it -- its names, places, manners, idiom, "
            "and what is and is not possible. Where the source speaks, be accurate to it; invent only where it is silent, and "
            "in its idiom.")


def own_view(darsana: str) -> str:
    """The world's own view, as every writer of the world is shown it."""
    return "THIS WORLD'S OWN VIEW -- it wins over the default wherever they differ:\n" + darsana.strip()


# Lengths, and other words, by the model that answers (Marc, 2026-09-26: "different length suggestions based on model called"; mainly for
# memories, prose too). The shared figures stand in rules 1 and 13. A provider whose model needs its own (llm/providers.py, `swaps`: the
# model-name fragment and the shared text it replaces) sends them in place of these, resolved by the client that answers (Anurupa); each
# provider carries its own reasons.
NARRATION_LENGTH = "80 to 150 words"
TUPLE_LENGTH = "Each is one to three sentences (the actor's may run to four); a bystander's is one."


def system_for(darsana: str | None = None, source: str | None = None, faces: bool = False) -> Anurupa:
    """The Narrator's system prompt for a world (v5 §7): the world's source
    first, then the contract, the default view, and the world's own view,
    which wins. Static per world, so it caches. Carries each provider's
    own swaps (llm/providers.py) as a variant; as text it is the shared prompt."""
    system = SYSTEM.replace("<<BOUNDARIES>>\n", _FACES_RULE if faces else "")
    if holds_on():
        system = holds_rules(system)
    parts = [p for p in (source_line(source),) if p] + [system, DEFAULT_VIEW]
    if darsana and darsana.strip():
        parts.append(own_view(darsana))
    text = "\n\n".join(parts)
    variants = {}
    for model, swaps in providers.prompt_swaps().items():
        # the contract's own words only: the world's view may say "80 to 150 words" of something else
        head, _, tail = text.partition(system)
        variant = system
        for old, new in swaps.items():
            variant = variant.replace(old, new)
        variants[model] = head + variant + tail
    return Anurupa(text, variants)


AUTONOMOUS_ACTION = "(none supplied: this perceiver acts on their own -- rule A)"

# v5 §38: the world takes initiative in a quiet scene -- a road's turn,
# asked of any scene near a player that has been quiet (roads retired).
SCENE_ARRIVAL_ACTION = "(the ACTOR is a SCENE that has been quiet -- rule B: has anyone come into it?)"

CHRONICLE_ACTION = (
    "(the ACTOR is THE WORLD AT LARGE -- rule C. The world at large as it was written at birth follows; the tuples above are what "
    "has moved since.)\n"
    "THE WORLD AT LARGE:\n"
)


@dataclass
class PulseOut:
    narration: str
    perceiver_tuples: list[dict]
    place_tuples: list[dict]  # the scene tuples (v5 §31); the field keeps v3's name inside the engine
    raw: dict
    suggested_next_action: str = ""
    sara: str = ""
    said: str = ""
    turn: str = ""
    suggested_next_action_from_memory: str = ""
    moves: list[dict] = field(default_factory=list)          # v5 §10: [{entity, into, from}]
    boundaries: list[dict] = field(default_factory=list)     # rule 15: [{thing, also_in}] -- read by pulse._place
    new_entities: list[dict] = field(default_factory=list)   # v5 §10: [{id, name, in, scene}]
    aliases: list[dict] = field(default_factory=list)        # v5 §16: [{entity, name}]
    becomings: list[dict] = field(default_factory=list)      # v5 §8: [{entity, key, clause, ends}]
    capabilities: list[dict] = field(default_factory=list)   # v5 §8: [{entity, capability, on}]
    round_narrations: list[dict] = field(default_factory=list)  # saṅgama: [{player, narration, suggested_next_action}]
    sankalpa: list[dict] = field(default_factory=list)  # v5 §28: [{who, intends}]


def handle_map(ctx: Context, labels_by_id: dict[str, list[str]]) -> HandleMap:
    """Readable handles at the model boundary (llm/handles.py): every id
    the context names, of any shape, is shown as a handle, and every
    handle returned is decoded before the Binder sees it."""
    shown = ctx.shown() | set(ids_in(*(t.memory for t in ctx.tuples), known=labels_by_id))  # and an id written inside a tuple's text
    return HandleMap.build([i for i in shown if i in labels_by_id], labels_by_id)


def _present(ctx: Context) -> str:
    """Who and what is present, by what holds what (the sample world,
    2026-09-30): the skiff and who is in it, and what the actor holds named
    as theirs -- so a thing let go is a move with its `from` (rule 3), and a
    thing carried is not left behind in the scene the actor leaves."""
    present = set(ctx.present)
    inside: dict[str, list[str]] = {}
    for x in present:
        c = ctx.holder.get(x)
        if c is not None and c != ctx.actor and c in present:
            inside.setdefault(c, []).append(x)
    theirs = sorted(x for x, c in ctx.holder.items() if c == ctx.actor)
    nested = {x for xs in inside.values() for x in xs}

    def show(x: str, seen: frozenset = frozenset()) -> str:
        kids = [y for y in sorted(inside.get(x, ())) if y not in seen]
        return x + (f" (holding: {', '.join(show(y, seen | {x}) for y in kids)})" if kids else "")
    text = "; ".join(show(x) for x in sorted(present) if x not in nested and x not in theirs) or "(nobody else)"
    def held(x: str) -> str:  # SMRITILILA_HOLDS: what a thing the actor holds holds, or that it is empty (a book that held nothing looked like any item)
        kids = ctx.holds.get(x)
        return x if kids is None else f"{x} (holding: {', '.join(kids)})" if kids else f"{x} (empty)"
    return text + (f"; held by the actor, {ctx.actor}: {', '.join(held(x) for x in theirs)}" if theirs else "")


def render(ctx: Context, action: str, place_ids: set[str], handles: HandleMap | None = None, thread: str = "",
           own_words: dict[int, tuple[str, str | None]] | None = None, action_source: str | None = None,
           view_text: str = "") -> str:
    """The prompt's user part. own_words (v5 §35 B): a player's own words,
    by tick, shown beside their own tuples, so a belief the Narrator once
    planted is visible as not theirs. action_source (v5 §35): "suggested"
    when the action is the page's suggestion sent unchanged."""
    own_words = own_words or {}
    lines = ["MEMORY (tuples in tick order: [id, tick, weight, use]; ids tagged [scene] are scenes):"]
    for t in ctx.tuples:
        tag = " [scene]" if t.perceiver in place_ids else ""
        line = f"[{t.perceiver}{tag}, {t.chronology}, {SALIENCE_NAMES.get(t.salience, t.salience)}, {t.use or 'none'}] {t.memory}"
        if t.perceiver == ctx.actor and t.chronology in own_words:
            said, source = own_words[t.chronology]
            if source == "autonomous":
                line += "   <<not their words: the Narrator acted for them that pulse, while they were away>>"
            else:
                src = {"suggested": "the page's suggestion, accepted",
                       # saṅgama: the round closed while they had not acted; their first suggestion was taken for them
                       "default": "not chosen by them: their first suggestion, taken when the round went on without their action"
                       }.get(source or "", "typed")
                line += f'   <<their own words that pulse ({src}): "{said}">>'
        lines.append(line)
    lines.append("")
    lines.append(f"ACTOR: {ctx.actor}")
    up = list(ctx.chain) + [a for a in ctx.above if a not in ctx.chain]
    if up:
        lines.append("ACTOR IS IN: " + " -> ".join(up))
    lines.append(f"SCENE: {ctx.place if ctx.place else 'none -- the actor is in no scene'}")
    lines.append(f"PRESENT (in the scene, by what they are in): {_present(ctx)}")
    if ctx.known:
        lines.append("KNOWN TO THE ACTOR, PLACED NOWHERE (named in pulses they were in; where each is, the world has not recorded; by declared name): "
                     + "; ".join(f'{eid} "{name}"' for eid, name in ctx.known))
    if ctx.namesakes:
        lines.append("SOMEONE OF THAT NAME IS IN THE WORLD (a name in the action or the memory above is part of theirs; if the one "
                     "meant is them, it is this id -- never a new person under the same name; if it is someone else, give them "
                     "a name of their own): " + "; ".join(f'{eid} "{name}" (the name {part})' for eid, name, part in ctx.namesakes))
    if ctx.players_named:
        lines.append("A PLAYER IS NAMED HERE, NOT PRESENT (rule 10: their deeds are theirs): " + "; ".join(f'{eid} "{name}"' for eid, name in ctx.players_named))
    if ctx.across:
        lines.append("ACROSS A BOUNDARY OF THIS SCENE (a thing that is in both this scene and another -- what is across it "
                     "is perceived as the view allows): "
                     + "; ".join(f'{face} -> {other}' + (f' ({", ".join(there)})' if there else '') for face, other, there in ctx.across))
    if ctx.elsewhere:
        lines.append("KNOWN TO THE ACTOR, ELSEWHERE (met in pulses they were in, or seen in the memory above, placed where this scene "
                     "does not reach; by declared name, and where each is -- one of them here this pulse is here only if `moves` brings "
                     "them): "
                     + "; ".join(f'{eid} "{name}", in {where}' for eid, name, where in ctx.elsewhere))
    if ctx.lately:
        lines.append("WITH YOU LATELY, NOT HERE NOW (they shared this actor's scene in the actor's last few moments; noticeable "
                     "by their absence -- see `who_stays`; never invent where they are): "
                     + "; ".join(f'{eid} "{name}"' for eid, name, _where in ctx.lately))
    for head, read, unread in ctx.reading:
        # v4.60: a chain of things held one inside the next, read in order (pages in a book, sheets of a letter)
        lines.append(f"READING {head}, ONE INSIDE THE NEXT, IN ORDER -- read already: {', '.join(read) or '(none)'}; "
                     f"NOT YET READ, in order: {', '.join(unread)} (their words are in the memory above). The actor reads "
                     "only as far as the ACTION reads -- a page turned is the first not yet read; the whole, if they read it "
                     "whole. Tell only what they read, word for word where they read it, and write a tuple for each link "
                     "they read, holding what was read; never one they did not reach.")
    if ctx.quiet:
        lines.append("PRESENT AND QUIET THIS PULSE (in the scene; only who they are is shown -- any of them may still be written for, "
                     "and may act): " + ", ".join(sorted(ctx.quiet)))
    lines.append(f"NEIGHBOURING SCENES: {', '.join(ctx.neighbours) or '(none)'}")
    if ctx.tree:
        # v4.30: the tree to reason with (rule 7) -- the scene and what holds it, from the world's top down
        # the heading names no kind (2026-10-03, the nirguna audit): what is listed is not all places -- the earth's loose
        # holdings (a latchkey, a man never met) are shown by is_place's third clause, and showing them anchors who they
        # are (replayed: left off, Harold Wheelock was minted again 3 of 3)
        lines.append("THE TREE (the scene and what holds it, from the world's top down; under each, who and what it directly holds):")
        for depth, (h, name, places) in enumerate(ctx.tree):
            mark = " (the world's top)" if depth == 0 else (" (this scene)" if h == ctx.place else "")
            lines.append("  " * depth + f'- {h} "{name}"{mark}' + (" holds: " + "; ".join(f'{x} "{n}"' for x, n in places) if places else ""))
    if ctx.last_region:
        # Stonemarch, 2026-10-10: the tree shows the actor's own branch, so a region they stood in two scenes ago, one level down from the world's top, was invisible and every
        # place made after it was filed in the world at large. Worded as the place rerun's line is, which binds; a plain fact line did not (the second model 0/3 against 3/3 here).
        region, rname, ago, rows = ctx.last_region
        lines.append(f'PLACES ALREADY IN {region} "{rname}" (the region the actor last stood in, {ago} moments ago; the actor now stands outside it -- if they are going to one of '
                     'these, use its id; a place made near here that is not one of these belongs inside the region): '
                     + "; ".join(f'{x} "{n}"' + (f" (in {up})" if up else "") for x, n, up in rows))
    for region, (name, rows) in ctx.beyond.items():
        lines.append(f"PLACES ALREADY IN {region} \"{name}\" (a place beyond the actor's own; the actor may be going to one of these "
                     "-- if so, use its id; declare a new place only if none of them is where they go): "
                     + "; ".join(f'{x} "{n}"' + (f" (in {up})" if up else "") for x, n, up in rows))
    if ctx.held:
        lines.append("HELD HERE AND IN WHAT HOLDS THE SCENE (not present; nothing perceived of them lately; by declared name): "
                     + "; ".join(f'{eid} "{name}", in {c}' for eid, name, c in ctx.held))
    if action_source == "suggested":
        lines.append(f"ACTION (the page's suggestion, accepted as offered -- a choice among the page's options, not a belief the player stated): {action}")
    else:
        lines.append(f"ACTION: {action}")
    lines.append(f"THREAD FROM MEMORY: {thread or '(none)'}")
    if view_text:
        lines.append(view_text)  # v5 §7-§8: the views that hold here, per perceiver, and what the structure allows
    text = "\n".join(lines)
    return handles.encode_text(text) if handles else text


_MINTED_TAIL = re.compile(r"_[0-9a-f]{6}$")  # pulse.py's mnemonic_id: words plus a short hex tail


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def decode_handle(value: str, handles: HandleMap) -> str:
    """A handle back to its id; one garbled by a character or two is the
    one handle within that distance -- exactly one -- or stays as it came."""
    decoded = handles.decode_id(value)
    if decoded != value or value.strip().upper().startswith("NEW") or _MINTED_TAIL.search(value):
        return decoded
    key = handle_key(value)
    if len(key) < 8:
        return value  # two edits turn a short word into another (west -> east); nearest_id's floor too
    close = [h for h in handles.handle_to_id if abs(len(h) - len(key)) <= 2 and _edit_distance(h, key) <= 2]
    if len(close) == 1:
        log_event("smrti.handle_near", pulse_id=current_pulse_id(), wrote=value, handle=close[0])
        return handles.handle_to_id[close[0]]
    return value


def decode_ids(out: dict, handles: HandleMap) -> None:
    """Every id-shaped field of append_tuples, in place."""
    dec = lambda v: decode_handle(v, handles) if isinstance(v, str) and v.strip() else v
    for r in out.get("perceiver_tuples") or []:
        if isinstance(r, dict):
            r["perceiver"] = dec(r.get("perceiver"))
    for r in out.get("scene_tuples") or []:
        if isinstance(r, dict):
            for k in ("scene", "said_by"):
                r[k] = dec(r.get(k))
            if isinstance(r.get("present"), list):
                r["present"] = [dec(x) for x in r["present"]]
    for r in out.get("moves") or []:
        if isinstance(r, dict):
            for k in ("entity", "into"):
                r[k] = dec(r.get(k))
            if r.get("from"):
                r["from"] = dec(r.get("from"))
    for r in out.get("boundaries") or []:
        if isinstance(r, dict):
            for k in ("thing", "also_in"):
                r[k] = dec(r.get(k))
    for r in out.get("new_entities") or []:
        if isinstance(r, dict):
            r["in"] = dec(r.get("in"))
    for key in ("aliases", "becomings", "capabilities"):
        for r in out.get(key) or []:
            if isinstance(r, dict):
                r["entity"] = dec(r.get("entity"))
    for r in out.get("round_narrations") or []:
        if isinstance(r, dict):
            r["player"] = dec(r.get("player"))
    for r in out.get("sankalpa") or []:
        if isinstance(r, dict):
            r["who"] = dec(r.get("who"))
    for r in out.get("in_doubt") or []:
        if isinstance(r, dict):
            r["player"] = dec(r.get("player"))
    for r in out.get("who_stays") or []:
        if isinstance(r, dict):
            r["who"] = dec(r.get("who"))


def plain_texts(out: dict, handles: HandleMap) -> None:
    """Every prose field of append_tuples, in place: a handle the model
    wrote into what is read or remembered becomes plain words."""
    leaks = 0

    def p(v: object) -> object:
        nonlocal leaks
        if not isinstance(v, str):
            return v
        text, n = handles.plain(v)
        leaks += n
        return text

    for key in ("narration", "suggested_next_action", "suggested_next_action_from_memory"):
        out[key] = p(out.get(key))
    sara = out.get("sara")
    if isinstance(sara, dict):
        for k in ("what", "said", "turn"):
            sara[k] = p(sara.get(k))
    else:
        out["sara"] = p(sara)
    for key, fields in (("perceiver_tuples", ("text",)), ("scene_tuples", ("text", "said")),
                        ("round_narrations", ("narration", "suggested_next_action")), ("who_stays", ("noticed",)),
                        ("sankalpa", ("intends",))):
        for r in out.get(key) or []:
            if isinstance(r, dict):
                for k in fields:
                    if k in r:
                        r[k] = p(r[k])
    if leaks:
        log_event("smrti.handle_leak", pulse_id=current_pulse_id(), count=leaks)


LIST_FIELDS = ("perceiver_tuples", "scene_tuples")


def unfold(out: dict) -> None:
    """A list-typed field that arrived as a string is parsed in place
    (Vadhu ticks 106, 116): the leading list is read with raw_decode;
    what follows is the rest of the object, closed if cut short; a key
    missing or empty is filled. Logged as smrti.shape_recovered."""
    dec = json.JSONDecoder()
    for key in LIST_FIELDS + ("moves", "new_entities", "aliases", "becomings", "capabilities", "boundaries"):
        v = out.get(key)
        if not isinstance(v, str):
            continue
        head, rest = None, v.strip()
        if rest.startswith("["):
            try:
                head, i = dec.raw_decode(rest)
                rest = rest[i:].lstrip(" ,")
            except ValueError:
                whole = rest
                head, rest = _closed_list(rest), ""
                if head is None:
                    # one broken row must not take the good ones with it (the
                    # crossing 515: a `"perceiver": "x, ": "text_placeholder"`
                    # row cost Marren, Ilo, Bren and Sela their tuples)
                    head = _salvage_objects(whole)
                    if head:
                        log_event("smrti.rows_salvaged", field=key, kept=len(head))
        if head is None:
            out[key] = []
            log_event("smrti.malformed_tuple", field=key, dropped=1)
            continue
        out[key] = head
        filled = [key]
        tail = _object_tail(rest) if rest else {}
        for k, x in tail.items():
            if k != key and (k not in out or out.get(k) in (None, "", [])):
                out[k] = x
                filled.append(k)
        log_event("smrti.shape_recovered", field=key, filled=filled)


def _salvage_objects(text: str) -> list | None:
    """Every object in a malformed list that parses on its own, in order."""
    dec = json.JSONDecoder()
    got, i = [], text.find("{")
    while i >= 0:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            got.append(obj)
        i = text.find("{", end)
    return got or None


def _closed_list(text: str) -> list | None:
    for closer in ('"}]', "}]", "]", '"]', '"}}]', "}}]"):
        try:
            got = json.loads(text + closer)
        except ValueError:
            continue
        return got[:-1] if isinstance(got, list) else None
    return None


def _object_tail(rest: str) -> dict:
    for closer in ("", '"', '"}', '"]', '"}]', '"]}', '"}]}', "}", "]", "}]", "]}", "}]}"):
        try:
            got = json.loads("{" + rest + closer + "}")
        except ValueError:
            continue
        return got if isinstance(got, dict) else {}
    return {}


def _plain(text: str) -> str:
    return re.sub(rf"[\s{re.escape(QUOTES)}]+", " ", text).strip().lower()  # v5 §7: every script's quote marks


def unpack_sara(value: object, narration: str) -> tuple[str, str, str]:
    """The sara's three fields; a bare string (the earlier shape) is `what`."""
    if isinstance(value, str):
        return value.strip(), "", ""
    if not isinstance(value, dict):
        return "", "", ""
    what = str(value.get("what", "") or "").strip()
    turn = str(value.get("turn", "") or "").strip()
    return what, verified_said(value.get("said"), narration, "smrti.sara_said_dropped"), turn


# A dialogue tag inside a spoken line: a closing quote after punctuation,
# a short run of prose, an opening quote -- `'No stone,' he says slowly,
# 'means...'` (Ravenmoor 342). v5 §32: the line keeps only the words said.
# An apostrophe inside a word (he'd, don't) is never the opening quote: the
# crossing, 623 -- a nested quotation, "...at the mill again.' I said it hard,
# so he'd believe me", lost 21 words between the closing quote and "he'd".
_WORD_QUOTES = "'‘"
_OPENERS = "".join(c for c in OPEN_QUOTES if c not in _WORD_QUOTES)
_TAG_INSIDE = re.compile(rf"(?<=[,.!?;—\-，。！？、])[{re.escape(CLOSE_QUOTES)}]\s*[^{re.escape(QUOTES)}]{{1,60}}?[,，]?\s*"
                         rf"(?:[{re.escape(_OPENERS)}]|(?<!\w)[{re.escape(_WORD_QUOTES)}])")


def strip_tag(said: str) -> str:
    stripped = _TAG_INSIDE.sub(" ", said)
    if stripped != said:
        log_event("smrti.quote_tag_stripped", pulse_id=current_pulse_id(), was=said[:80])
    return re.sub(r"\s{2,}", " ", stripped).strip()


def verified_said(value: object, narration: str, dropped_event: str) -> str:
    """A quoted line is the narration's own words or nothing, with any
    dialogue tag inside it removed (v5 §31-§32)."""
    said = strip_tag(str(value or "").strip().strip(QUOTES + " "))
    if not said:
        return ""
    plain_narration = _plain(narration)
    if _plain(said) in plain_narration:
        return said
    if _words_in_order(_plain(said).split(), plain_narration.split()):
        return said
    if _sentences_in_order(said, plain_narration):
        return said
    log_event(dropped_event, pulse_id=current_pulse_id(), said=said[:80])
    return ""


TAG_GAP = 6  # words of prose a dialogue tag may put between two words of one line


def _sentences_in_order(said: str, plain_narration: str) -> bool:
    """Every sentence of the line verbatim in the narration, in order --
    however long the tag between them (the crossing 496: a seven-word tag,
    "he says, plain, no dressing on it", split a line quoted exactly).
    Indirect speech never passes: "he says he needs" is not "I need"."""
    parts = [_plain(x).strip(" .,;:!?-。！？、，") for x in sentences(said)]
    parts = [x for x in parts if len(x.split()) >= 2 or weight(x) >= 6]
    if len(parts) < 2:
        return False
    pos = 0
    for x in parts:
        i = plain_narration.find(x, pos)
        if i < 0:
            return False
        pos = i + len(x)
    return True


def _words_in_order(said: list[str], prose: list[str], max_gap: int = TAG_GAP) -> bool:
    strip = lambda w: w.strip(".,;:!?-—–()")
    said = [strip(w) for w in said if strip(w)]
    prose = [strip(w) for w in prose]
    if not said:
        return False
    for start in (k for k, w in enumerate(prose) if w == said[0]):
        pos = start
        for w in said[1:]:
            window = prose[pos + 1: pos + 2 + max_gap]
            if w not in window:
                break
            pos = pos + 1 + window.index(w)
        else:
            return True
    return False


def narrate(client: LLMClient, ctx: Context, action: str, place_ids: set[str], handles: HandleMap | None = None, thread: str = "",
            own_words: dict[int, tuple[str, str | None]] | None = None, action_source: str | None = None,
            darsana: str | None = None, view_text: str = "", source: str | None = None, faces: bool = False,
            in_round: bool = False) -> PulseOut:
    schema = _SCHEMA_FACES if faces else SCHEMA
    out = client.complete_with_tool(
        system_for(darsana, source, faces),
        render(ctx, action, place_ids, handles, thread, own_words, action_source, view_text),
        tool_name="append_tuples",
        tool_description="The pulse's narration and what to append to memory.",
        input_schema=round_schema(schema) if in_round else schema,
        # thinking shares the output budget (2026-09-23): at 8192 the the second model
        # JSON path truncated 3 answers in 36, and the larger model thinks ~440 tokens
        max_tokens=NARRATOR_MAX_TOKENS,
    )
    unfold(out)
    if handles is not None and handles.id_to_handle:
        decode_ids(out, handles)
        plain_texts(out, handles)

    def rows(key: str, required: tuple[str, ...], min_text: bool = True) -> list[dict]:
        good, bad = [], 0
        for r in out.get(key) or []:
            if isinstance(r, dict) and all(isinstance(r.get(k), (str, int, list, bool)) for k in required) and (not min_text or weight(str(r.get("text", ""))) >= MIN_TUPLE_TEXT):
                good.append(r)
            else:
                bad += 1
        if bad:
            log_event("smrti.malformed_tuple", field=key, dropped=bad)
        return good

    narration = str(out.get("narration", "") or "")
    sara_what, sara_said, sara_turn = unpack_sara(out.get("sara"), narration)
    if not narration.strip():
        # the crossing 480: the model left the required narration out and
        # the player read nothing on the pulse Bren was found. The page's
        # own summary is the same voice; nothing reaches memory from here
        sara = out.get("sara")
        narration = str((sara.get("what") if isinstance(sara, dict) else sara) or "").strip()
        log_event("smrti.narration_missing", pulse_id=current_pulse_id(), fallback="sara" if narration else "none")
    scene_tuples = rows("scene_tuples", ("scene", "text"))
    for p in scene_tuples:
        p["place"] = p.get("scene", "")  # the engine keeps v3's field name
        p["said"] = verified_said(p.get("said"), narration, "smrti.place_said_dropped")
    return PulseOut(
        narration=narration,
        perceiver_tuples=rows("perceiver_tuples", ("perceiver", "text")),
        place_tuples=scene_tuples,
        raw=out,
        suggested_next_action=str(out.get("suggested_next_action", "") or ""),
        sara=sara_what, said=sara_said, turn=sara_turn,
        suggested_next_action_from_memory=str(out.get("suggested_next_action_from_memory", "") or ""),
        moves=[r for r in (out.get("moves") or []) if isinstance(r, dict) and str(r.get("entity", "") or "").strip()],
        boundaries=[r for r in (out.get("boundaries") or []) if isinstance(r, dict)],
        new_entities=[r for r in (out.get("new_entities") or []) if isinstance(r, dict) and str(r.get("id", "") or "").strip()],
        aliases=[r for r in (out.get("aliases") or []) if isinstance(r, dict) and str(r.get("entity", "") or "").strip() and str(r.get("name", "") or "").strip()],
        becomings=[r for r in (out.get("becomings") or []) if isinstance(r, dict) and str(r.get("entity", "") or "").strip() and str(r.get("key", "") or "").strip()],
        capabilities=[r for r in (out.get("capabilities") or []) if isinstance(r, dict) and str(r.get("entity", "") or "").strip()
                      and r.get("capability") in ("perceives", "reaches")],
        round_narrations=[r for r in (out.get("round_narrations") or []) if isinstance(r, dict)
                          and str(r.get("player", "") or "").strip() and str(r.get("narration", "") or "").strip()],
        sankalpa=[r for r in (out.get("sankalpa") or []) if isinstance(r, dict) and str(r.get("who", "") or "").strip()],
    )
