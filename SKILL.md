---
name: compose-video
description: Turns excerpts and slide content from internal training recordings into the storyboard of a short narrated video lesson with newly designed slides - what the voice says in each scene and what the screen shows.
version: 0.2.0
---

# Compose video lesson

You write the storyboard for a short narrated video lesson for employees, using only
excerpts from internal training recordings. Transcript excerpts are labelled [S1], [S2] ...
The content of the original slides is labelled [P1], [P2] ...

A voice reads the `narration` of each scene while the screen shows its `visual`.
The voice explains; the screen shows only the key points.

## Design new slides

The original slides are source material, not something to show. Take what they and the
speaker say, and redesign it so it teaches better:
- choose the visual that fits the content's shape: a process -> `steps`, two or three options
  or before/after -> `comparison`, several items with the same attributes -> `table`, a key
  term -> `concept`, a few separate points -> `bullets`
- regroup, reorder and simplify; combine what several slides say about one idea
- never copy a slide's layout or its sentences; on-screen text is your own short wording
- never read the screen aloud: the narration explains and gives the why, the screen shows
  the gist

## Rules

1. Use only facts from the excerpts and slides. Never add outside knowledge, numbers or names.
   If they don't cover part of the topic, leave that part out.
2. Narration is spoken English: short sentences, active voice, "you" for the learner, no
   filler, no meeting logistics, no speaker names ("the trainer" if needed).
3. Each scene's narration is 25-70 words. Hit the total word target within 15%.
4. Match the learner level:
   - beginner: define every term, explain why it matters, one simple example
   - intermediate: skip common definitions, focus on how and when
   - expert: skip basics, focus on details, exceptions and trade-offs
5. List in `sources` the S-labels each scene's narration is based on. Only use labels that exist.

## Visuals (one per scene)

- `title`: first scene only. `heading` = the lesson title. Narration: what the learner will get.
- `bullets`: `heading` plus 2-5 `items`, each at most 8 words, no full stops.
- `steps`: a process or sequence. `heading` plus 3-5 `items` in order, each at most 6 words.
- `comparison`: `heading` plus 2-3 `columns`, each with a `heading` (1-4 words) and 2-4 `items`
  (at most 6 words each).
- `table`: `heading`, `table_header` (2-4 column names) and `table_rows` (2-6 rows, one short
  cell per column, at most 5 words per cell).
- `concept`: `heading` = the term; `items` = [its definition in at most 20 words, optionally a
  short example].
- `slide`: only if the prompt offers "SLIDES YOU MAY SHOW"; put the P-label in `slide`.
- The last scene is `bullets` with heading "Key takeaways" and 3 items.

Mix the visuals: never the same visual three times in a row.

## Fields

- title: specific to the topic, not generic
- goal: one sentence on what the learner can do after watching
- scenes: in teaching order (basics first)
