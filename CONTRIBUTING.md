# Contributing to Smritilila

Smritilila is licensed under the Apache License 2.0 (`LICENSE`), and its brief and the two stories in `stories/` under CC BY 4.0 (see `NOTICE.md`).

## Sign off your commits

Every contribution is made under the **Developer Certificate of Origin** (`DCO`): a statement that you wrote the change, or have the right to
submit it, under the license of the file you are changing. You make it by signing off each commit:

```bash
git commit -s -m "what you changed"
```

That adds a line to the commit message, with your real name and email as git has them:

```
Signed-off-by: Your Name <you@example.com>
```

A commit without that line cannot be accepted. To add it to the last commit: `git commit --amend -s --no-edit`.

A contribution to the code is licensed under Apache-2.0, as section 5 of that license says; a contribution to the brief or the stories, under CC BY 4.0.
You keep the copyright in what you write.

## Before you send a change

- Run the tests (`pytest`). The live-database tests need a database of their own, named in `TEST_DATABASE_URL`: they empty every table they use, so
  never point it at a world you care about.
- Read the dated comments near the change: the code keeps its reasons in them. `Smritilila in Brief.md` says what the engine is for.
- Keep a change to one thing, and say in its message what you saw and what it does about it.

## Reporting a problem

Bug reports and code questions are not wanted. A fix may be sent as a contribution (above). Questions about the ontology are welcome: marc7sullivan@icloud.com.
