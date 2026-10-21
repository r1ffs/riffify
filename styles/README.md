# Style notes

A style note is the "producer's direction" sent to every Claude request for a run. Pass one with
`--style "..."` or, easier for longer notes, put it in a text file and use `--style @styles/my_movie.txt`.
A style note **replaces** the built-in default, so start it with the film-first line if you want to keep it.

What works well:

- **Name the film's comedy targets** - the characters, running situations and set pieces the hecklers
  should go after (the more specific, the better the jokes).
- **Say what's fair game** - fictional groups, gore, the villain, the genre's clichés.
- **Say what's off-limits for this film** - sensitive real events, a character's disability, abuse, etc.
  ("keep riffing through those scenes at the usual pace, but aim the jokes at something else on screen,
  never at the sensitive subject itself"). Saying only "riff around them" can leave long silent gaps.

The built-in rules always apply on top: no jokes at real-world groups (race, ethnicity, religion,
sexuality, disability), no repeating slurs from the dialogue, PG-13.

Your own notes in this folder are git-ignored; only the `example_*.txt` files are part of the repo.
