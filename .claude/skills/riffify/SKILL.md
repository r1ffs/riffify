---
name: riffify
description: Turn a video file into an MST3K/RiffTrax-style riffed version with riffify.py - checks setup, inspects the video, helps write a style note, estimates cost, runs the long job in the background and reports the results. Use when the user wants to riff a movie or video, re-render a riff track, or make the YouTube/MP3 versions.
---

# Running riffify

riffify.py lives at the root of this repo. It writes riff jokes with Claude, voices three robot
hecklers (OTTO, GEARALD, CHIP) with OpenAI or ElevenLabs text-to-speech, and renders the result.
Runs take a long time (a feature film: ~1 hour of writing, then rendering) and cost real money, so
work through these steps with the user rather than firing off the command.

## 1. Check the setup (once)

- Python 3.11+, `ffmpeg` and `ffprobe` on PATH, and `pip install -r requirements.txt` done.
- `ANTHROPIC_API_KEY` set (required). For voices: `OPENAI_API_KEY` (default engine) or
  `ELEVENLABS_API_KEY` (`--tts elevenlabs`). Check that they are set - never print their values.
- Personal settings (each robot's voices, usual options) live in `config.toml` next to riffify.py,
  copied from `config.example.toml`. Read it if present so you know their defaults; offer to create
  it when they want the same voices/options every time. Command-line flags override it.
- Worn-out jokes to avoid live in `avoid.txt` (copied from `avoid.example.txt`, read automatically).
  When the user complains about a joke the robots keep making, offer to add a line to it.
- If the user has an NVIDIA GPU, transcription is ~20x faster with
  `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"`. Only matters when the video has no
  English subtitles.

## 2. Look at the video

Run `ffprobe -v error -show_entries "format=duration:stream=index,codec_type,codec_name:stream_tags=language,title" -of compact "<video>"`
and tell the user:
- the length (over 25 minutes runs in ~20-minute blocks automatically),
- which audio track will be used (the first English one; `--audio-track` overrides),
- whether there are English text subtitles (used for the dialogue - no transcription needed).

## 3. Agree the options

Ask (one short message, with sensible defaults):
- `--title "Movie (Year)"` - helps Claude know the film.
- `--quality budget|balanced|max` - roughly $2-3 / $10-16 / $100 per hour of video. `balanced` is the
  recommended default; fast-cutting action films cost more.
- `--tts openai|elevenlabs` and whether they want `--youtube` and/or `--mp3`.
- A **style note**. Offer to draft one: start from "Riff on the movie itself first: what the
  characters say and do, their choices and motives, the plot's twists and logic, and what's happening
  on screen." then name the film's best comedy targets and anything sensitive to keep off-limits
  ("keep riffing through those scenes at the usual pace, but aim the jokes at something else on
  screen, never at the sensitive subject itself" - just "riff around them" can leave long silent
  gaps that the fill pass then crams full). Show the full draft, let the user edit it, then save it
  to `styles/<movie>.txt` and pass `--style @styles/<movie>.txt`. See `styles/README.md`.
- **Background notes** - only bring these up when the robots would otherwise be lost: a sequel or
  later entry in a series, a remake or spin-off that plays off the original, or a film that leans on
  lore, history or a famous source the audience is expected to know. Offer to write a short recap
  (one or two pages), save it to `background/<movie>.md`, let the user check it, and pass
  `--background background/<movie>.md`. Keep it factual and point out the comedy targets (recurring
  characters, silly lore, plot holes, what fans mock), but write no jokes, because the robots tend to
  reuse ready-made jokes word for word. Never summarize the plot of the film being riffed: that's
  spoilers. Every writing call gets the notes, so shorter is cheaper.

## 4. Run it in the background

The run outlasts normal command timeouts, so start it detached and log to a file, e.g.
`python riffify.py "<video>" --title "..." --quality balanced --style @styles/<movie>.txt --youtube --mp3 > riffify_run.log 2>&1`
(on Windows PowerShell use `Start-Process ... -RedirectStandardOutput ... -RedirectStandardError ...`).

Check the log every several minutes and keep the user posted (each block prints its cost):
- **End credits:** the log prints "end credits from H:MM:SS". For dark films, look at a few frames
  around that time (`ffmpeg -ss <t> -i "<video>" -frames:v 1 frame.png`) to confirm it is really the
  credits and not a dark final scene; if it's wrong, stop the run before the writing starts and rerun
  with `--credits H:MM:SS` (or `--credits off`).
- If a run fails, rerunning the same command resumes from the saved blocks/sections.

## 5. Report

List the output files (`<name>_riffed.mp4`, `_riffed_youtube.mp4`, `_riffed_riffs.mp3`), the
Claude cost from the log, and riffs dropped. Outputs are never overwritten - reruns get `_v2`, `_v3`.

Re-rendering without new Claude cost: `--riffs "<name>_riffed.riffs.json"` (e.g. to try other
voices with `--voice`/`--el-voice`, change `--riff-gain`, or add `--youtube`).

## Rules

- Only riff videos the user has the right to use. The YouTube version carries no movie audio or
  subtitles, but that is no guarantee against copyright claims - say so if they plan to post it.
- Don't edit riffify.py to change settings; use command-line options for one run, or `config.toml`
  for lasting ones.
