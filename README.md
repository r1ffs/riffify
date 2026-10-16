# riffify

**Turn any video into a riffed version of itself** — three wisecracking robot hecklers sit in the front
row, silhouetted against the screen, and crack jokes about the movie as it plays. Think *Mystery Science
Theater 3000* or *RiffTrax*, written by AI.

```
python riffify.py "Night of the Living Dead (1968).mp4" --title "Night of the Living Dead (1968)"
```

…and out comes `Night of the Living Dead (1968)_riffed.mp4`: the original film with a riff track mixed in,
the movie's audio dipping politely whenever a robot talks, and the speaking robot bobbing along with its
lines.

## Meet the hecklers

| Seat | Robot | Personality |
|---|---|---|
| Left | **GEARALD** — box head, single antenna | Pompous and theatrical; bursts into song or Shakespeare, pretends to be cultured |
| Middle | **OTTO** — retro TV-set head with rabbit ears | The dry, deadpan host; sarcastic observations and mock sincerity |
| Right | **CHIP** — satellite-dish head | Snarky wiseguy; cheap shots, fake character voices, groan-worthy puns |

## What you get

- **The riffed movie** (`_riffed.mp4`) — original video + riffs, robot silhouettes along the bottom of the
  picture (they sit on the picture itself in letterboxed/pillarboxed films), every original subtitle track
  carried over.
- **Optional YouTube version** (`--youtube`, `_riffed_youtube.mp4`) — a companion track for posting: a
  movie-theater scene with the robots, a big *MOVIE TIME* clock and a countdown so viewers can sync their
  own copy of the film, plus a tiny, heavily blurred, silent glimpse of the film at the start and every
  10 minutes to check sync. **No movie audio and no subtitles** — only the riffs.
- **Optional MP3** (`--mp3`, `_riffed_riffs.mp3`) — just the riffs, timed to start with the film.

## How it works

1. **Dialogue** — English subtitles are used if the video has them (embedded text track or a `.srt`/`.vtt`
   next to it); otherwise the audio is transcribed with Whisper (on an NVIDIA GPU if available, else CPU).
2. **Pictures** — every cut is detected and contact sheets of the film (~1 frame every 3 seconds, labelled
   with shot number and time) are built for Claude to look at.
3. **Writing** — Claude writes the jokes in two passes, like a TV writers' room:
   - a **writers' room** brainstorms ~3x as many riffs as needed, in parallel 4-minute sections;
   - a **head writer** keeps only the best, punches them up, builds running gags and callbacks, removes
     repeated joke templates and checks the timing; a **fill pass** covers any dead stretch over 20 s.
   - Long videos (over 25 minutes) are written in ~20-minute **blocks**, in order: each block's writers see
     the previous block's riffs, and its head writer sees everything so far plus a **gag sheet** (running
     gags and jokes already used) — so callbacks work across the whole film.
4. **Timing rules** — a riff never starts before the line it answers has finished, visual jokes wait for the
   image to register, riffs prefer gaps in the dialogue, and back-and-forth exchanges are kept together.
5. **Voices** — OpenAI text-to-speech (default) or ElevenLabs, each robot with its own voice and acting
   direction; every line is leveled to the same loudness.
6. **Mix and render** — ffmpeg ducks the movie under the riffs, draws and animates the silhouettes, and
   encodes the outputs (on an NVIDIA GPU when possible).

End credits are detected automatically and not riffed (it keeps going 90 seconds into them), so you don't
pay to riff the credits.

## Requirements

- **Python 3.11+**
- **ffmpeg** (with ffprobe) on your PATH — Windows: `winget install Gyan.FFmpeg`, macOS: `brew install ffmpeg`,
  Linux: your package manager.
- **An Anthropic API key** (writes the jokes) — [console.anthropic.com](https://console.anthropic.com)
- **A voice API key**, one of:
  - **OpenAI** (default; cheap — a few cents per film) — [platform.openai.com](https://platform.openai.com)
  - **ElevenLabs** (`--tts elevenlabs`; more expressive) — [elevenlabs.io](https://elevenlabs.io). The free
    plan's 10,000 characters a month covers ~20 minutes of riffs; a feature film uses ~40,000. Note: the free
    plan can only use ElevenLabs' stock voices through the API (the defaults here are stock voices).

## Install

```bash
git clone https://github.com/r1ffs/riffify.git
cd riffify
pip install -r requirements.txt
```

Set your keys as environment variables:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."            # or ELEVENLABS_API_KEY="sk_..."
```

```powershell
# Windows (saved for future terminals)
setx ANTHROPIC_API_KEY "sk-ant-..."
setx OPENAI_API_KEY "sk-..."
```

**Optional — your own settings:** copy `config.example.toml` to `config.toml` and edit it to set each robot's
voices and the options you always use (quality, voice engine, `--youtube`, volume...). `config.toml` is
git-ignored, so your settings never end up in the repo and pulling updates never overwrites them.
Command-line flags still override it.

**Optional — much faster transcription on an NVIDIA GPU** (only needed for videos without English subtitles):

```bash
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
```

GPU video *encoding* (NVENC) is used automatically when your NVIDIA driver supports it (ffmpeg 7+/8+ builds
need a recent driver); otherwise it falls back to the CPU.

## Usage

```bash
# basic
python riffify.py movie.mp4 --title "Movie Title (1959)"

# a full movie, with the YouTube and MP3 versions, ElevenLabs voices and a style note
python riffify.py movie.mkv --title "Movie Title (1959)" --tts elevenlabs --youtube --mp3 --style @styles/my_movie.txt

# cheaper
python riffify.py movie.mp4 --quality budget --batch

# re-render a finished riff track (new voices, volume, YouTube version...) without paying Claude again
python riffify.py movie.mp4 --riffs "movie_riffed.riffs.json" --tts elevenlabs --youtube
```

Outputs are written next to the video and **never overwritten** — a second run produces `_v2`, `_v3`, …
If a run fails part-way, running the same command again resumes from the saved sections/blocks.

### Run it with Claude Code (optional)

This repo includes a [Claude Code](https://claude.com/claude-code) skill (`.claude/skills/riffify/SKILL.md`)
that walks through a run with you: it checks your setup, inspects the video (length, audio tracks,
subtitles), helps you write a style note, estimates the cost, runs the job in the background, checks the
detected end credits, and reports the results.

```bash
cd riffify
claude
> riff "Night of the Living Dead (1968).mp4" for me
```

To use it from any folder, copy `.claude/skills/riffify/` into `~/.claude/skills/` (and keep `riffify.py`
somewhere the skill can find it — tell Claude where).

## Quality and cost

| `--quality` | Writers' room | Head writer | Approx. Claude cost per hour of video |
|---|---|---|---|
| `budget` | Claude Sonnet 5.5 (medium effort) | Claude Sonnet 5.5 (medium) | ~$2–3 |
| `balanced` (default) | Claude Sonnet 5.5 (high) | Claude Opus 5.5 (high) | ~$10–16 |
| `max` | Claude Fable 5.1 (max) | Claude Fable 5.1 (max) | ~$100 |

Fast-cutting action films cost more than slow ones (more shots to look at). Every Claude call prints its
cost, and each run prints a total. `--batch` halves the cost using the Message Batches API, at the price of
waiting (minutes to hours). Mix and match with `--writer-model`, `--head-model`, `--writer-effort`,
`--head-effort`.

## Making it funnier

### Style notes

A style note tells the writers what to go after in *this* film. Write one per movie (see
[`styles/README.md`](styles/README.md) and the examples there) and pass it with `--style @styles/movie.txt`.
Good notes name the film's comedy targets ("the villain's ridiculous cape, the scientist who explains
everything twice") and anything that should stay off-limits for this film.

The default note (used when you don't pass `--style`) is:

> Riff on the movie itself first: what the characters say and do, their choices and motives, the plot's
> twists and logic, and what's happening on screen. Jokes about the acting, dialogue writing, costumes,
> sets, effects and filmmaking are welcome seasoning, not the main course.

Built-in rules always apply: fictional groups in the story are fair game, but no jokes at real-world groups
(race, ethnicity, religion, sexuality, disability), no repeating slurs from the dialogue, PG-13.

### The avoid list

`avoid.txt` lists jokes and themes that have been used too much — one per line. Copy `avoid.example.txt` to
`avoid.txt` (it's git-ignored, like `config.toml`) and add to it as you notice the robots leaning on the
same bits. It's read automatically on every run when it's next to `riffify.py`; without it, nothing is
avoided.

## Voices

**OpenAI** (default): OTTO = `ash`, GEARALD = `echo`, CHIP = `verse`, each with acting direction plus a
per-line delivery note from the writers.

**ElevenLabs** (`--tts elevenlabs`): defaults are stock voices every account has — OTTO = Chris,
GEARALD = Brian, CHIP = Callum. Delivery notes become ElevenLabs v3 `[audio tags]`.

To change voices for good, set them in your `config.toml` under `[voices.openai]` / `[voices.elevenlabs]`.
To try one for a single run: `--voice GEARALD=onyx` or `--el-voice OTTO=MyVoice`.

The best results come from designing your own voices with ElevenLabs **Voice Design** (paid plans can use
designed and library voices through the API). Starting prompts:

- **OTTO:** *An American man in his mid-thirties with a relaxed, mid-range voice and a bone-dry, deadpan sense
  of humor, like a witty everyman muttering wisecracks to his friends during a bad movie. General American
  accent. Understated and conversational, never shouty; sarcasm delivered with mock sincerity and perfect
  comic timing. Clean, studio-quality recording, close mic, no background noise.*
- **GEARALD:** *A middle-aged American man with a rich, resonant baritone and the grand, self-important delivery
  of a vain stage actor who believes he is far more cultured than he is. General American accent, not
  British, polished with an affected, over-enunciated theatrical precision. Big dynamic range: booming
  melodramatic declarations, hushed dramatic pauses, an occasional mock-operatic sing-song. Clean,
  studio-quality recording, close mic, no background noise.*
- **CHIP:** *An American man in his late thirties with a bright, slightly nasal, higher-pitched voice and the
  quick, smug energy of a wisecracking heckler who can't help himself. Fast, punchy delivery with snappy
  comic rhythm; bounces between fake innocence and gleeful mockery. Clean, studio-quality recording, close
  mic, no background noise.*

Save them on your ElevenLabs account, then put their names in your `config.toml`:

```toml
[voices.elevenlabs]
OTTO = "My Otto"
GEARALD = "My Gearald"
CHIP = "My Chip"
```

## Options

Run `python riffify.py --help` for the full list. The most useful:

| Option | What it does |
|---|---|
| `--title "..."` | What the video is (helps the writers) |
| `--quality budget\|balanced\|max` | Model/effort preset (see above) |
| `--style "..."` / `--style @file.txt` | Style note for this film |
| `--avoid FILE\|off` | Jokes/themes to avoid (default `avoid.txt`, if present) |
| `--density 12` | Average seconds between riffs |
| `--tts openai\|elevenlabs` | Voice engine |
| `--voice SPEAKER=VOICE`, `--el-voice SPEAKER=NAME` | Swap a robot's voice for this run |
| `--riff-gain 1.5`, `--duck 6` | Riff volume, and how much the movie dips under them |
| `--visual-delay 0.3` | How long visual jokes wait after the moment they're about |
| `--youtube`, `--mp3` | Also make the YouTube sync version / riffs-only MP3 |
| `--credits auto\|off\|H:MM:SS`, `--credits-riff 90` | End-credits handling |
| `--audio-track eng\|INDEX` | Which audio track is the movie's (default: first English) |
| `--subtitles file.srt`, `--no-subtitles` | Dialogue source |
| `--riffs file.riffs.json` | Re-render a saved riff track without calling Claude |
| `--batch` | Half-price Claude calls (slower) |
| `--encoder auto\|gpu\|cpu` | Video encoder (auto: NVIDIA GPU if it works) |

## Troubleshooting

- **"GPU transcription unavailable"** — install the CUDA libraries above, or ignore it (CPU works, slower).
- **"GPU encoder unavailable … driver"** — update your NVIDIA driver, or ignore it (CPU encoding works).
- **ElevenLabs 402 "Free users cannot use library voices via the API"** — use stock voices or a paid plan.
- **ElevenLabs 429** — the free plan allows 2 requests at a time; riffify already stays within that.
- **Credits detected in the wrong place** (e.g. a dark final scene) — rerun with `--credits H:MM:SS`.
- **Wrong language audio** — pick the track with `--audio-track eng` or by index (see `ffprobe`).

## Legal and fair use

riffify is a tool; **you are responsible for having the right to use the videos you process and for what you
publish**. Public-domain films (e.g. from the [Internet Archive](https://archive.org/details/feature_films)
and the [Library of Congress](https://www.loc.gov/film-and-videos/)) are a great place to start. The YouTube
version contains no movie audio or subtitles, which avoids the most common copyright-detection triggers, but
that is not a guarantee and not legal advice.

riffify is an independent project and is not affiliated with or endorsed by *Mystery Science Theater 3000*,
*RiffTrax*, or their owners; those names are trademarks of their respective owners.

## License

[MIT](LICENSE)
