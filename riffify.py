#!/usr/bin/env python3
"""riffify.py - turn any video into a RiffTrax / MST3K-style riffed version of itself.

Pipeline
  1. Dialogue with timestamps: English subtitles if the video has them (sidecar .srt/.vtt or an
     embedded text track), else faster-whisper on a CUDA GPU, else on the CPU
  2. ffmpeg detects every scene cut and builds labeled contact sheets of each shot
  3. Claude writes riffs in two passes: a writers' room brainstorms ~3x too many,
     then a head writer keeps only the best and punches them up. Every riff is
     anchored to the shot it reacts to, so visual jokes never land early.
  4. OpenAI TTS (or edge-tts / Windows voices) performs each heckler with per-line direction
  5. ffmpeg ducks the original audio under the riffs and overlays the theater-seat
     silhouettes, with each heckler bobbing while they talk

Requirements
  pip install anthropic openai faster-whisper edge-tts pillow numpy
  pip install nvidia-cublas-cu12 nvidia-cudnn-cu12   # optional: NVIDIA GPU transcription (~20x faster)
  ffmpeg + ffprobe on PATH          (Windows: winget install Gyan.FFmpeg)
  ANTHROPIC_API_KEY set, or `ant auth login`
  OPENAI_API_KEY set for the best voices (otherwise falls back to edge-tts, then Windows voices)
  Internet access

Usage
  python riffify.py movie.mp4                                   # --quality balanced (default)
  python riffify.py movie.mkv --title "Tears of Steel (2012)" --style "lots of 80s references"
  python riffify.py movie.mp4 --quality budget --batch          # cheapest: Sonnet + half-price batch
  python riffify.py movie.mp4 --quality max                     # Fable at max effort (expensive)
  python riffify.py movie.mp4 --riffs movie_riffed.riffs.json   # re-render, skip Claude
  python riffify.py movie.mp4 --candidates movie_riffed.candidates.json   # redo only the head writer

Every Claude call prints its token usage and cost, plus a projected cost per hour of video.

Output is never overwritten: movie_riffed.mp4, then movie_riffed_v2.mp4, ...
A .riffs.json with the script is saved next to the video so you can edit it and re-render.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48000
FRAME_FPS = 2          # frames pulled per second of video for the contact sheets
TILES_PER_SHEET = 6    # 3 x 2 grid per image sent to Claude
SECTION_SECONDS = 240  # writers' room brainstorms the video in parallel sections of about this length
MAX_SILENCE = 20       # longest stretch without a riff before the fill pass adds some

# --quality presets: (model, effort) for the writers' room (bulk brainstorming) and the head writer
# (taste: picking and punching up), plus how many frames go on the contact sheets.
PRESETS = {
    "budget":   {"writer": ("claude-sonnet-5-5", "medium"), "head": ("claude-sonnet-5-5", "medium"),
                 "max_frames": 240},
    "balanced": {"writer": ("claude-sonnet-5-5", "high"), "head": ("claude-opus-5-5", "high"),
                 "max_frames": 480},
    "max":      {"writer": ("claude-fable-5-1", "max"), "head": ("claude-fable-5-1", "max"),
                 "max_frames": 480},
}
# USD per million tokens (input, output), for the cost readout. --batch halves these.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0), "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0),
}

# The three hecklers. oai_* = OpenAI gpt-4o-mini-tts voice + acting direction (the default engine);
# el_voice = ElevenLabs voice name (--tts elevenlabs); defaults are stock voices every account has.
# voice/rate/pitch = edge-tts (`edge-tts --list-voices` shows options). win_* settings are the
# offline fallback (Windows built-in voices): win_rate is SAPI's -10..10 speed, win_shift is an
# ffmpeg pitch multiplier (<1 deeper, >1 higher) that gives each a distinct sound.
RIFFERS = {
    "OTTO": {
        "voice": "en-US-GuyNeural", "rate": "+8%", "pitch": "+0Hz",
        "win_voice": "Microsoft David Desktop", "win_rate": 1, "win_shift": 1.0,
        "oai_voice": "ash",
        "el_voice": "Chris",
        "oai_instructions": "A quick-witted comedian riffing on a bad movie with friends. Wry and sardonic but "
                            "animated: real pitch movement, a beat of timing before the punchline, then land the "
                            "last word with a little extra emphasis.",
        "persona": "The TV-headed robot host. Dry, deadpan everyman; sarcastic observations, "
                   "mock-sincere commentary, sighs at bad filmmaking.",
    },
    "GEARALD": {
        "voice": "en-US-ChristopherNeural", "rate": "+4%", "pitch": "-4Hz",
        "win_voice": "Microsoft David Desktop", "win_rate": 0, "win_shift": 0.86,
        "oai_voice": "echo",
        "el_voice": "Brian",
        "oai_instructions": "A pompous, theatrical robot hamming it up for an audience, with a General American "
                            "accent. Big dynamic range: grand "
                            "swells, dramatic pauses, over-enunciated words, occasional mock-operatic flourishes. "
                            "Clearly having the time of his life.",
        "persona": "A pompous box-headed robot with a single antenna. Theatrical and vain; takes the movie "
                   "personally and his pride is easily wounded. Name-drops famous celebrities as if they're "
                   "close personal friends. His references are well-known pop culture. Sometimes bursts "
                   "into a famous song hook that fits the moment, sung straight or with the words changed.",
    },
    "CHIP": {
        "voice": "en-US-EricNeural", "rate": "+14%", "pitch": "+28Hz",
        "win_voice": "Microsoft David Desktop", "win_rate": 2, "win_shift": 1.32,
        "oai_voice": "verse",
        "el_voice": "Callum",
        "oai_instructions": "A snarky, high-energy wiseguy robot. Quick, punchy, smug, with a bright nasal edge; "
                            "bounces between fake innocence and gleeful mockery, punches the funny word hard.",
        "persona": "A satellite-dish-headed robot. Snarky, impatient, a bit of a wiseguy; "
                   "cheap shots, fake voices for on-screen characters, groan-worthy puns.",
    },
}

SYSTEM_PROMPT = """You are the writers' room for a Mystery Science Theater 3000 / RiffTrax style riff track.
Three hecklers sit in the front row of a theater, silhouetted at the bottom of the screen, and riff on the video as it plays.

The hecklers:
{personas}

What you get: contact sheets covering every shot of the video (each tile is stamped with its shot number and timestamp), a shot list with exact start/end times, and a timestamped dialogue transcript (machine-made, so it may mishear words).

Comedy craft:
- Your first idea for a moment is everyone's first idea. Find the second or third angle: the unexpected comparison, the oddly specific detail, the wrong-but-perfect interpretation.
- Specific beats generic. Riff on what is actually on screen or actually said - a prop, a costume choice, an actor's expression, a line reading, a continuity oddity.
- Economy. Cut every word that isn't working and put the funniest word last.
- Vary the forms: dubbing fake dialogue over a character whose mouth is moving (or who is silently staring), voicing someone's inner thoughts, mock sound effects, a snatch of song, an announcer or nature-documentary voice, a quick back-and-forth between two hecklers, a one-word reaction, a callback.
- Build running gags: establish something early, escalate it, and pay it off later. A good final callback is gold.
- Avoid stock riff templates and filler: "Peak acting", "Bold.", "We get it.", "Also my nickname in high school", "X: the Y of tomorrow", "That's either an X or a Y", "This is longer than the movie". If a riff could be pasted onto any other film, cut it.
- Never reuse a joke structure. Once a template has been used ("Says the guy with a ___ for a ___", "Nothing says ___ like ___", "The face of a man who ___"), it is spent for the rest of the video.
- Punch at the movie and everything in it: the characters, and any fictional groups, factions, gangs, cults or creatures in the story, are fair game - don't hold back on them. Never punch at real-world groups of people (race, ethnicity, religion, sexuality, disability). Keep it PG-13, and never repeat slurs or racial terms from the film's dialogue, even when quoting a character.
- Write plain text: no emoji, music-note symbols or escape codes. For singing, just write the words and put "sung" in the delivery note.

Timing (critical - a riff that lands at the wrong moment is not funny):
- Every riff names the `shot` it reacts to (the shot number from the shot list).
- A riff about something you SEE must start after that shot begins - normally 0.4-1.5 s into it, once the audience has registered the image. Never start a visual joke before the shot that shows the thing.
- A riff reacting to a line of dialogue sets `responds_to_line` to that line's L number from the transcript; it will automatically start only after that line has finished. Time it right after the line ends. Use 0 when the riff isn't answering a specific line.
- Aim riffs at the gaps between lines: the PAUSES list gives the exact stretches with no dialogue. Most riffs should start and finish inside one; if a pause is coming within a few seconds, wait for it. Talking over dialogue is acceptable occasionally - when the joke only works at that exact moment, or the talking runs on with no usable pause - but keep it the exception, and never over a line that matters to the plot or sets up the joke.
- Keep the jokes coming: never leave more than about 20 seconds without a riff. A long dead stretch is worse than a riff that overlaps some dialogue.
- Plan for real speaking time: about 2 words per second, plus extra for sung or drawn-out delivery. Leave at least half a second between the end of one riff and the start of the next.
- Short: most riffs are 2-12 words (1-3 seconds spoken). Longer only over a long silent stretch.
- Back-and-forth exchanges: each reply is its own entry with `reply_to_previous: true`, timed right after the line it answers. A reply is only kept if its setup is, so the pair always lands together.

Fields: `time` = seconds from the start of the video when the line STARTS being spoken; `delivery` = a brief acting note for the voice actor (e.g. "mock-sincere", "fake movie-trailer voice", "whispered, horrified", "sung"); `reply_to_previous` = true only when this line directly answers the riff right before it. Spread riffs across all three hecklers, matched to their personalities."""

# Always sent as the producer's direction unless --style replaces it.
DEFAULT_STYLE = ("Riff on the movie itself first: what the characters say and do, their choices and "
                 "motives, the plot's twists and logic, and what's happening on screen. Jokes about the "
                 "acting, dialogue writing, costumes, sets, effects and filmmaking are welcome seasoning, "
                 "not the main course.")

RIFF_SCHEMA = {
    "type": "object",
    "properties": {
        "riffs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "shot": {"type": "integer"},
                    "time": {"type": "number"},
                    "speaker": {"type": "string", "enum": list(RIFFERS)},
                    "line": {"type": "string"},
                    "delivery": {"type": "string"},
                    "reply_to_previous": {"type": "boolean"},
                    "responds_to_line": {"type": "integer"},
                },
                "required": ["shot", "time", "speaker", "line", "delivery", "reply_to_previous",
                             "responds_to_line"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["riffs"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- helpers

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def require_tools():
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        sys.exit(f"Missing {', '.join(missing)} on PATH. Install ffmpeg first "
                 "(Windows: winget install Gyan.FFmpeg).")


def probe(video: Path):
    out = run(["ffprobe", "-v", "error", "-show_entries",
               "format=duration:stream=codec_type,width,height",
               "-of", "json", str(video)]).stdout
    info = json.loads(out)
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        sys.exit(f"{video} has no video stream.")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    width = int(v["width"])
    m = re.match(r"scale=(\d+):", square_pixels(video))
    if m:                                  # report the display width for non-square-pixel sources
        width = int(m[1])
    return float(info["format"]["duration"]), width, int(v["height"]), has_audio


def square_pixels(video: Path) -> str:
    """ffmpeg filter prefix that converts non-square pixels (old DVD/MPEG sources, e.g. 368x480 shown
    as 4:3) to square ones, so overlays and frame grabs aren't distorted. '' if already square."""
    info = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height,sample_aspect_ratio", "-of", "json", str(video)]).stdout)
    s = info["streams"][0]
    num, _, den = (s.get("sample_aspect_ratio") or "1:1").partition(":")
    sar = float(num) / float(den or 1) if num not in ("0", "") else 1.0
    if abs(sar - 1.0) < 0.01:
        return ""
    return f"scale={round(s['width'] * sar / 2) * 2}:{s['height']},setsar=1,"


def probe_fps(video: Path) -> float:
    rate = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                "-of", "default=nw=1:nk=1", str(video)]).stdout.decode().strip()
    num, _, den = rate.partition("/")
    return float(num) / float(den or 1) if float(num) else 24.0


AUDIO = {"stream": "a:0"}   # which audio track to use; set once in main() by pick_audio()


def pick_audio(video: Path, choice: str | None = None) -> str:
    """The movie audio track to use, as an ffmpeg stream specifier ('0' style index). Explicit
    --audio-track wins (a stream index like 3, or a language code like eng); otherwise the first
    English track that isn't commentary or audio description; otherwise the first track."""
    info = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                           "stream=index:stream_tags=language,title", "-of", "json", str(video)]).stdout)
    tracks = info.get("streams", [])
    if not tracks:
        return "a:0"
    def lang(s): return s.get("tags", {}).get("language", "").lower()
    def title(s): return s.get("tags", {}).get("title", "").lower()
    if choice:
        pick = next((s for s in tracks if str(s["index"]) == choice or lang(s) == choice.lower()), None)
        if pick is None:
            sys.exit(f"--audio-track {choice}: no such audio track (have: "
                     + ", ".join(f"{s['index']}={lang(s) or '?'}" for s in tracks) + ")")
    else:
        english = [s for s in tracks if lang(s) in ("eng", "en")
                   and not any(w in title(s) for w in ("comment", "descri"))]
        pick = (english or tracks)[0]
    return str(pick["index"])


def detect_picture(video: Path, duration: float, width: int, height: int):
    """Find black bars (e.g. a 4:3 film in a 16:9 file). Returns (x, y, w, h) of the picture area."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-ss", f"{duration * 0.25:.1f}", "-t", "60", "-i", str(video), "-an",
         "-vf", f"{square_pixels(video)}fps=2,cropdetect=24:2:0", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace")
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", proc.stderr)
    if not found:
        return 0, 0, width, height
    w, h, x, y = map(int, found[-1])
    if w >= width * 0.97 and h >= height * 0.97:   # no real bars
        return 0, 0, width, height
    return x, y, w, h


def detect_credits(video: Path, duration: float, picture):
    """When the end credits start, or None. Scans the last quarter of the video (at most 15 min) at
    one frame per second for the point after which the picture stays mostly black with sparse bright
    text (or fully black) right to the end."""
    x, y, w, h = picture
    fh = round(160 * h / w / 2) * 2
    span = min(duration * 0.25, 900)
    while True:
        start = duration - span
        raw = run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-i", str(video), "-an", "-vf",
                   f"{square_pixels(video)}crop={w}:{h}:{x}:{y},fps=1,scale=160:{fh},format=gray",
                   "-f", "rawvideo", "-"]).stdout
        frames = np.frombuffer(raw, np.uint8)
        frames = frames[: len(frames) // (160 * fh) * 160 * fh].reshape(-1, fh, 160)
        if len(frames) < 30:
            return None
        credit = ((frames < 35).mean(axis=(1, 2)) > 0.80) & ((frames > 150).mean(axis=(1, 2)) < 0.15)
        # credits can't start before the last stretch of real footage (6+ seconds in a row that don't
        # look like credits) - dark final scenes and black title cards mixed in otherwise fool it
        after = 0
        for i in range(len(credit) - 6, -1, -1):
            if not credit[i:i + 6].any():
                after = i + 6
                break
        found = next((i for i in range(after, len(credit) - 30) if credit[i] and credit[i:].mean() >= 0.85), None)
        if found is None:
            return None
        if found > 0 or start <= 0 or span >= 1800:   # credits begin inside the window: done
            return start + found
        span = min(duration, span * 2)                   # credits fill the window: look further back


def parse_clock(text: str) -> float:
    """'5400', '90:00' or '1:30:00' -> seconds."""
    secs = 0.0
    for part in text.split(":"):
        secs = secs * 60 + float(part)
    return secs


def next_free_path(path: Path) -> Path:
    """Never overwrite: foo.mp4 -> foo_v2.mp4 -> foo_v3.mp4 ..."""
    if not path.exists():
        return path
    n = 2
    while True:
        cand = path.with_name(f"{path.stem}_v{n}{path.suffix}")
        if not cand.exists():
            return cand
        n += 1


def fmt_ts(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


# --------------------------------------------------------------------------- analysis

TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
SUB_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})")


def parse_subtitles(text: str):
    """SRT / WebVTT text -> [(start, end, line)], with formatting tags stripped."""
    lines = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        m = SUB_TIME.search(block)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        rest = block[m.end():]
        body = rest.split("\n", 1)[1].strip().split("\n") if "\n" in rest else []   # skip VTT cue settings
        words = re.sub(r"<[^>]+>|\{[^}]*\}", "", " ".join(body)).strip()   # <i>, {\an8} etc.
        if words:
            lines.append((start, end, words))
    return lines


def sidecar_subtitles(video: Path):
    """English subtitle files next to the video: movie.srt, movie.en.srt, movie.eng.vtt, ..."""
    found = {}
    for suffix in ("", ".en", ".eng", ".english", ".English", ".EN"):
        for ext in (".srt", ".vtt"):
            p = video.with_name(video.stem + suffix + ext)
            if p.exists():
                found.setdefault(str(p.resolve()).lower(), p)   # Windows names are case-insensitive
    return list(found.values())


def subtitle_tracks(video: Path, explicit: Path | None = None):
    """Subtitles to carry into the output: every text track inside the video (all languages), plus an
    explicit or sidecar English file. Returns (embedded stream indexes, extra subtitle files)."""
    info = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
                           "stream=index,codec_name", "-of", "json", str(video)]).stdout)
    embedded = [s["index"] for s in info.get("streams", []) if s.get("codec_name") in TEXT_SUB_CODECS]
    files = [explicit] if explicit else sidecar_subtitles(video)
    return embedded, files


def find_subtitles(video: Path, explicit: Path | None = None):
    """English subtitles: an explicit file, a sidecar file next to the video (movie.srt,
    movie.en.srt, ...), or a text subtitle track inside the video. Returns (lines, description) or None.
    Image-based tracks (Blu-ray PGS, DVD VobSub) can't be read as text and are skipped."""
    if explicit:
        return parse_subtitles(explicit.read_text(encoding="utf-8", errors="replace")), explicit.name
    for side in sidecar_subtitles(video):
        lines = parse_subtitles(side.read_text(encoding="utf-8", errors="replace"))
        if lines:
            return lines, side.name
    info = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
                           "stream=index,codec_name:stream_tags=language,title:stream_disposition=forced",
                           "-of", "json", str(video)]).stdout)
    tracks = [s for s in info.get("streams", []) if s.get("codec_name") in TEXT_SUB_CODECS
              and not s.get("disposition", {}).get("forced")]
    english = [s for s in tracks if s.get("tags", {}).get("language", "").lower() in ("eng", "en")]
    english.sort(key=lambda s: "sdh" in s.get("tags", {}).get("title", "").lower())   # plain before [SDH]
    unlabeled = [s for s in tracks if not s.get("tags", {}).get("language")]
    for s in english + unlabeled:
        srt = run(["ffmpeg", "-v", "error", "-i", str(video), "-map", f"0:{s['index']}", "-f", "srt", "-"]).stdout
        lines = parse_subtitles(srt.decode("utf-8", errors="replace"))
        if lines:
            return lines, f"embedded subtitle track #{s['index']}"
    return None


def _add_cuda_dll_dirs():
    """faster-whisper's GPU path needs NVIDIA's cuBLAS/cuDNN DLLs. If they came from the pip packages
    (pip install nvidia-cublas-cu12 nvidia-cudnn-cu12), make Windows able to find them."""
    try:
        import nvidia
    except ImportError:
        return
    for base in nvidia.__path__:
        for sub in ("cublas", "cudnn", "cuda_nvrtc"):
            d = os.path.join(base, sub, "bin")
            if os.path.isdir(d):
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def gpu_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def transcribe(video: Path, whisper_model: str, use_gpu: bool = True, use_cpu: bool = True):
    """Whisper transcription on the GPU when one is usable, else the CPU. Returns [] if neither is allowed."""
    from faster_whisper import WhisperModel

    # decode with ffmpeg ourselves so faster-whisper never touches PyAV (version mismatches break it)
    raw = run(["ffmpeg", "-v", "error", "-i", str(video), "-map", f"0:{AUDIO['stream']}", "-f", "f32le",
               "-ac", "1", "-ar", "16000", "-"]).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    segments = None
    if use_gpu and gpu_available():
        _add_cuda_dll_dirs()
        try:
            print(f"  whisper '{whisper_model}' on GPU ...")
            model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
            segments = list(model.transcribe(audio, vad_filter=True)[0])
        except Exception as e:  # missing CUDA libraries, out of GPU memory, ...
            print(f"  GPU transcription failed ({str(e)[:100]})"
                  + ("; falling back to CPU" if use_cpu else "")
                  + "\n  (for GPU support: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12)")
    elif use_gpu:
        print("  no CUDA GPU found")
    if segments is None:
        if not use_cpu:
            print("  CPU transcription disabled; riffing without a transcript")
            return []
        print(f"  whisper '{whisper_model}' on CPU (slow) ...")
        model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        segments = list(model.transcribe(audio, vad_filter=True)[0])
    lines = []
    for s in segments:
        text = s.text.strip()
        if text:
            lines.append((s.start, s.end, text))
            print(f"    [{fmt_ts(s.start)}] {text}")
    return lines


def cap_dialogue(transcript, tight=False):
    """Whisper stretches a line's end time across any music that follows it (a two-word line can
    'last' two minutes), so cap each line at a speaking time for its word count. The default cap is
    generous (safe for "don't answer a line before it ends"); tight=True is a typical speaking pace,
    used for finding the pauses between lines."""
    per_word, slack = (0.34, 0.35) if tight else (0.38, 0.6)
    return [(s, min(e, s + per_word * len(t.split()) + slack), t) for s, e, t in transcript]


def find_pauses(dialogue, duration, min_len=1.5):
    """Stretches of at least `min_len` seconds with no dialogue: where riffs land cleanly."""
    pauses, cursor = [], 0.0
    for s, e, _ in sorted(dialogue):
        if s - cursor >= min_len:
            pauses.append((cursor, s))
        cursor = max(cursor, e)
    if duration - cursor >= min_len:
        pauses.append((cursor, duration))
    return pauses


def clean_line(text: str) -> str:
    """Undo literal \\uXXXX escapes the model sometimes copies from JSON."""
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), text).strip()


def tts_text(riff):
    """Text + delivery for the voice: music notes become a 'sung' direction instead of symbols."""
    line, delivery = riff["line"], riff.get("delivery", "")
    if "♪" in line or "♫" in line:
        line = line.replace("♪", "").replace("♫", "").strip()
        delivery = f"sung, {delivery}" if delivery else "sung"
    return line, delivery


SHOT_THRESHOLD = 0.1   # scene-change score that counts as a cut; soft modern cuts score ~0.12-0.25,
                       # ordinary frames < 0.03 (0.3 missed most cuts in darker, graded films)


def detect_shots(video: Path, duration: float, threshold: float = SHOT_THRESHOLD, min_len: float = 0.6):
    """Scene-cut detection -> list of (start, end) shots."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video), "-an",
         "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace")
    bounds = [0.0]
    for t in (float(x) for x in re.findall(r"pts_time:([\d.]+)", proc.stderr)):
        if t - bounds[-1] >= min_len and duration - t >= min_len:
            bounds.append(t)
    bounds.append(duration)
    return list(zip(bounds[:-1], bounds[1:]))


def extract_frames(video: Path, picture, tmp: Path):
    """Every frame at FRAME_FPS, black bars cropped, 450 px wide. Frame n is at n / FRAME_FPS s."""
    frame_dir = tmp / "frames"
    frame_dir.mkdir()
    x, y, w, h = picture
    run(["ffmpeg", "-y", "-i", str(video), "-an", "-vf",
         f"{square_pixels(video)}crop={w}:{h}:{x}:{y},fps={FRAME_FPS},scale=450:-2",
         "-q:v", "2", str(frame_dir / "f_%06d.jpg")])
    return sorted(frame_dir.glob("f_*.jpg"))


def build_contact_sheets(frames, shots, max_frames: int):
    """One frame shortly after each shot starts, plus one every few seconds in long shots, tiled
    into labeled grids. `shots` = [(shot number, start, end)]. Returns [(caption, jpeg, first, last shot)]."""
    from PIL import Image, ImageDraw, ImageFont

    def samples(step):
        out = []
        for n, s, e in shots:
            t = s + min(0.5, (e - s) / 2)
            out.append((n, t))
            t += step
            while t < e - 0.5:
                out.append((n, t))
                t += step
        return out

    # every 3 s if the frame budget allows, otherwise space frames out until it fits
    for step in (3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45, 60, 1e9):
        picks = samples(step)
        if len(picks) <= max_frames:
            break
    if len(picks) > max_frames:                    # very long videos: thin evenly
        k = len(picks) / max_frames
        picks = [picks[int(i * k)] for i in range(max_frames)]

    font = ImageFont.load_default(size=18)
    sheets = []
    for i in range(0, len(picks), TILES_PER_SHEET):
        group = picks[i:i + TILES_PER_SHEET]
        tiles = []
        for shot, t in group:
            img = Image.open(frames[min(round(t * FRAME_FPS), len(frames) - 1)]).convert("RGB")
            d = ImageDraw.Draw(img)
            label = f"S{shot}  {t:.1f}s"
            box = d.textbbox((6, 4), label, font=font)
            d.rectangle([box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3], fill=(0, 0, 0))
            d.text((6, 4), label, font=font, fill=(255, 255, 0))
            tiles.append(img)
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (tw * 3 + 8, th * 2 + 4), (40, 40, 40))
        for j, tile in enumerate(tiles):
            sheet.paste(tile, ((j % 3) * (tw + 4), (j // 3) * (th + 4)))
        buf = io.BytesIO()
        sheet.save(buf, "JPEG", quality=92)
        caption = f"Sheet {i // TILES_PER_SHEET + 1}: shots S{group[0][0]}-S{group[-1][0]}, {group[0][1]:.1f}s-{group[-1][1]:.1f}s"
        sheets.append((caption, buf.getvalue(), group[0][0], group[-1][0]))
    return sheets, len(picks)


LONG_VIDEO = 25 * 60     # riffed length above which the film is written in blocks
BLOCK_SECONDS = 20 * 60  # target block length; blocks are equal-sized, boundaries snapped to cuts
MAX_PARALLEL = 6         # writers' room requests in flight at once (stays clear of rate limits)


def plan_blocks(shots, riff_end: float, long_after: float = LONG_VIDEO, block_len: float = BLOCK_SECONDS):
    """Split numbered shots [(n, start, end)] into equal ~block_len blocks (one block for short videos)."""
    count = 1 if riff_end <= long_after else max(2, round(riff_end / block_len))
    cuts = [min(range(len(shots)), key=lambda i: abs(shots[i][1] - riff_end * k / count))
            for k in range(1, count)]
    edges = [0] + sorted(set(c for c in cuts if c > 0)) + [len(shots)]
    return [shots[a:b] for a, b in zip(edges, edges[1:]) if b > a]


def anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def compact(riffs) -> str:
    """Riffs as short readable lines for context: '[123s] OTTO: line'."""
    return "\n".join(f"[{r['time']:.0f}s] {r['speaker']}: {r['line']}"
                     for r in sorted(riffs, key=lambda r: r["time"]))


def make_gag_sheet(client, riffs, title: str, args):
    """Short notes for the next block's head writer: running gags (with how they've escalated) and
    jokes/templates already spent. One text-only call (--gag-model / --gag-effort, default: the head
    writer's model at high effort, so the summary is accurate); returns None if it fails."""
    import anthropic
    try:
        msg = client.messages.create(
            model=args.gag_model, max_tokens=16000, output_config={"effort": args.gag_effort},
            messages=[{"role": "user", "content":
                f"This is the riff track so far for a Mystery Science Theater 3000 style riff of {title}:\n\n"
                f"{compact(riffs)}\n\nWrite a compact GAG SHEET (under 300 words) for the writer of the next "
                f"part of the film:\nRUNNING GAGS - each recurring bit, how it has escalated so far, and "
                f"where it could go next.\nSPENT - specific jokes, targets and joke templates already used "
                f"that shouldn't be repeated.\nPlain text, no preamble."}])
    except anthropic.APIError as e:
        print(f"  gag sheet skipped ({type(e).__name__})")
        return None
    track_cost(msg, args.gag_model, "gag sheet:", batch=False)
    return next((b.text for b in msg.content if b.type == "text"), None)


# --------------------------------------------------------------------------- writing riffs

SPEND = {"usd": 0.0}


def track_cost(msg, model, label, batch):
    """Print and accumulate what a response cost (cache writes ~1.25x input, reads ~0.1x)."""
    u = msg.usage
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    usd = ((u.input_tokens + 1.25 * (u.cache_creation_input_tokens or 0)
            + 0.1 * (u.cache_read_input_tokens or 0)) * p_in + u.output_tokens * p_out) / 1e6
    if batch:
        usd /= 2
    SPEND["usd"] += usd
    print(f"  {label} {model}: {u.input_tokens:,} in / {u.output_tokens:,} out tokens = ${usd:.2f} "
          f"(run total ${SPEND['usd']:.2f})")


def parse_riffs(msg, label):
    if msg.stop_reason == "refusal":
        detail = msg.stop_details.explanation if msg.stop_details else ""
        raise RuntimeError(f"{label} Claude declined to riff this video. {detail}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError(f"{label} riff script was cut off (max_tokens). Try a larger --density.")
    text = next(b.text for b in msg.content if b.type == "text")
    riffs = json.loads(text)["riffs"]
    for r in riffs:
        r["line"] = clean_line(r["line"])
        r["delivery"] = clean_line(r.get("delivery", ""))
    return riffs


def request_params(model, effort, messages):
    return {
        "model": model,
        "max_tokens": 128000,
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": RIFF_SCHEMA}},
        "system": SYSTEM_PROMPT.format(
            personas="\n".join(f"- {name}: {r['persona']}" for name, r in RIFFERS.items())),
        "messages": messages,
    }


def call_claude(client, model, effort, messages, label="", attempts=3):
    """One streamed structured-output request, retried on dropped connections and server errors.
    (The SDK retries failed requests itself, but not a stream that dies midway.)"""
    import time
    import anthropic

    for attempt in range(1, attempts + 1):
        try:
            with client.beta.messages.stream(
                **request_params(model, effort, messages),
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # if the model declines, the API reroutes to a fallback model
            ) as stream:
                msg = stream.get_final_message()
            break
        except anthropic.APIStatusError as e:
            if e.status_code < 500 and e.status_code != 429 or attempt == attempts:
                raise
            print(f"  {label} API error {e.status_code}; retrying ({attempt}/{attempts}) ...")
        except Exception as e:  # dropped stream / connection reset
            if attempt == attempts:
                raise
            print(f"  {label} connection lost ({type(e).__name__}); retrying ({attempt}/{attempts}) ...")
        time.sleep(10 * attempt)
    track_cost(msg, model, label, batch=False)
    return parse_riffs(msg, label)


def run_batch(client, jobs):
    """Message Batches API: half price, no long-lived connection, but results can take a while
    (usually minutes, up to 24 h). jobs = {custom_id: (model, effort, messages)} -> {custom_id: riffs}."""
    import time
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch = client.messages.batches.create(requests=[
        Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**request_params(m, e, msgs)))
        for cid, (m, e, msgs) in jobs.items()])
    print(f"  batch {batch.id} submitted ({len(jobs)} requests); polling ...")
    while batch.processing_status != "ended":
        time.sleep(30)
        batch = client.messages.batches.retrieve(batch.id)
        c = batch.request_counts
        print(f"    {batch.processing_status}: {c.processing} processing, {c.succeeded} done, {c.errored} errored")
    out = {}
    for result in client.messages.batches.results(batch.id):
        cid = result.custom_id
        if result.result.type != "succeeded":
            raise RuntimeError(f"batch request {cid} {result.result.type}: {getattr(result.result, 'error', '')}")
        msg = result.result.message
        track_cost(msg, jobs[cid][0], f"{cid}:", batch=True)
        out[cid] = parse_riffs(msg, f"{cid}:")
    return out


def write_riffs(transcript, shots, sheets, span, title, args, candidates_path: Path, pauses=(), context=None):
    """Two passes over one stretch of film. `transcript` = [(line no, start, end, text)] and
    `shots` = [(shot no, start, end)], numbered across the whole film; `span` = (start, end) seconds.
    `context` (long videos only) = {"block": (k, of, film length), "prev_riffs": [...],
    "all_riffs": [...], "gag_sheet": str|None}. The writers' room candidates are saved to
    `candidates_path` before the head writer runs, so a failure can resume without re-paying for them."""
    import anthropic

    t_start, t_end = span
    target = max(3, round((t_end - t_start) / args.density))
    script = ("\n".join(f"L{n} [{s:.1f}-{e:.1f}] {t}" for n, s, e, t in transcript)
              or "(no dialogue in this stretch - riff on the visuals)")
    shot_list = "\n".join(f"S{n}: {s:.1f}-{e:.1f}s" for n, s, e in shots)
    if context:
        k, of, film = context["block"]
        where = (f"Video: {title}\nThis request covers block {k} of {of}: {t_start:.1f}-{t_end:.1f}s of a "
                 f"{film:.1f}-second film ({len(shots)} shots). Riff only on this stretch.")
    else:
        where = f"Video: {title}\nLength: {t_end:.1f} seconds, {len(shots)} shots."

    background = ""
    if getattr(args, "background_text", ""):
        background = ("BACKGROUND NOTES (reference from the producer - context, not direction: use what makes "
                      "a joke land, don't recite it, and never reveal this video's events before they "
                      f"happen on screen):\n{args.background_text}\n\n")

    def build_context(sheet_subset):
        blocks = [{"type": "text", "text":
                   f"{where}\n\n{background}"
                   "Contact sheets follow. Each tile is labeled with its shot number and the exact "
                   "timestamp of that frame."}]
        for caption, jpeg, _, _ in sheet_subset:
            blocks.append({"type": "text", "text": caption})
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.standard_b64encode(jpeg).decode("utf-8")}})
        pause_list = ", ".join(f"{a:.1f}-{b:.1f}" for a, b in pauses) or "(none found)"
        blocks.append({"type": "text", "text": f"SHOT LIST:\n{shot_list}\n\nTRANSCRIPT (seconds):\n{script}\n\n"
                       f"PAUSES (no dialogue for 1.5 s or more - the clean places to riff):\n{pause_list}"})
        return blocks

    producer = f"\n\nExtra direction from the producer: {args.style}" if args.style else ""
    if getattr(args, "avoid_list", None):
        producer += ("\n\nAVOID - these jokes and themes have been used too much in past riffs; don't use them "
                     "or close variants:\n" + "\n".join(f"- {item}" for item in args.avoid_list))
    client = anthropic.Anthropic()
    earlier = ""   # what the writers' room knows about the previous block (long videos only)
    if context and context["prev_riffs"]:
        earlier = (f"\n\nRIFFS FROM THE PREVIOUS BLOCK (already in the finished track):\n"
                   f"{compact(context['prev_riffs'])}\n\nDon't repeat those. Where it's funny, pitch "
                   f"continuations, escalations and callbacks of the running gags you can see there.")

    if args.candidates and not context:
        candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["riffs"]
        print(f"  loaded {len(candidates)} writers' room candidates from {args.candidates}")
    else:
        # Writers' room in parallel sections of ~SECTION_SECONDS: shorter requests survive flaky
        # connections, finish sooner, and each section is saved the moment it's done.
        from concurrent.futures import ThreadPoolExecutor

        sections, cur = [], []
        for shot in shots:
            cur.append(shot)
            if shot[2] - cur[0][1] >= SECTION_SECONDS:
                sections.append(cur)
                cur = []
        if cur:
            if sections and cur[-1][2] - cur[0][1] < SECTION_SECONDS / 3:
                sections[-1] += cur     # fold a short tail into the previous section
            else:
                sections.append(cur)

        w_model, w_effort = args.writer_model, args.writer_effort
        parts, jobs, bounds = {}, {}, {}
        for k, sec in enumerate(sections, 1):
            part = candidates_path.with_suffix(f".part{k}.json")
            if part.exists():
                print(f"  section {k}: reusing {part.name}")
                parts[k] = json.loads(part.read_text(encoding="utf-8"))["riffs"]
                continue
            a, b = sec[0][0], sec[-1][0]
            t0, t1 = sec[0][1], sec[-1][2]
            bounds[k] = (t0, t1)
            n_want = max(3, round((t1 - t0) / args.density * 3))
            subset = [sh for sh in sheets if sh[3] >= a and sh[2] <= b]
            jobs[k] = [{"role": "user", "content": build_context(subset) + [{"type": "text", "text":
                f"WRITERS' ROOM PASS, section {k} of {len(sections)}: shots S{a}-S{b} "
                f"({t0:.1f}-{t1:.1f}s). The contact sheets above cover just this section; the shot list "
                f"and transcript cover the whole {'block' if context else 'video'} for context. "
                f"Brainstorm about {n_want} candidate riffs for this section only - roughly three times "
                f"what the final track needs - so the head writer has plenty to choose from. Go for range "
                f"and surprise; it's fine if some miss. Pitch possible running gags too; the head writer "
                f"will connect them across sections." + earlier + producer}]}]

        def save_part(k, riffs):
            t0, t1 = bounds[k]
            riffs = [r for r in riffs if t0 - 1 <= r["time"] <= t1 + 1]
            part = candidates_path.with_suffix(f".part{k}.json")
            part.write_text(json.dumps({"riffs": riffs}, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  section {k}: {len(riffs)} candidates saved -> {part.name}")
            parts[k] = riffs

        if jobs:
            print(f"  writers' room: {len(jobs)} section(s), {len(sheets)} contact sheets, "
                  f"{len(transcript)} transcript lines -> {w_model} (effort {w_effort})"
                  f"{' via batch' if args.batch else ''} ...")
            if args.batch:
                results = run_batch(client, {f"section-{k}": (w_model, w_effort, m) for k, m in jobs.items()})
                for k in jobs:
                    save_part(k, results[f"section-{k}"])
            else:
                with ThreadPoolExecutor(max_workers=min(len(jobs), MAX_PARALLEL)) as pool:
                    futures = {k: pool.submit(call_claude, client, w_model, w_effort, m, f"section {k}:")
                               for k, m in jobs.items()}
                    for k, f in futures.items():
                        save_part(k, f.result())
        candidates = sorted((r for k in sorted(parts) for r in parts[k]), key=lambda r: r["time"])
        candidates_path.write_text(json.dumps({"riffs": candidates}, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"  {len(candidates)} candidates saved -> {candidates_path}")
    for r in candidates:
        r["line"] = clean_line(r["line"])
    pitch = [{k: r[k] for k in ("shot", "time", "speaker", "line", "delivery", "responds_to_line") if k in r}
             for r in candidates]

    h_model, h_effort = args.head_model, args.head_effort
    print(f"  head writer: {h_model} (effort {h_effort}){' via batch' if args.batch else ''} ...")
    history = ""   # what the head writer knows about earlier blocks (long videos only)
    if context and context["all_riffs"]:
        history = f"RIFFS ALREADY IN THE TRACK FROM EARLIER BLOCKS:\n{compact(context['all_riffs'])}\n\n"
        if context.get("gag_sheet"):
            history += f"GAG SHEET (running gags so far, and what's already spent):\n{context['gag_sheet']}\n\n"
    continuity = ("- This block continues the film: escalate and call back running gags from earlier "
                  "blocks where it's funny, and never repeat an earlier joke or reuse a spent template.\n"
                  if history else "")
    messages = [{"role": "user", "content": build_context(sheets) + [{"type": "text", "text":
            history +
            f"The writers' room pitched these candidate riffs:\n{json.dumps(pitch, indent=1, ensure_ascii=False)}\n\n"
            f"HEAD WRITER PASS: you are the ruthless head writer. From those candidates, build the final "
            f"riff track of about {target} riffs{' for this block' if context else ''}.\n" + continuity +
            "- Keep only riffs that would get a real laugh from an audience. Cut anything generic, "
            "templated, or that just describes what's on screen.\n"
            "- Punch up what survives: tighter wording, funniest word last, better specifics. You may "
            "also write new riffs if you see a better joke, including to fill a dead stretch.\n"
            "- No repeated joke structures: the sections were written separately, so check the whole "
            "track for templates used more than once (\"Says the guy with...\", \"Nothing says X like Y\") "
            "and keep only the best instance.\n"
            "- Make running gags escalate and pay off, and end strong.\n"
            "- Pacing: never more than about 20 seconds without a riff; a long dead stretch is worse "
            "than an occasional overlap.\n"
            "- Gaps first: check each riff against the PAUSES list. If it talks over dialogue and a pause "
            "is within a few seconds, move it into the pause; if it's too long for the pause, tighten "
            "the wording. Keep overlaps the exception.\n"
            "- Timing: visual riffs start 0.4-1.5 s after their shot begins. A riff answering a line sets "
            "responds_to_line and starts after that line ends. Budget about 2 words per second of "
            "speech and leave at least 0.5 s after each riff ends before the next one starts. Mark every "
            "direct reply to another heckler with reply_to_previous: true.\n"
            "- Keep the speakers balanced and in character, and give each line a useful delivery note."
            + producer}]}]
    if args.batch:
        riffs = run_batch(client, {"head-writer": (h_model, h_effort, messages)})["head-writer"]
    else:
        riffs = call_claude(client, h_model, h_effort, messages, label="head writer:")

    # Safety net for pacing: find stretches with no riffs for over MAX_SILENCE seconds (using an
    # estimated speaking time per riff) and ask for riffs in just those windows.
    def est_end(r):
        return r["time"] + 0.5 * len(r["line"].split()) + 0.3
    ordered = sorted(riffs, key=lambda r: r["time"])
    last = not context or context["block"][0] == context["block"][1]
    stop = t_end - 5 if last else t_end     # the film's last few seconds may stay quiet
    marks = [(t_start, t_start)] + [(r["time"], est_end(r)) for r in ordered] + [(stop, stop)]
    holes = [(a_end, b_start) for (_, a_end), (b_start, _) in zip(marks, marks[1:])
             if b_start - a_end > MAX_SILENCE]
    if not holes:
        return riffs
    quota = [max(1, round((b - a) / 11)) for a, b in holes]   # about one riff per 11 s of gap
    windows = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in holes)
    asks = "; ".join(f"{a:.1f}-{b:.1f}s: exactly {n} riff{'s' if n > 1 else ''}"
                     for (a, b), n in zip(holes, quota))
    print(f"  {len(holes)} dead stretch(es) over {MAX_SILENCE} s ({windows}); filling ...")
    fill_msg = [{"role": "user", "content": build_context(sheets) + [{"type": "text", "text":
        f"The finished riff track so far:\n{json.dumps(ordered, indent=1, ensure_ascii=False)}\n\n"
        f"The writers' room pitched these candidates:\n{json.dumps(pitch, indent=1, ensure_ascii=False)}\n\n"
        f"FILL PASS: these stretches currently have no riffs at all. Fill each with the stated number "
        f"of riffs - {asks}. As head writer, choose the funniest options for each slot: unused "
        f"candidates or new jokes, held to the same standard as the rest of the track. Spread them "
        f"across the window, prefer the PAUSES inside it, but a riff over dialogue beats leaving it "
        f"dead. Don't repeat jokes or templates already in the track. Return only the new riffs."
        + producer}]}]
    if args.batch:
        extra = run_batch(client, {"fill": (h_model, h_effort, fill_msg)})["fill"]
    else:
        extra = call_claude(client, h_model, h_effort, fill_msg, label="fill pass:")
    kept = []
    for (a, b), limit in zip(holes, quota):   # backstop only, if the model overshoots its quota
        inside = sorted((r for r in extra if a - 1 <= r["time"] <= b), key=lambda r: r["time"])
        step = len(inside) / limit if len(inside) > limit else 1
        kept += [inside[int(i * step)] for i in range(min(limit, len(inside)))]
    extra = kept
    print(f"  added {len(extra)} riffs to fill the dead stretches")
    return riffs + extra


VISUAL_SHOT_LEAD = 0.6   # a visual riff can't start until this long after its shot appears
VISUAL_DELAY = 0.4       # ...and starts this much later than written: time for the image to register


def anchor_riffs(riffs, shots, transcript, duration, visual_delay=VISUAL_DELAY):
    """Hard timing floors: a riff never starts before the dialogue line it answers has finished, and a
    visual riff never starts before VISUAL_SHOT_LEAD into its shot, nor before visual_delay after the
    moment it was written for (which may be mid-shot). Records the floor as `not_before` so the
    scheduler can slide riffs without ever breaking it. The written time is kept in `written`, so
    re-anchoring a saved script never stacks the delay twice."""
    out = []
    for r in riffs:
        written = r.setdefault("written", r["time"])
        floor = 0.0
        n = r.get("shot")
        if isinstance(n, int) and 1 <= n <= len(shots):
            floor = shots[n - 1][0] + VISUAL_SHOT_LEAD
        line = r.get("responds_to_line")
        if isinstance(line, int) and 1 <= line <= len(transcript):
            floor = transcript[line - 1][1] + 0.15         # answering dialogue: wait for the line
        else:
            floor = max(floor, written + visual_delay)    # visual: give the image time to register
        if floor:
            r["not_before"] = round(floor, 2)
            r["time"] = round(max(written, floor), 2)
        if 0 <= r["time"] < duration:
            out.append(r)
    out = sorted(out, key=lambda r: r["time"])
    warn_repeats(out)
    return out


def warn_repeats(riffs):
    """Flag riffs that open with the same three words - usually a reused joke template."""
    seen = {}
    for r in riffs:
        key = " ".join(re.sub(r"[^a-z' ]", "", r["line"].lower()).split()[:3])
        if len(key.split()) == 3:
            seen.setdefault(key, []).append(r["line"])
    for key, lines in seen.items():
        if len(lines) > 1:
            print(f"  note: {len(lines)} riffs open with '{key}...': " + " | ".join(lines))


# --------------------------------------------------------------------------- voicing

async def _synth_all(riffs, out_dir: Path):
    import edge_tts

    sem = asyncio.Semaphore(8)

    async def one(i, r):
        v = RIFFERS[r["speaker"]]
        path = out_dir / f"riff_{i:04d}.mp3"
        async with sem:
            await edge_tts.Communicate(tts_text(r)[0], v["voice"], rate=v["rate"],
                                       pitch=v["pitch"]).save(str(path))
        return path

    return await asyncio.gather(*(one(i, r) for i, r in enumerate(riffs)))


WINDOWS_TTS_PS1 = r"""
param([string]$JobFile)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($job in (Get-Content -Raw -Encoding UTF8 $JobFile | ConvertFrom-Json)) {
    $synth.SelectVoice($job.voice)
    $synth.Rate = [int]$job.rate
    $synth.SetOutputToWaveFile($job.out)
    $synth.Speak($job.text)
    $synth.SetOutputToNull()
}
"""


def _synth_windows(riffs, out_dir: Path):
    """Offline fallback: Windows' built-in SAPI voices, all lines in one PowerShell process."""
    jobs, paths = [], []
    for i, r in enumerate(riffs):
        v = RIFFERS[r["speaker"]]
        path = out_dir / f"riff_{i:04d}.wav"
        jobs.append({"text": tts_text(r)[0], "voice": v["win_voice"], "rate": v["win_rate"], "out": str(path)})
        paths.append(path)
    job_file = out_dir / "jobs.json"
    job_file.write_text(json.dumps(jobs), encoding="utf-8")
    script = out_dir / "tts.ps1"
    script.write_text(WINDOWS_TTS_PS1, encoding="utf-8")
    run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(job_file)])
    return paths


def _synth_openai(riffs, out_dir: Path):
    """OpenAI gpt-4o-mini-tts: natural voices that take acting direction per heckler and per line."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    client = OpenAI()

    def one(i, r):
        v = RIFFERS[r["speaker"]]
        path = out_dir / f"riff_{i:04d}.mp3"
        line, delivery = tts_text(r)
        direction = v["oai_instructions"] + (f" For this line: {delivery}." if delivery else "")
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=v["oai_voice"],
            input=line,
            instructions=direction,
            response_format="mp3",
        ) as resp:
            resp.stream_to_file(path)
        return path

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(one, range(len(riffs)), riffs))


EL_API = "https://api.elevenlabs.io"
EL_FORMAT = "mp3_44100_192"   # 192 kbps needs the Creator plan or above (free plan: mp3_44100_128)


def _el_request(path: str, body: dict | None = None):
    import urllib.request
    req = urllib.request.Request(EL_API + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"],
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def el_voice_ids():
    """{lowercase voice name: voice_id} for every voice the account can use."""
    voices = json.loads(_el_request("/v1/voices"))["voices"]
    return {v["name"].split(" - ")[0].strip().lower(): v["voice_id"] for v in voices}


def el_tagged_text(riff):
    """Eleven v3 takes direction as inline [audio tags]: each comma-separated part of the line's
    delivery note becomes a short tag. (The voice itself carries the character's personality.)"""
    line, delivery = tts_text(riff)
    parts = [p.strip().lower() for p in delivery.split(",") if p.strip()]
    parts = ["sings" if p in ("sung", "sing-song", "singing") else p for p in parts]
    tags = list(dict.fromkeys(p for p in parts if len(p.split()) <= 4))[:2]   # dedupe, keep order
    # trailing pause: without it Eleven v3 often stops the audio before the last word has finished
    return "".join(f"[{t}] " for t in tags) + line + " [short pause]"


def _synth_elevenlabs(riffs, out_dir: Path):
    """ElevenLabs Eleven v3 with per-line audio tags (needs ELEVENLABS_API_KEY)."""
    import time
    import urllib.error
    from concurrent.futures import ThreadPoolExecutor

    ids = el_voice_ids()
    missing = {v["el_voice"] for v in RIFFERS.values() if v["el_voice"].lower() not in ids}
    if missing:
        raise RuntimeError(f"ElevenLabs voice(s) not found: {', '.join(sorted(missing))}. "
                           f"Available: {', '.join(sorted(ids))}")
    chars = sum(len(el_tagged_text(r)) for r in riffs)
    print(f"  ElevenLabs: {len(riffs)} lines, ~{chars:,} characters")

    # Every voiced line is cached on disk (keyed by voice + text + settings), so a failed or repeated
    # run never pays for the same line twice.
    import hashlib
    cache = Path(__file__).with_name(".tts_cache") / "elevenlabs"
    cache.mkdir(parents=True, exist_ok=True)

    def one(i, r):
        v = RIFFERS[r["speaker"]]
        body = {"text": el_tagged_text(r), "model_id": "eleven_v3",
                "voice_settings": {"stability": 0.5}}   # v3: 0.0 creative / 0.5 natural / 1.0 robust
        voice_id = ids[v["el_voice"].lower()]
        key = hashlib.sha1(json.dumps([voice_id, body, EL_FORMAT], sort_keys=True).encode()).hexdigest()
        path = cache / f"{key}.mp3"
        if path.exists():
            return path
        for attempt in range(8):
            try:
                audio = _el_request(f"/v1/text-to-speech/{voice_id}?output_format={EL_FORMAT}", body)
                path.write_bytes(audio)
                return path
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 7:      # concurrency / rate limit: back off and retry
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"ElevenLabs {e.code}: {e.read().decode(errors='replace')[:300]}")

    with ThreadPoolExecutor(max_workers=2) as pool:   # free plan allows 2 concurrent requests
        return list(pool.map(one, range(len(riffs)), riffs))


def synthesize(riffs, out_dir: Path, engine: str):
    """Returns (clip paths, the engine that actually voiced them)."""
    if engine == "elevenlabs":
        return _synth_elevenlabs(riffs, out_dir), "elevenlabs"
    if engine == "openai" or (engine == "auto" and os.environ.get("OPENAI_API_KEY")):
        try:
            return _synth_openai(riffs, out_dir), "openai"
        except Exception as e:
            if engine == "openai":
                raise
            print(f"  OpenAI TTS failed ({type(e).__name__}: {str(e)[:80]}); trying edge-tts")
    if engine in ("auto", "edge"):
        try:
            return asyncio.run(_synth_all(riffs, out_dir)), "edge"
        except Exception as e:
            if engine == "edge" or sys.platform != "win32":
                raise
            print(f"  edge-tts unavailable ({type(e).__name__}: {str(e)[:80]}); using Windows voices")
    return _synth_windows(riffs, out_dir), "windows"


def decode_audio(path: Path, shift: float = 1.0) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if shift != 1.0:
        # resample trick: changes pitch by `shift`, atempo restores the original speed
        cmd += ["-af", f"aresample={SAMPLE_RATE},asetrate={round(SAMPLE_RATE * shift)},"
                       f"aresample={SAMPLE_RATE},atempo={1 / shift:.4f}"]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    return np.frombuffer(run(cmd).stdout, dtype=np.float32)


GAP = 0.4          # minimum silence between riffs
MAX_DELAY = 2.5    # how late a riff may start relative to its intended time
MAX_EARLY = 1.0    # how early a riff may be pulled to make room for the next one


def trim_silence(audio: np.ndarray, pad: float = 0.03) -> np.ndarray:
    """Cut the silent lead-in/tail TTS engines add, so a clip's length is its actual speech."""
    loud = np.flatnonzero(np.abs(audio) > 0.02)
    if not len(loud):
        return audio
    p = int(pad * SAMPLE_RATE)
    return audio[max(0, loud[0] - p): loud[-1] + p]


def trim_silence_soft(audio: np.ndarray, pad: float = 0.03, tail: float = 0.12) -> np.ndarray:
    """ElevenLabs version: words trail off softly, so measure silence relative to the clip's own
    peak (-40 dB), keep a longer tail, and fade the last 15 ms so the cut never clicks."""
    peak = np.abs(audio).max()
    loud = np.flatnonzero(np.abs(audio) > peak * 0.01) if peak else []
    if not len(loud):
        return audio
    out = audio[max(0, loud[0] - int(pad * SAMPLE_RATE)): loud[-1] + int(tail * SAMPLE_RATE)].copy()
    fade = min(len(out), int(0.015 * SAMPLE_RATE))
    out[len(out) - fade:] *= np.linspace(1.0, 0.0, fade)
    return out


SPEECH_DB = -22.5   # speech level every riff clip is leveled to (dBFS): what the OpenAI voices produce


def active_rms(audio: np.ndarray) -> float:
    frames = audio[: len(audio) // 480 * 480].reshape(-1, 480)   # 10 ms frames
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    voiced = rms[rms > 0.005]
    return float(np.sqrt(np.mean(voiced ** 2))) if len(voiced) else 0.0


def talk_envelope(audio: np.ndarray, fps: float) -> list:
    """0..1 loudness per video frame - drives the talking bob so it follows the syllables."""
    hop = int(SAMPLE_RATE / fps)
    n = max(1, len(audio) // hop)
    rms = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1))
    ref = np.percentile(rms, 90) or 1.0
    env = np.clip(rms / ref, 0, 1)
    env = np.where(env < 0.2, 0, env)                       # stay still in the pauses
    return [round(float(x), 2) for x in np.maximum(env, np.r_[0, env[:-1]] * 0.6)]  # soft release


GAP_EARLY = 0.8        # how much earlier than intended a riff may move to land in a dialogue pause
OVERLAP_WEIGHT = 1.5   # 1 s less talking-over-dialogue is worth moving a riff 1.5 s off its mark


def schedule(units, duration, dialogue=()):
    """Place units (a riff plus any direct replies) using their real voiced lengths. A unit may start
    up to MAX_DELAY late; if it would need more, the previous unit is pulled up to MAX_EARLY earlier
    (never before its floor: shot on screen / answered line finished). Units that still don't fit are
    dropped whole, so a reply is never left without its setup. A unit that would talk over dialogue
    may slide a little (GAP_EARLY earlier .. MAX_DELAY later) to land in a pause. Returns [(unit, start)]."""
    def talk_seconds(a, b):
        return sum(max(0.0, min(b, e) - max(a, s)) for s, e, _ in dialogue if s < b and e > a)

    placed = []
    for unit in units:
        first = unit[0][0]
        span = unit[-1][2]                                   # unit length incl. internal gaps
        t = first["time"]
        floor = max(r.get("not_before", 0.0) - off for r, off, _ in unit)
        earliest = placed[-1][1] + placed[-1][0][-1][2] + GAP if placed else 0.0
        start = max(t, earliest, floor)
        if start - t > MAX_DELAY and placed:
            prev_unit, prev_start = placed[-1]
            prev_floor = max(prev_unit[0][0].get("not_before", 0.0), prev_unit[0][0]["time"] - MAX_EARLY,
                             placed[-2][1] + placed[-2][0][-1][2] + GAP if len(placed) > 1 else 0.0)
            need = start - t - MAX_DELAY
            if 0 < need <= prev_start - prev_floor and start - need + span <= duration:
                placed[-1] = (prev_unit, prev_start - need)   # only move it if that makes room
                start -= need
        if dialogue and start - t <= MAX_DELAY:
            # Prefer pauses: try positions from GAP_EARLY before to MAX_DELAY after the intended time
            # (never below the floor or into the previous riff) and take the best trade-off between
            # seconds of talking over dialogue and seconds of being off the intended moment.
            lo = max(earliest, floor, t - GAP_EARLY)
            best = None
            for c in np.arange(lo, t + MAX_DELAY + 1e-9, 0.05):
                if c + span > duration:
                    break
                cost = OVERLAP_WEIGHT * talk_seconds(c, c + span) + abs(c - t)
                if best is None or cost < best[0] - 1e-9:
                    best = (cost, c)
            if best:
                start = float(best[1])
        if start - t > MAX_DELAY or start + span > duration:
            print("    dropped (no room): " + " / ".join(f"{r['speaker']}: {r['line']}" for r, _, _ in unit))
            continue
        placed.append((unit, start))
    return placed


def build_riff_track(riffs, duration, tmp: Path, out_wav: Path, engine: str = "auto", fps: float = 24.0,
                     dialogue=()):
    """Voices every riff first, then schedules with the real clip lengths and lays them on one track.
    Returns [{speaker, start, end, env}] for what was actually placed."""
    tts_dir = tmp / "tts"
    tts_dir.mkdir()
    print(f"  voicing {len(riffs)} riffs ...")
    clips, used = synthesize(riffs, tts_dir, engine)
    audios = [decode_audio(c, RIFFERS[r["speaker"]]["win_shift"] if used == "windows" else 1.0)
              for r, c in zip(riffs, clips)]

    audios = [trim_silence_soft(a) if used == "elevenlabs" else trim_silence(a) for a in audios]
    # level every clip to the same absolute speech loudness, whatever the voice engine (ElevenLabs
    # comes out ~6 dB hotter than OpenAI), so the riffs sit at the same level against the movie
    target = 10 ** (SPEECH_DB / 20)
    levels = [active_rms(a) for a in audios]
    audios = [a * min(max(target / lv, 0.25), 4.0) if lv > 0 else a for a, lv in zip(audios, levels)]

    # group each riff with its direct replies; replies follow 0.3-1.0 s after the line they answer
    units = []
    for r, a in zip(riffs, audios):
        length = len(a) / SAMPLE_RATE
        if r.get("reply_to_previous") and units:
            unit = units[-1]
            prev_end = unit[-1][2]
            offset = max(prev_end + 0.3, min(r["time"] - unit[0][0]["time"], prev_end + 1.0))
            unit.append((r, offset, offset + length, a))
        else:
            units.append([(r, 0.0, length, a)])
    shaped = [[(r, off, end) for r, off, end, _ in u] for u in units]
    audio_of = {id(r): a for u in units for r, _, _, a in u}

    track = np.zeros(int(duration * SAMPLE_RATE) + 1, dtype=np.float32)
    placed = []
    for unit, start in schedule(shaped, duration, dialogue):
        for r, off, end in unit:
            a = audio_of[id(r)]
            i = int((start + off) * SAMPLE_RATE)
            track[i:i + len(a)] += a[:len(track) - i]
            placed.append({"speaker": r["speaker"], "start": start + off, "end": start + end,
                           "env": talk_envelope(a, fps)})
            print(f"    [{fmt_ts(start + off)}] {r['speaker']}: {r['line']}")

    pcm = (np.clip(track, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return placed


# --------------------------------------------------------------------------- silhouette

def draw_layers(width: int, height: int, out_dir: Path, picture=None, seat_span=None):
    """Full-frame transparent PNGs: one per heckler (so each can move) plus the seat row on top.
    `picture` = (x, y, w, h) of the image area inside any black bars, so the silhouettes sit on the
    bottom edge of the picture rather than inside a letterbox or pillarbox bar.
    Returns {name: path} in back-to-front order."""
    from PIL import Image, ImageDraw

    ss = 3  # supersample for smooth edges
    W, H = width * ss, height * ss
    pic_x, pic_y, pic_w, pic_h = (v * ss for v in picture) if picture else (0, 0, W, H)
    B = pic_y + pic_h      # bottom edge of the picture: the seats sit on it
    ink = (8, 8, 10, 255)
    u = H / 100  # 1% of frame height
    seat_top = B - 11 * u
    layers = {}

    def X(frac):  # horizontal position as a fraction of the picture area
        return pic_x + frac * pic_w

    def layer(name, draw_fn):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        def ellipse(cx, cy, rx, ry):
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=ink)

        draw_fn(d, ellipse)
        path = out_dir / f"layer_{name.lower()}.png"
        img.resize((width, height), Image.LANCZOS).save(path)
        layers[name] = path

    def gearald(d, ellipse):  # left robot, seen from behind: boxy head, side bolts, single ball antenna
        sx = X(0.62)
        d.rounded_rectangle([sx - 7 * u, seat_top - 3 * u, sx + 7 * u, B], radius=2.5 * u, fill=ink)
        d.rectangle([sx - 1.8 * u, seat_top - 6.5 * u, sx + 1.8 * u, seat_top - 2 * u], fill=ink)
        d.rounded_rectangle([sx - 5.5 * u, seat_top - 16 * u, sx + 5.5 * u, seat_top - 5.5 * u],
                            radius=1.5 * u, fill=ink)
        for left in (sx - 7 * u, sx + 5 * u):                           # bolts on the sides
            d.rounded_rectangle([left, seat_top - 12.5 * u, left + 2 * u, seat_top - 9 * u],
                                radius=0.6 * u, fill=ink)
        d.rectangle([sx - 0.35 * u, seat_top - 20 * u, sx + 0.35 * u, seat_top - 15.5 * u], fill=ink)
        ellipse(sx, seat_top - 20.5 * u, 1.2 * u, 1.2 * u)              # antenna ball

    def otto(d, ellipse):  # middle robot, seen from behind: wide retro TV-set head with rabbit ears
        mx = X(0.75)
        d.rounded_rectangle([mx - 8 * u, seat_top - 2 * u, mx + 8 * u, B], radius=3 * u, fill=ink)
        d.rectangle([mx - 1.5 * u, seat_top - 6 * u, mx + 1.5 * u, seat_top - 1.5 * u], fill=ink)
        d.rounded_rectangle([mx - 7 * u, seat_top - 14 * u, mx + 7 * u, seat_top - 5 * u],
                            radius=2.5 * u, fill=ink)
        for tip in ((mx - 5 * u, seat_top - 21 * u), (mx + 4 * u, seat_top - 20 * u)):
            d.line([(mx, seat_top - 13.5 * u), tip], fill=ink, width=max(1, round(0.6 * u)))
            ellipse(tip[0], tip[1], 0.7 * u, 0.7 * u)                   # antenna tips

    def chip(d, ellipse):  # right robot, seen from behind: tilted satellite-dish head on a post
        import math
        cx = X(0.87)
        d.rounded_rectangle([cx - 5 * u, seat_top - 2 * u, cx + 5 * u, B], radius=2 * u, fill=ink)
        d.rectangle([cx - 0.9 * u, seat_top - 9 * u, cx + 0.9 * u, seat_top - 1 * u], fill=ink)
        ellipse(cx, seat_top - 9.5 * u, 2 * u, 1.4 * u)                 # swivel mount
        hx, hy, tilt = cx, seat_top - 13.5 * u, math.radians(-22)       # dish, tilted up and left
        dish = [(hx + 7.2 * u * math.cos(a) * math.cos(tilt) - 3.4 * u * math.sin(a) * math.sin(tilt),
                 hy + 7.2 * u * math.cos(a) * math.sin(tilt) + 3.4 * u * math.sin(a) * math.cos(tilt))
                for a in (k * math.pi / 30 for k in range(60))]
        d.polygon(dish, fill=ink)
        tip = (hx + 2.2 * u, hy - 7.5 * u)                              # feed arm peeking over the rim
        d.line([(hx, hy), tip], fill=ink, width=max(1, round(0.7 * u)))
        ellipse(tip[0], tip[1], 1 * u, 1 * u)

    def seats(d, ellipse):  # row of seats in front, occluding the bodies
        x0, x1 = (seat_span[0] * ss, seat_span[1] * ss) if seat_span else (pic_x, pic_x + pic_w)
        d.rectangle([x0, B - 4 * u, x1, B], fill=ink)
        seat_w = 14 * u
        x = x0 - seat_w * 0.3
        while x < x1:
            d.rounded_rectangle([x + 0.6 * u, seat_top, x + seat_w - 0.6 * u, B],
                                radius=3 * u, fill=ink)
            x += seat_w

    layer("GEARALD", gearald)
    layer("OTTO", otto)
    layer("CHIP", chip)
    layer("SEATS", seats)
    return layers


def write_bob_commands(placed, fps: float, height: int, path: Path, offset: float = 0.0) -> bool:
    """ffmpeg sendcmd script that nudges each heckler up with the loudness of their own voice,
    frame by frame, so the bob follows the syllables. Returns False if there's nothing to animate."""
    amp = max(2, round(height * 0.007))
    cmds = []
    for p in placed:
        target = f"overlay@{p['speaker'].lower()}"
        last = 0
        for j, e in enumerate(p["env"]):
            y = -round(amp * e)
            if y != last:
                cmds.append((offset + p["start"] + j / fps, target, y))
                last = y
        cmds.append((offset + p["end"] + 0.01, target, 0))
    cmds.sort()
    path.write_text("".join(f"{t:.3f} {tgt} y {y};\n" for t, tgt, y in cmds), encoding="utf-8")
    return bool(cmds)


# --------------------------------------------------------------------------- final mix

_NVENC_OK = None


def nvenc_available() -> bool:
    """Can ffmpeg encode on this NVIDIA GPU? (One-second test encode, checked once per run.)"""
    global _NVENC_OK
    if _NVENC_OK is None:
        proc = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=1",
                               "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, text=True, errors="replace")
        _NVENC_OK = proc.returncode == 0
        if not _NVENC_OK:
            reason = next((l for l in proc.stderr.splitlines() if "driver" in l.lower()), proc.stderr.strip()[-160:])
            print(f"  GPU encoder unavailable ({reason.strip()})")
    return _NVENC_OK


def video_codec(args, quality: int):
    """H.264 encoder arguments: NVIDIA GPU (NVENC) when available, CPU (x264) otherwise.
    --encoder auto tries the GPU and falls back; gpu requires it; cpu never uses it."""
    mode = getattr(args, "encoder", "auto")
    if mode != "cpu" and nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(quality + 1), "-b:v", "0"]
    if mode == "gpu":
        sys.exit("--encoder gpu: the NVIDIA GPU encoder isn't available (see the reason above).")
    return ["-c:v", "libx264", "-crf", str(quality), "-preset", "medium"]


def render(video: Path, riff_wav: Path, layers, bob_cmds: Path | None, has_audio: bool,
           duration: float, out: Path, args):
    inputs = ["-i", str(video), "-i", str(riff_wav)]
    riff = f"[1:a]aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo,volume={args.riff_gain}"
    if has_audio:
        filters = [
            f"{riff},asplit=2[key][rf]",
            f"[0:{AUDIO['stream']}]aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo[orig]",
            # duck the movie whenever a heckler talks
            f"[orig][key]sidechaincompress=threshold=0.015:ratio={args.duck}:attack=15:release=400[duck]",
            "[duck][rf]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[aout]",
        ]
    else:
        filters = [f"{riff}[aout]"]

    vmap = "0:v:0"
    square = square_pixels(video)
    if square:                                     # draw on square pixels so shapes aren't stretched
        filters.append(f"[0:v]{square.rstrip(',')}[vsq]")
        vmap = "[vsq]"
    if layers:
        prev = "vsq" if square else "0:v"
        if bob_cmds:   # run from bob_cmds' folder so the filter arg needs no Windows path escaping
            filters.append(f"[{prev}]sendcmd=f={bob_cmds.name}[vcmd]")
            prev = "vcmd"
        for k, (name, path) in enumerate(layers.items()):
            inputs += ["-i", str(path)]
            label = f"v{k}"
            filters.append(f"[{prev}][{k + 2}:v]overlay@{name.lower()}=x=0:y=0:eval=frame:format=auto[{label}]")
            prev = label
        vmap = f"[{prev}]"

    # carry the original subtitles over (as MP4 text tracks, keeping language/title tags)
    sub_args = []
    if not getattr(args, "no_subtitle_tracks", False):
        embedded, files = subtitle_tracks(video, getattr(args, "subtitles", None))
        sub_maps = [f"0:{i}" for i in embedded]
        for f in files:
            sub_maps.append(f"{inputs.count('-i')}:0")
            inputs += ["-i", str(f)]
        for m in sub_maps:
            sub_args += ["-map", m]
        for k in range(len(embedded), len(sub_maps)):   # sidecar files carry no language tag
            sub_args += [f"-metadata:s:s:{k}", "language=eng"]
        if sub_maps:
            sub_args += ["-c:s", "mov_text"]
            print(f"  keeping {len(sub_maps)} subtitle track(s)")

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters),
           "-map", vmap, "-map", "[aout]", *sub_args, "-t", f"{duration:.3f}",
           *video_codec(args, quality=20), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    print("  rendering ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          cwd=bob_cmds.parent if bob_cmds else None)
    if proc.returncode != 0:
        sys.exit("ffmpeg failed:\n" + proc.stderr[-3000:])


# --------------------------------------------------------------------------- YouTube sync version

YT_W, YT_H = 1920, 1080
YT_SCREEN = (300, 50, 1620, 900)     # the theater's movie screen: left, top, right, bottom
# fonts for the YouTube version's text, first match wins (Windows, macOS, Linux)
FONT_BOLD = [r"C:\Windows\Fonts\arialbd.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
             "/Library/Fonts/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]
FONT_MONO = [r"C:\Windows\Fonts\consolab.ttf", "/System/Library/Fonts/Menlo.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf"]


def find_font(candidates) -> Path:
    found = next((Path(p) for p in candidates if Path(p).exists()), None)
    if not found:
        sys.exit("No usable font found for the YouTube version; looked for: " + ", ".join(candidates))
    return found


def draw_theater(path: Path):
    """Dark theater interior: red folded curtains, scalloped valance, a dimly glowing empty screen."""
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGB", (YT_W, YT_H), (10, 7, 8))
    d = ImageDraw.Draw(img)
    l, t, r, b = YT_SCREEN
    # soft light spill around the screen, then the screen itself with a faint centre glow
    glow = Image.new("L", (YT_W, YT_H), 0)
    ImageDraw.Draw(glow).rectangle([l - 40, t - 30, r + 40, b + 50], fill=70)
    img.paste((40, 38, 46), mask=glow.filter(ImageFilter.GaussianBlur(60)))
    d.rectangle([l, t, r, b], fill=(46, 49, 58))
    centre = Image.new("L", (YT_W, YT_H), 0)
    ImageDraw.Draw(centre).ellipse([l + 200, t + 80, r - 200, b - 60], fill=40)
    img.paste((72, 77, 90), mask=centre.filter(ImageFilter.GaussianBlur(120)))
    # curtains: vertical folds shaded with a sine wave, left and right, down to the floor line
    for x0, x1 in ((0, l - 30), (r + 30, YT_W)):
        for x in range(x0, x1):
            shade = 0.55 + 0.45 * (0.5 + 0.5 * np.sin((x - x0) / 22.0))
            d.line([(x, 0), (x, b)], fill=(int(120 * shade), int(16 * shade), int(22 * shade)))
    # floor below the screen: faint light spill fading toward the viewer
    for yy in range(b, YT_H):
        k = (yy - b) / (YT_H - b)
        d.line([(0, yy), (YT_W, yy)], fill=(int(30 - 18 * k), int(24 - 15 * k), int(28 - 17 * k)))
    # the row nearest the viewer: bigger, near-black seat backs, staggered against the riffers' row
    seat_w = 0.14 * YT_H * 1.45                     # riffers' seats are 14% of the frame height wide
    sx = -seat_w / 2
    while sx < YT_W:
        d.rounded_rectangle([sx + 6, b + 18, sx + seat_w - 6, YT_H + 40], radius=46, fill=(5, 4, 5))
        sx += seat_w
    # valance across the top with scalloped edge
    d.rectangle([0, 0, YT_W, 34], fill=(96, 12, 18))
    for cx in range(0, YT_W + 80, 80):
        d.ellipse([cx - 40, 8, cx + 40, 54], fill=(96, 12, 18))
    img.save(path)


def yt_filter_text(text: str) -> str:
    """Escape literal text for an ffmpeg drawtext value inside single quotes."""
    return text.replace("\\", "\\\\").replace("'", "\u2019").replace(":", "\\:").replace("%", "\\%")


def render_youtube(video: Path, riff_wav: Path, placed, picture, duration: float, title: str,
                   out: Path, tmp: Path, args):
    """Companion version for public posting: no movie audio, no subtitles. A theater scene with the
    robot row, a MOVIE TIME clock, and a small, heavily blurred, silent glimpse of the film for a few
    seconds at the start and every --yt-sync-every seconds so viewers can sync their own copy."""
    pre, offset = args.yt_preroll, args.time_offset
    total = pre + duration
    yt = tmp / "yt"
    yt.mkdir(exist_ok=True)
    draw_theater(yt / "theater.png")
    shutil.copy(find_font(FONT_BOLD), yt / "bold.ttf")   # fonts copied next to the job: no path
    shutil.copy(find_font(FONT_MONO), yt / "mono.ttf")   # escaping inside the filter arguments
    l, t, r, b = YT_SCREEN
    # robots sit on the screen's bottom edge; their row of seats runs the full width, over the curtains
    layers = draw_layers(YT_W, YT_H, yt, (l, t, r - l, b - t), seat_span=(0, YT_W))
    has_bob = write_bob_commands(placed, probe_fps(video), YT_H, yt / "bob.txt", offset=pre)

    # sync checkpoints in film-clock time -> output time
    windows = [(pre + s, pre + s + (args.yt_first_sync if s == 0 else args.yt_sync_length))
               for s in np.arange(0, duration - args.yt_sync_length, args.yt_sync_every)]
    show = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in windows) or "0"

    l, t, r, b = YT_SCREEN
    x, y, w, h = picture
    box_w = 480
    box_h = min(240, round(box_w * h / w / 2) * 2)
    box_w = round(box_h * w / h / 2) * 2
    bx, by = (l + r - box_w) // 2, t + 60
    clock_y = by + box_h + 30

    hms = (r"%{eif\:trunc((t-PRE+OFF)/3600)\:d\:1}\:%{eif\:mod(trunc((t-PRE+OFF)/60)\,60)\:d\:2}"
           r"\:%{eif\:mod(trunc(t-PRE+OFF)\,60)\:d\:2}").replace("PRE", f"{pre}").replace("OFF", f"{offset}")
    filters = [
        "[2:v]sendcmd=f=bob.txt[bg]" if has_bob else "[2:v]null[bg]",
        # the glimpse: picture area only, shrunk to 40 px wide and blown back up (pixelated), then blurred
        f"[0:v]{square_pixels(video)}crop={w}:{h}:{x}:{y},scale=40:-2,scale={box_w}:{box_h}:flags=neighbor,"
        f"gblur=sigma=12,setpts=PTS+{pre}/TB,fps=30[peek]",
        f"[bg][peek]overlay={bx}:{by}:enable='{show}':eof_action=pass[v0]",
        f"[v0]drawtext=fontfile=bold.ttf:text='{yt_filter_text('SYNC CHECK')}':fontsize=34:"
        f"fontcolor=0xFFD24A:x=(w-tw)/2:y={by - 50}:enable='{show}',"
        f"drawtext=fontfile=bold.ttf:text='MOVIE TIME':fontsize=36:fontcolor=0xBFC4CC:"
        f"x=(w-tw)/2:y={clock_y}:enable='gte(t,{pre})',"
        f"drawtext=fontfile=mono.ttf:text='{hms}':fontsize=130:fontcolor=white@0.92:"
        f"x=(w-tw)/2:y={clock_y + 46}:enable='gte(t,{pre})',"
        f"drawtext=fontfile=bold.ttf:text='{yt_filter_text(title)}':fontsize=44:fontcolor=white:"
        f"x=(w-tw)/2:y={t + 170}:enable='lt(t,{pre})',"
        f"drawtext=fontfile=bold.ttf:text='{yt_filter_text('Start your copy of the movie when the clock reaches ' + fmt_clock(offset))}':"
        f"fontsize=40:fontcolor=0xBFC4CC:x=(w-tw)/2:y={t + 250}:enable='lt(t,{pre})',"
        f"drawtext=fontfile=mono.ttf:text='%{{eif\\:ceil({pre}-t)\\:d}}':fontsize=200:fontcolor=0xFFD24A:"
        f"x=(w-tw)/2:y={t + 340}:enable='lt(t,{pre})'[v1]",
    ]
    inputs = ["-i", str(video), "-i", str(riff_wav), "-loop", "1", "-framerate", "30", "-i", str(yt / "theater.png")]
    prev = "v1"
    for k, (name, path) in enumerate(layers.items()):
        inputs += ["-i", str(path)]
        filters.append(f"[{prev}][{k + 3}:v]overlay@{name.lower()}=x=0:y=0:eval=frame[l{k}]")
        prev = f"l{k}"
    # audio: the riffs only, shifted by the countdown
    filters.append(f"[1:a]aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo,volume={args.riff_gain},"
                   f"adelay={int(pre * 1000)}:all=1,apad,atrim=0:{total:.3f}[aout]")
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters),
           "-map", f"[{prev}]", "-map", "[aout]", "-t", f"{total:.3f}", "-r", "30",
           *video_codec(args, quality=22), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    print("  rendering YouTube sync version ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", cwd=yt)
    if proc.returncode != 0:
        sys.exit("ffmpeg failed (YouTube version):\n" + proc.stderr[-3000:])


def export_mp3(riff_wav: Path, duration: float, out: Path, args):
    """The riff track by itself as one MP3: same timing as the video, so it starts with the film."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(riff_wav), "-af", f"volume={args.riff_gain},alimiter=limit=0.95",
         "-t", f"{duration:.3f}", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "192k", str(out)],
        capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        sys.exit("ffmpeg failed (MP3):\n" + proc.stderr[-2000:])


def fmt_clock(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


# --------------------------------------------------------------------------- main

CONFIG_FILE = Path(__file__).with_name("config.toml")


def apply_config(parser, path: Path):
    """Personal settings from config.toml (copy config.example.toml to start one). [voices.openai] and
    [voices.elevenlabs] set each robot's voice; [defaults] sets option defaults by option name
    (e.g. quality = "balanced", youtube = true). Command-line flags still win."""
    if not path.exists():
        return
    import tomllib
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"{path.name}: {e}")
    known = {"voices", "defaults"}
    if set(cfg) - known:
        sys.exit(f"{path.name}: unknown section(s) {sorted(set(cfg) - known)}; expected {sorted(known)}")
    for engine, field in (("openai", "oai_voice"), ("elevenlabs", "el_voice")):
        for speaker, voice in cfg.get("voices", {}).get(engine, {}).items():
            if speaker.upper() not in RIFFERS:
                sys.exit(f"{path.name}: [voices.{engine}] {speaker} - robots are {', '.join(RIFFERS)}")
            RIFFERS[speaker.upper()][field] = voice
    defaults = {k.replace("-", "_"): v for k, v in cfg.get("defaults", {}).items()}
    options = {a.dest for a in parser._actions}
    if set(defaults) - options:
        sys.exit(f"{path.name}: unknown option(s) in [defaults]: {sorted(set(defaults) - options)} "
                 f"(use the option names from --help, e.g. riff_gain)")
    parser.set_defaults(**defaults)
    print(f"  settings from {path.name}")


def main():
    p = argparse.ArgumentParser(description="Turn a video into an MST3K / RiffTrax style riffed version.")
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--output", type=Path, help="output .mp4 (default: <name>_riffed.mp4)")
    p.add_argument("--title", help='what the video is, e.g. "Tears of Steel (2012)" (default: file name)')
    efforts = ["low", "medium", "high", "xhigh", "max"]
    p.add_argument("--quality", choices=list(PRESETS), default="balanced",
                   help="budget: Sonnet throughout | balanced (default): Sonnet brainstorms, Opus picks | "
                        "max: Fable at max effort for both (much pricier)")
    p.add_argument("--writer-model", help="override the writers' room model")
    p.add_argument("--writer-effort", choices=efforts, help="override the writers' room effort")
    p.add_argument("--head-model", help="override the head writer model")
    p.add_argument("--head-effort", choices=efforts, help="override the head writer effort")
    p.add_argument("--batch", action="store_true",
                   help="use the Message Batches API: half price, but results can take minutes to hours")
    p.add_argument("--density", type=float, default=12, help="average seconds between riffs (default 12)")
    p.add_argument("--style", default=DEFAULT_STYLE,
                   help='direction for the writers, or @file to read it from a text file; replaces the '
                        'default style note entirely '
                        '(e.g. "lean into 90s references; the cult in this movie is fair game")')
    p.add_argument("--background", action="append", default=[], metavar="FILE",
                   help="reference notes for the writers (e.g. a recap of earlier films in a series), sent as "
                        "context rather than direction; repeatable. Keep it short: every writing call gets it")
    p.add_argument("--whisper-model", default="large-v3", help="tiny/base/small/medium/large-v3")
    p.add_argument("--avoid", metavar="FILE|off",
                   help="file of jokes/themes to avoid (one per line, # for comments); default: avoid.txt next "
                        "to this script if it exists (start from avoid.example.txt); 'off' to skip it")
    p.add_argument("--audio-track", metavar="INDEX|LANG",
                   help="movie audio track to use: a stream index (e.g. 3) or language (e.g. eng); "
                        "default: the first English track")
    p.add_argument("--subtitles", type=Path, help="use this .srt/.vtt file for the dialogue")
    p.add_argument("--no-subtitles", action="store_true",
                   help="ignore subtitles (sidecar or embedded) and always transcribe")
    p.add_argument("--no-subtitle-tracks", action="store_true",
                   help="don't copy the original subtitle tracks into the output video")
    p.add_argument("--no-gpu", action="store_true", help="never transcribe on the GPU")
    p.add_argument("--no-cpu", action="store_true",
                   help="never fall back to slow CPU transcription (riff without a transcript instead)")
    p.add_argument("--max-frames", type=int,
                   help="max frames on the contact sheets, 6 per image sent to Claude (default: from --quality)")
    p.add_argument("--riffs", type=Path, help="reuse an existing .riffs.json instead of calling Claude")
    p.add_argument("--candidates", type=Path,
                   help="reuse a saved .candidates.json and run only the head writer pass")
    p.add_argument("--tts", choices=["auto", "openai", "elevenlabs", "edge", "windows"], default="auto",
                   help="voice engine (auto: OpenAI if OPENAI_API_KEY is set, then edge-tts, then Windows; "
                        "elevenlabs needs ELEVENLABS_API_KEY)")
    p.add_argument("--el-voice", action="append", default=[], metavar="SPEAKER=NAME",
                   help="override an ElevenLabs voice by name for this run, e.g. --el-voice CHIP=Liam")
    p.add_argument("--voice", action="append", default=[], metavar="SPEAKER=VOICE",
                   help="override an OpenAI voice for this run, e.g. --voice GEARALD=echo (repeatable). "
                        "Voices: alloy ash ballad cedar coral echo fable marin nova onyx sage shimmer verse")
    p.add_argument("--visual-delay", type=float, default=VISUAL_DELAY,
                   help=f"seconds visual riffs start after the moment they were written for (default {VISUAL_DELAY})")
    p.add_argument("--riff-gain", type=float, default=1.5, help="heckler volume multiplier")
    p.add_argument("--duck", type=float, default=6, help="how hard to duck the movie under riffs (ratio)")
    p.add_argument("--silhouette", action=argparse.BooleanOptionalAction, default=True,
                   help="robots and theater seats over the main video; --no-silhouette leaves the picture "
                        "clean (the YouTube version always has them)")
    p.add_argument("--youtube", action="store_true",
                   help="also make a YouTube-ready sync version: theater scene, robots, movie clock, riffs only "
                        "(no movie audio or subtitles), with brief blurred sync glimpses of the film")
    p.add_argument("--credits", default="auto", metavar="auto|off|TIME",
                   help="where the end credits start (not riffed, not sent to Claude; YouTube/MP3 versions stop "
                        "there): auto-detect (default), off, or a time like 1:38:20")
    p.add_argument("--credits-riff", type=float, default=90,
                   help="keep riffing this many seconds into the end credits (default 90)")
    p.add_argument("--mp3", action="store_true",
                   help="also export the riffs alone as one MP3, timed to the video (to play alongside it)")
    p.add_argument("--yt-preroll", type=float, default=10, help="countdown before the movie starts (s)")
    p.add_argument("--yt-sync-every", type=float, default=600, help="seconds between sync glimpses")
    p.add_argument("--yt-first-sync", type=float, default=20,
                   help="length of the first sync glimpse, at the start (s)")
    p.add_argument("--yt-sync-length", type=float, default=10, help="length of each sync glimpse (s)")
    p.add_argument("--time-offset", type=float, default=0,
                   help="film time at the start of this video, for the clock (e.g. 605 for a part starting at 10:05)")
    p.add_argument("--gag-model", help="model for the gag sheet (default: the head writer's model)")
    p.add_argument("--gag-effort", choices=efforts, default="high", help="effort for the gag sheet (default high)")
    p.add_argument("--block-minutes", type=float, default=BLOCK_SECONDS / 60,
                   help="long videos: target block length in minutes (default 20)")
    p.add_argument("--long-after", type=float, default=LONG_VIDEO / 60,
                   help="use block mode when more than this many minutes are riffed (default 25)")
    p.add_argument("--gag-sheet", action=argparse.BooleanOptionalAction, default=True,
                   help="long videos: after each block, summarize running gags and spent jokes for the next "
                        "block's head writer (default on; --no-gag-sheet to skip)")
    p.add_argument("--encoder", choices=["auto", "gpu", "cpu"], default="auto",
                   help="video encoding: auto (NVIDIA GPU if it works, else CPU), gpu (GPU or stop), cpu")
    p.add_argument("--keep-temp", action="store_true")
    apply_config(p, CONFIG_FILE)   # config.toml next to the script: voices + option defaults
    args = p.parse_args()
    avoid_file = None if args.avoid == "off" else Path(args.avoid) if args.avoid else Path(__file__).with_name("avoid.txt")
    args.avoid_list = []
    if avoid_file and avoid_file.exists():                            # jokes/themes worn out by past use
        args.avoid_list = [l.strip() for l in avoid_file.read_text(encoding="utf-8").splitlines()
                           if l.strip() and not l.strip().startswith("#")]
        print(f"  avoiding {len(args.avoid_list)} worn-out joke theme(s) from {avoid_file.name}")
    elif args.avoid and args.avoid != "off":
        sys.exit(f"--avoid: {args.avoid} not found")
    if args.style and args.style.startswith("@"):                  # --style @styles/movie.txt
        args.style = Path(args.style[1:]).read_text(encoding="utf-8").strip()
    notes = []
    for name in args.background:                                    # --background background/movie.md
        path = Path(name.removeprefix("@"))
        if not path.exists():
            sys.exit(f"--background: {path} not found")
        notes.append(f"[{path.name}]\n{path.read_text(encoding='utf-8').strip()}")
        print(f"  background notes from {path.name} (~{len(notes[-1]) // 4:,} tokens per writing call)")
    args.background_text = "\n\n".join(notes)
    preset = PRESETS[args.quality]
    args.writer_model = args.writer_model or preset["writer"][0]
    args.writer_effort = args.writer_effort or preset["writer"][1]
    args.head_model = args.head_model or preset["head"][0]
    args.head_effort = args.head_effort or preset["head"][1]
    args.gag_model = args.gag_model or args.head_model
    args.max_frames = args.max_frames or preset["max_frames"]
    for spec in args.voice:
        speaker, _, voice = spec.partition("=")
        if speaker.upper() not in RIFFERS or not voice:
            sys.exit(f"--voice expects SPEAKER=VOICE with SPEAKER one of {', '.join(RIFFERS)}; got {spec!r}")
        RIFFERS[speaker.upper()]["oai_voice"] = voice.strip()
        print(f"  {speaker.upper()} voice: {voice.strip()}")
    for spec in args.el_voice:
        speaker, _, voice = spec.partition("=")
        if speaker.upper() not in RIFFERS or not voice:
            sys.exit(f"--el-voice expects SPEAKER=NAME with SPEAKER one of {', '.join(RIFFERS)}; got {spec!r}")
        RIFFERS[speaker.upper()]["el_voice"] = voice.strip()
        print(f"  {speaker.upper()} ElevenLabs voice: {voice.strip()}")

    require_tools()
    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"Not found: {video}")

    out = next_free_path((args.output or video.with_name(f"{video.stem}_riffed.mp4")).resolve())
    script_path = out.with_suffix(".riffs.json")

    duration, width, height, has_audio = probe(video)
    print(f"{video.name}: {fmt_ts(duration)}, {width}x{height}, audio={'yes' if has_audio else 'no'}")
    picture = detect_picture(video, duration, width, height)
    if has_audio:
        AUDIO["stream"] = pick_audio(video, args.audio_track)
        print(f"  movie audio: stream {AUDIO['stream']}")
    if picture != (0, 0, width, height):
        print(f"  picture area {picture[2]}x{picture[3]} at x={picture[0]}, y={picture[1]} (black bars)")
    riff_end = duration   # end credits aren't riffed (or sent to Claude); the YouTube/MP3 versions stop there
    if args.credits != "off":
        credits_at = detect_credits(video, duration, picture) if args.credits == "auto" else parse_clock(args.credits)
        if credits_at and credits_at + args.credits_riff < duration - 5:
            riff_end = credits_at + args.credits_riff     # keep riffing a little way into the credits
            print(f"  end credits from {fmt_ts(credits_at)} ({'detected' if args.credits == 'auto' else 'set'}); "
                  f"riffing stops at {fmt_ts(riff_end)}")

    tmp = Path(tempfile.mkdtemp(prefix="riffify_"))
    try:
        if args.riffs:
            riffs = json.loads(args.riffs.read_text(encoding="utf-8"))["riffs"]
            for r in riffs:
                r["line"] = clean_line(r["line"])
            riffs.sort(key=lambda r: r["time"])
            print(f"[1-3/5] loaded {len(riffs)} riffs from {args.riffs}")
            # dialogue timing (for nudging riffs into pauses) from subtitles or the cached transcript
            subs = None if args.no_subtitles else find_subtitles(video, args.subtitles)
            analysis_path = video.with_name(f"{video.stem}.analysis.json")
            cached = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
            raw_transcript = subs[0] if subs else cached.get("transcript", [])
            transcript = raw_transcript if subs else cap_dialogue(raw_transcript)   # subtitle timings are exact
            # re-apply the timing floors with current timings (shot list from the analysis cache)
            riffs = anchor_riffs(riffs, cached.get("shots", []), transcript, duration, args.visual_delay)
        else:
            # Dialogue, in priority order: English subtitles -> Whisper on GPU -> Whisper on CPU.
            # Whisper results and the shot list are cached next to the video so re-runs skip them.
            analysis_path = video.with_name(f"{video.stem}.analysis.json")
            cached = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
            print("[1/5] dialogue")
            subs = None if args.no_subtitles else find_subtitles(video, args.subtitles)
            if subs:
                transcript, source = subs
                print(f"  using subtitles from {source} ({len(transcript)} lines)")
            elif not has_audio:
                transcript = []
            elif cached.get("whisper_model") == args.whisper_model and "transcript" in cached:
                transcript = cached["transcript"]
                print(f"  using cached Whisper transcript from {analysis_path.name}")
            else:
                if not args.no_subtitles:
                    print("  no English text subtitles found; transcribing")
                transcript = transcribe(video, args.whisper_model, not args.no_gpu, not args.no_cpu)
                if transcript:
                    cached.update(whisper_model=args.whisper_model, transcript=transcript)
            print("[2/5] shots")
            if "shots" in cached and cached.get("shot_threshold") == SHOT_THRESHOLD:
                shots = cached["shots"]
                print(f"  using cached shot list from {analysis_path.name}")
            else:
                shots = detect_shots(video, duration)
                cached.update(shots=shots, shot_threshold=SHOT_THRESHOLD)
            analysis_path.write_text(json.dumps(cached), encoding="utf-8")
            shots = [(s, min(e, riff_end)) for s, e in shots if s < riff_end - 0.5]   # nothing after the credits
            raw_transcript = [line for line in transcript if line[0] < riff_end]
            transcript = raw_transcript if subs else cap_dialogue(raw_transcript)   # subtitle timings are exact
            frames = extract_frames(video, picture, tmp)
            shots_n = [(n, s, e) for n, (s, e) in enumerate(shots, 1)]           # numbered film-wide
            lines_n = [(n, s, e, t) for n, (s, e, t) in enumerate(transcript, 1)]
            pauses = find_pauses(raw_transcript if subs else cap_dialogue(raw_transcript, tight=True), riff_end)
            title = args.title or video.stem.replace("_", " ")
            blocks = plan_blocks(shots_n, riff_end, args.long_after * 60, args.block_minutes * 60)
            print("[3/5] writing riffs" + (f" in {len(blocks)} blocks of about "
                                           f"{riff_end / len(blocks) / 60:.0f} min" if len(blocks) > 1 else ""))
            if len(blocks) == 1:
                sheets, n_frames = build_contact_sheets(frames, shots_n, args.max_frames)
                print(f"  {len(shots)} shots, {n_frames} frames on {len(sheets)} contact sheets")
                riffs = write_riffs(lines_n, shots_n, sheets, (0.0, riff_end), title, args,
                                    out.with_suffix(".candidates.json"), pauses)
            else:
                # Long video: blocks run in order so each can build on the ones before it. The
                # writers' room sees the previous block's riffs; the head writer sees every earlier
                # riff plus (unless --no-gag-sheet) a short gag sheet. Each block is saved as it
                # finishes, so a failed run resumes from the next block.
                riffs, prev, gags = [], [], None
                for k, bshots in enumerate(blocks, 1):
                    t0, t1 = bshots[0][1], bshots[-1][2]
                    print(f"  -- block {k}/{len(blocks)}: {fmt_ts(t0)}-{fmt_ts(t1)}")
                    saved = out.with_suffix(f".block{k}.json")
                    if saved.exists():
                        block_riffs = json.loads(saved.read_text(encoding="utf-8"))["riffs"]
                        print(f"  reusing {len(block_riffs)} riffs from {saved.name}")
                    else:
                        budget = max(TILES_PER_SHEET, round(args.max_frames * (t1 - t0) / 600))
                        sheets, n_frames = build_contact_sheets(frames, bshots, budget)
                        print(f"  {len(bshots)} shots, {n_frames} frames on {len(sheets)} contact sheets")
                        ctx = {"block": (k, len(blocks), riff_end), "prev_riffs": prev,
                               "all_riffs": riffs, "gag_sheet": gags}
                        block_riffs = write_riffs(
                            [x for x in lines_n if t0 <= x[1] < t1], bshots, sheets, (t0, t1), title, args,
                            out.with_suffix(f".block{k}.candidates.json"),
                            [(a, b) for a, b in pauses if b > t0 and a < t1], ctx)
                        saved.write_text(json.dumps({"riffs": block_riffs}, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
                    riffs, prev = riffs + block_riffs, block_riffs
                    if args.gag_sheet and k < len(blocks):
                        gag_file = out.with_suffix(f".block{k}.gags.txt")
                        if gag_file.exists():
                            gags = gag_file.read_text(encoding="utf-8")
                        else:
                            gags = make_gag_sheet(anthropic_client(), riffs, title, args)
                            if gags:
                                gag_file.write_text(gags, encoding="utf-8")
            riffs = anchor_riffs(riffs, shots, transcript, riff_end, args.visual_delay)
            script_path.write_text(json.dumps({"riffs": riffs}, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  saved script -> {script_path}")
            if SPEND["usd"]:
                print(f"  Claude cost this run: ${SPEND['usd']:.2f} "
                      f"(~${SPEND['usd'] / riff_end * 3600:.0f} per hour of video at these settings)")

        print("[4/5] voicing the hecklers")
        riff_wav = tmp / "riffs.wav"
        placed = build_riff_track(riffs, duration, tmp, riff_wav, args.tts, probe_fps(video),
                                  raw_transcript if subs else cap_dialogue(raw_transcript, tight=True))
        timeline = [{"speaker": p["speaker"], "start": round(p["start"], 2), "end": round(p["end"], 2)}
                    for p in placed]
        out.with_suffix(".timeline.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
        gaps = [(a["end"], b["start"]) for a, b in zip(timeline, timeline[1:]) if b["start"] - a["end"] > 25]
        for a, b in gaps:
            print(f"  note: no riffs from {fmt_ts(a)} to {fmt_ts(b)}")

        layers, bob_cmds = None, None
        if args.silhouette:
            layers = draw_layers(width, height, tmp, picture)
            bob_cmds = tmp / "bob_cmds.txt"
            if not write_bob_commands(placed, probe_fps(video), height, bob_cmds):
                bob_cmds = None

        print("[5/5] mixing")
        render(video, riff_wav, layers, bob_cmds, has_audio, duration, out, args)
        print(f"\nDone: {out}  ({len(placed)} riffs)")
        if args.mp3:
            mp3_out = next_free_path(out.with_name(f"{out.stem}_riffs.mp3"))
            export_mp3(riff_wav, min(duration, riff_end + 2), mp3_out, args)
            print(f"Riffs-only MP3: {mp3_out}")
        if args.youtube:
            yt_out = next_free_path(out.with_name(f"{out.stem}_youtube.mp4"))
            render_youtube(video, riff_wav, placed, picture, min(duration, riff_end + 2),
                           args.title or video.stem.replace("_", " "), yt_out, tmp, args)
            print(f"YouTube sync version: {yt_out}")
    finally:
        if args.keep_temp:
            print(f"temp files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
