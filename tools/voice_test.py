#!/usr/bin/env python3
"""Voice test: can one voice both speak and sing the poem? See notes/voice-test.md.

    python3 tools/voice_test.py models     # list audio-output models -> audio/voice-test/models.json
    python3 tools/voice_test.py plan       # the calls `run` would make, with worst-case costs
    python3 tools/voice_test.py selftest   # check the measurements on known signals (no API calls)
    python3 tools/voice_test.py run        # generate, measure, export MP3s, write the report

`run` is resumable: finished renders and transcripts are reused, so it can be re-run after a failure.
Every paid call is checked against the $5 cap first and logged to audio/voice-test/spend.json.
"""

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure as M  # noqa: E402
import openrouter as orr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "audio" / "voice-test"
WAV = OUT / "wav"  # not committed
RAW = OUT / "raw"  # not committed
STT = OUT / "stt"
CAP_USD = 5.00
TAKES = 2
OUT_SR = 44100

# ---------------------------------------------------------------- text and direction


def passage(first_line, n):
    """n lines of the locked poem from `first_line`, as (mode, text). The ♪ marks never leave here."""
    lines = (ROOT / "poem" / "near-enough.md").read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.strip().lstrip("♪").strip() == first_line)
    out = []
    for line in lines[i : i + n]:
        s = line.strip()
        out.append(("sing" if s.startswith("♪") else "speak", s.lstrip("♪").strip()))
    assert all(t for _, t in out), "passage runs past a stanza break"
    return out


PASSAGES = {
    "A": passage("It reads like the first line of something,", 7),
    "B": passage("and I'm a little annoyed with it,", 3),
}

VOICE = (
    "A woman's voice: unhurried, warm, a little wry, with some grain. "
    "Not an audiobook, not breathy, not an announcer."
)
SPOKEN = "She speaks like someone thinking aloud, not reciting."
SUNG = (
    "plain and small, like a half-remembered folk tune hummed into words, with no belting, "
    "no runs and no vibrato. She drifts into the tune and back out of it."
)


def text_of(lines):
    return " ".join(t for _, t in lines)


def blocks(lines):
    """Consecutive runs of one mode: [(mode, [line texts])]."""
    out = []
    for mode, t in lines:
        if out and out[-1][0] == mode:
            out[-1][1].append(t)
        else:
            out.append((mode, [t]))
    return out


def sung_list(lines):
    sung = [f'"{t}"' for m, t in lines if m == "sing"]
    return sung[0] if len(sung) == 1 else ", ".join(sung[:-1]) + " and " + sung[-1]


# ---------------------------------------------------------------- models and prices

GEMINI = "google/gemini-3.8-flash-tts"
SEED = "bytedance-seed/seed-audio-1-0"
ELEVEN = "elevenlabs/eleven-v4"
LYRIA = "google/lyria-3-clip-preview"
FISH = "fish-audio/drama-3"
WHISPER = "openai/whisper-large-v3"

# Voices picked from their published one-word descriptions (Sulafat: "warm"; Lily: "velvety"). Untested.
GEMINI_VOICE = "Sulafat"
ELEVEN_VOICE = "lily"

ROUTE1 = ["r1-gemini", "r1-seed", "r1-eleven"]  # one model performs the whole passage
ROUTE2 = ["r2-lyria", "r2-seed", "r2-fish"]  # Gemini speaks, another model sings, joined
MODEL_OF = {
    "r1-gemini": GEMINI, "r1-seed": SEED, "r1-eleven": ELEVEN,
    "r2-lyria": LYRIA, "r2-seed": SEED, "r2-fish": FISH,
}


def prices():
    data = json.loads((OUT / "models.json").read_text())
    return {m["id"]: m["pricing"] for m in data["models"]}


def cost_fn(model, price):
    """(worst_case_usd, estimate(result) -> usd) for a speech call. Estimates are only a fallback."""
    p, c = float(price.get("prompt", 0)), float(price.get("completion", 0))

    def secs(res):
        return audio_seconds(res["audio"], res["content_type"])

    if model == SEED:  # per second of audio, at most 120 s per request
        return 120 * c, lambda res: secs(res) * c
    if p and c:  # token-priced (Gemini): ~32 audio tokens per second
        return 0.03, lambda res: 200 * p + 32 * secs(res) * c
    return None, None  # per character, filled in by the caller


# ---------------------------------------------------------------- audio helpers


def ffmpeg_decode(audio, content_type, sr=OUT_SR):
    """Bytes from an API -> mono float32 at sr. Raw PCM is read using the rate in the content type."""
    args = []
    if "pcm" in content_type or "L16" in content_type:
        rate = 24000
        for part in content_type.split(";"):
            if part.strip().startswith("rate="):
                rate = int(part.split("=")[1])
        args = ["-f", "s16le", "-ar", str(rate), "-ac", "1"]
    out = subprocess.run(
        ["ffmpeg", "-v", "error", *args, "-i", "pipe:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        input=audio, check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def audio_seconds(audio, content_type):
    return len(ffmpeg_decode(audio, content_type, sr=8000)) / 8000


def wav_bytes(x, sr):
    tmp = RAW / "_tmp.wav"
    M.save_wav(tmp, x, sr)
    return tmp.read_bytes()


def data_uri(x, sr):
    return "data:audio/wav;base64," + base64.b64encode(wav_bytes(x, sr)).decode()


def join(parts, sr=OUT_SR, gap=0.45):
    silence = np.zeros(int(gap * sr), dtype=np.float32)
    out = []
    for k, x in enumerate(parts):
        if k:
            out.append(silence)
        out.append(M.level(M.trim_silence(x, sr), sr))
    return np.concatenate(out)


# ---------------------------------------------------------------- generation


class Generator:
    def __init__(self):
        self.ledger = orr.Ledger(OUT / "spend.json", CAP_USD)
        self.api = orr.Client(self.ledger)
        self.price = prices()
        self.notes = {}  # take id -> list of notes about how it was made

    def speech(self, label, model, **body):
        worst, estimate = cost_fn(model, self.price[model])
        if worst is None:  # per-character price
            chars = len(body["input"] if isinstance(body["input"], str) else "") + len(body.get("instructions") or "")
            per = float(self.price[model]["prompt"])
            worst, estimate = chars * per * 1.5 + 0.005, (lambda res, n=chars: n * per)
        res = self.api.speech(label, worst, estimate, model=model, **body)
        (RAW / f"{label}.bin").write_bytes(res["audio"])
        x = ffmpeg_decode(res["audio"], res["content_type"])
        # A wrong PCM sample rate would show up as an absurd speaking rate.
        n = len(M.words(body["input"] if isinstance(body["input"], str) else ""))
        rate = n / max(0.1, len(x) / OUT_SR)
        if model != SEED and not 0.6 <= rate <= 5:
            self.notes.setdefault(label, []).append(
                f"{rate:.1f} words/s from {res['content_type']!r}: check the sample rate")
        return x

    def transcribe(self, label, x16):
        """Whisper transcript with word timestamps of 16 kHz audio, cached in audio/voice-test/stt."""
        path = STT / f"{label}.json"
        if path.exists():
            return json.loads(path.read_text())
        d = self.api.transcribe(f"stt {label}", wav_bytes(x16, M.SR), model=WHISPER)
        d.pop("units", None)
        path.write_text(json.dumps(d, indent=1) + "\n")
        return d

    # Route 1: one model, the whole passage in one generation.

    def r1(self, rid, P, take):
        lines = PASSAGES[P]
        label = f"{P}-{rid}-t{take}"
        if rid == "r1-gemini":
            # Gemini reads `input` verbatim, so all direction goes in `instructions`.
            return self.speech(
                label, GEMINI, input="\n".join(t for _, t in lines), voice=GEMINI_VOICE, response_format="pcm",
                instructions=f"{VOICE} {SPOKEN} The exception is {sung_list(lines)}, which she sings, {SUNG}",
            )
        if rid == "r1-seed":
            # Seed Audio treats `input` as a prompt describing the performance; no voice is given.
            prompt = (
                f"{VOICE} {SPOKEN} The exception is {sung_list(lines)}, which she sings, {SUNG} "
                f'Her words, exactly:\n"{text_of(lines)}"'
            )
            return self.speech(label, SEED, input=prompt, response_format="mp3")
        if rid == "r1-eleven":
            # Eleven v4 takes inline audio tags; [sings] is ElevenLabs' documented (experimental) tag.
            script = " ".join(
                ("[thoughtful] " if mode == "speak" else "[sings] ") + " ".join(ts) for mode, ts in blocks(lines)
            )
            return self.speech(label, ELEVEN, input=script, voice=ELEVEN_VOICE, response_format="mp3")
        raise ValueError(rid)

    # Route 2: Gemini speaks the spoken blocks; another model sings the sung blocks.

    def spoken_part(self, P, take, k, block):
        path = WAV / "parts" / f"{P}-spoken-t{take}-b{k}.wav"
        if path.exists():
            return M.load(path, OUT_SR)
        x = self.speech(
            path.stem, GEMINI, input=" ".join(block), voice=GEMINI_VOICE, response_format="pcm",
            instructions=f"{VOICE} {SPOKEN}",
        )
        M.save_wav(path, x, OUT_SR)
        return x

    def sung_part(self, rid, P, take, k, block, ref, ref_text):
        path = WAV / "parts" / f"{P}-{rid}-t{take}-b{k}.wav"
        if path.exists():
            return M.load(path, OUT_SR)
        label, text = path.stem, " ".join(block)
        if rid == "r2-lyria":
            x = self.lyria(label, block)
        elif rid == "r2-seed":
            # Voice cloned from this take's spoken audio (one reference clip).
            x = self.speech(
                label, SEED, response_format="mp3",
                input=f'She sings these words quietly, {SUNG} The words, exactly: "{text}"',
                input_references=[{"type": "input_audio", "input_audio": {"data": data_uri(ref, OUT_SR)}}],
            )
        elif rid == "r2-fish":
            # Drama 3 takes plain-language direction in square brackets; voice cloned from the spoken audio.
            x = self.speech(
                label, FISH, response_format="mp3",
                input=f"[singing softly, a plain half-remembered folk tune, no vibrato] {text}",
                input_references=[
                    {"type": "input_audio", "input_audio": {"data": data_uri(ref, OUT_SR)}},
                    {"type": "text", "text": ref_text},
                ],
            )
        else:
            raise ValueError(rid)
        M.save_wav(path, x, OUT_SR)
        return x

    def lyria(self, label, block):
        prompt = (
            "A cappella: one woman singing alone. No instruments, no backing vocals, no percussion, no intro. "
            "Her voice is unhurried, warm, a little wry, with some grain. A plain, small melody like a "
            "half-remembered folk tune, sung quietly. No belting, no runs, no vibrato. "
            "Sing these lyrics once, then stop:\n" + "\n".join(block)
        )
        price = 0.04
        res = self.api.chat_audio(label, 2 * price, lambda _: price, LYRIA, prompt)
        (RAW / f"{label}.json").write_text(json.dumps({"text": res["text"], "stream": res["raw"]}, indent=1))
        if not res["audio"]:
            raise orr.APIError(f"{label}: no audio in the Lyria response (see raw/{label}.json)")
        (RAW / f"{label}.bin").write_bytes(res["audio"])
        clip = ffmpeg_decode(res["audio"], "")
        M.save_wav(WAV / "parts" / f"{label}-full.wav", clip, OUT_SR)
        # The clip is ~30 s; keep the one stretch where the lyrics are sung (found from the transcript).
        d = self.transcribe(f"{label}-full", _resample(clip, OUT_SR, M.SR))
        ws = [(w, float(x["start"]), float(x["end"])) for x in d.get("words", []) for w in M.words(x["word"])]
        errors, a, b = M.fit_span(M.words(" ".join(block)), [w for w, _, _ in ws])
        if b > a and errors <= len(M.words(" ".join(block))) // 2:
            s, e = ws[a][1] - 0.25, ws[b - 1][2] + 0.5
            self.notes.setdefault(label, []).append(f"cut {s:.1f}-{e:.1f} s from the Lyria clip")
            return clip[max(0, int(s * OUT_SR)) : int(e * OUT_SR)]
        self.notes.setdefault(label, []).append("lyrics not found in the Lyria clip; kept the whole clip")
        return clip

    def r2(self, rid, P, take):
        lines = PASSAGES[P]
        bs = blocks(lines)
        spoken = {k: self.spoken_part(P, take, k, ts) for k, (m, ts) in enumerate(bs) if m == "speak"}
        ref = np.concatenate([M.trim_silence(x, OUT_SR) for x in spoken.values()])
        ref_text = " ".join(" ".join(ts) for m, ts in bs if m == "speak")
        parts = [
            spoken[k] if m == "speak" else self.sung_part(rid, P, take, k, ts, ref, ref_text)
            for k, (m, ts) in enumerate(bs)
        ]
        return join(parts)

    def take(self, rid, P, take):
        tid = f"{P}-{rid}-t{take}"
        path = WAV / f"{tid}.wav"
        if path.exists():
            return path
        x = self.r1(rid, P, take) if rid in ROUTE1 else self.r2(rid, P, take)
        if rid in ROUTE1:
            x = M.level(M.trim_silence(x, OUT_SR), OUT_SR)
        M.save_wav(path, x, OUT_SR)
        return path


def _resample(x, sr_in, sr_out):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr_in), "-ac", "1", "-i", "pipe:0",
         "-ar", str(sr_out), "-f", "f32le", "-"],
        input=x.astype(np.float32).tobytes(), check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


# ---------------------------------------------------------------- measurement


def measure_take(gen, tid, P):
    lines = PASSAGES[P]
    x = M.load(WAV / f"{tid}.wav")  # 16 kHz
    d = gen.transcribe(tid, x)
    stt_words = d.get("words") or d.get("segments") or []
    spans = M.line_times([t for _, t in lines], stt_words)
    per_line, pooled = [], {"speak": [], "sing": []}
    for (mode, text), span in zip(lines, spans):
        row = {"mode": mode, "text": text, "span_s": None}
        if span:
            a, b = int(span[0] * M.SR), int(span[1] * M.SR)
            seg = x[a:b]
            f0, _ = M.yin(seg)
            voiced = ~np.isnan(f0)
            h = M.held_fraction(f0)
            row.update(
                span_s=[round(span[0], 2), round(span[1], 2)],
                voiced_s=round(float(voiced.sum()) / 100, 2),
                median_f0_hz=round(float(np.nanmedian(f0)), 1) if voiced.any() else None,
                held=None if np.isnan(h) else round(h, 3),
            )
            pooled[mode].append((seg, f0))
        per_line.append(row)

    def pool(mode):
        """Held fraction over all lines of one mode (weighted by voiced frames), and their audio."""
        held = voiced = 0
        for _, f0 in pooled[mode]:
            v, h = int((~np.isnan(f0)).sum()), M.held_fraction(f0)
            if not np.isnan(h):
                held, voiced = held + h * v, voiced + v
        if voiced < 10:
            return None, None
        return round(held / voiced, 3), np.concatenate([M.trim_silence(seg) for seg, _ in pooled[mode]])

    held_spoken, spoken_audio = pool("speak")
    held_sung, sung_audio = pool("sing")
    sim = None
    if spoken_audio is not None and sung_audio is not None:
        sim = M.similarity(spoken_audio, sung_audio)
    return {
        "take": tid,
        "passage": P,
        "duration_s": round(len(x) / M.SR, 2),
        "transcript": d.get("text", "").strip(),
        "wer": round(M.wer(text_of(lines), d.get("text", "")), 3),
        "held_spoken": held_spoken,
        "held_sung": held_sung,
        "sung_seconds": round(len(sung_audio) / M.SR, 2) if sung_audio is not None else 0,
        "similarity_spoken_vs_sung": sim,
        "lines": per_line,
    }


def baselines(results):
    """Reference points for the similarity scale, from spoken audio only."""

    def spoken(tid):
        r = next((r for r in results if r["take"] == tid), None)
        if not r:
            return None
        x = M.load(WAV / f"{tid}.wav")
        segs = [x[int(ln["span_s"][0] * M.SR) : int(ln["span_s"][1] * M.SR)]
                for ln in r["lines"] if ln["mode"] == "speak" and ln["span_s"]]
        return np.concatenate(segs) if segs else None

    out = {}
    pairs = {
        "same voice, two generations (Gemini A t1 vs t2, spoken)": ("A-r1-gemini-t1", "A-r1-gemini-t2"),
        "different voices (Gemini vs Eleven, A t1, spoken)": ("A-r1-gemini-t1", "A-r1-eleven-t1"),
    }
    for name, (a, b) in pairs.items():
        xa, xb = spoken(a), spoken(b)
        out[name] = M.similarity(xa, xb) if xa is not None and xb is not None else None
    return out


# ---------------------------------------------------------------- export and report


def export_mp3(tid):
    src, dst = WAV / f"{tid}.wav", OUT / f"{tid}.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "44100",
                    "-codec:a", "libmp3lame", "-b:a", "96k", str(dst)], check=True)
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(dst)],
        check=True, capture_output=True, text=True).stdout)
    s = probe["streams"][0]
    ok = s["codec_name"] == "mp3" and int(s["channels"]) == 1 and float(probe["format"]["duration"]) > 0.5
    decode = subprocess.run(["ffmpeg", "-v", "error", "-i", str(dst), "-f", "null", "-"],
                            capture_output=True, text=True)
    if not ok or decode.stderr.strip():
        raise RuntimeError(f"{dst.name} failed the ffprobe check: {s} {decode.stderr[:200]}")
    return {"file": dst.name, "kbytes": dst.stat().st_size // 1024, "duration_s": float(probe["format"]["duration"])}


def pct(v):
    return "n/a" if v is None else f"{100 * v:.0f}%"


def describe(tid):
    rid = tid.split("-", 1)[1].rsplit("-t", 1)[0]
    if rid in ROUTE1:
        return f"{MODEL_OF[rid]}, one take"
    return f"{GEMINI} speaks, {MODEL_OF[rid]} sings" + (" (voice cloned from the spoken part)" if rid != "r2-lyria" else "")


def write_report(results, base, failures, ledger):
    lines = [
        "# Voice test results",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `tools/voice_test.py run`. "
        "See `notes/voice-test.md` for the plan.*",
        "",
        "These numbers are proxies, not verdicts. Chris's ears decide.",
        "",
        "- **WER**: word error rate of a Whisper large-v3 transcript against the passage.",
        "- **Held**: share of voiced frames inside a held note (at least 150 ms within ±50 cents). "
        "Sung lines should be high, spoken lines low. Pitch is plain YIN, not librosa's pYIN.",
        "- **Similarity**: cosine similarity of WeSpeaker speaker embeddings, spoken lines vs sung "
        "lines of the same take. Singing lowers this even for one real person, so compare takes with "
        "each other and with the baselines below, not with 1.0. Under ~2 s of singing it is noisy.",
        "",
        "| Take | How | WER | Held, spoken | Held, sung | Similarity | Length |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| [{r['take']}]({r['take']}.mp3) | {describe(r['take'])} | {pct(r['wer'])} | "
            f"{pct(r['held_spoken'])} | {pct(r['held_sung'])} | "
            f"{'n/a' if r['similarity_spoken_vs_sung'] is None else r['similarity_spoken_vs_sung']}"
            f"{' (short)' if 0 < r['sung_seconds'] < 2 else ''} | {r['duration_s']:.0f} s |"
        )
    lines += ["", "**Similarity baselines** (spoken audio only):", ""]
    lines += [f"- {k}: {'n/a' if v is None else v}" for k, v in base.items()]
    if failures:
        lines += ["", "**Takes that failed:**", ""] + [f"- {t}: {e}" for t, e in failures.items()]
    lines += ["", f"Spend: ${ledger.spent:.2f} of the ${ledger.cap:.2f} cap (`spend.json`).", ""]
    (OUT / "README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------- commands


def cmd_models():
    found = []
    for m in orr.models("all"):
        mods = m.get("architecture", {}).get("output_modalities") or []
        if "audio" not in mods and "speech" not in mods:
            continue
        eps = orr.endpoints(m["id"])
        found.append({
            "id": m["id"],
            "name": m.get("name"),
            "output_modalities": mods,
            "pricing": m.get("pricing"),
            "price_unit": "per second of audio" if m["id"] == SEED else
                          "per song/clip (see description)" if "lyria" in m["id"] else
                          "per token" if float(m["pricing"].get("completion", 0)) else "per character",
            "api_endpoints": m.get("endpoints"),
            "voices": m.get("supported_voices"),
            "providers": [{
                "provider": e["provider_name"],
                "voice_cloning": e.get("supports_voice_cloning"),
                "multiple_audio_references": e.get("supports_multiple_audio_references"),
                "uptime_last_1d": e.get("uptime_last_1d"),
            } for e in eps],
            "description": m.get("description"),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {
        "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "GET https://openrouter.ai/api/v1/models?output_modalities=all, filtered to audio or speech output",
        "models": found,
    }
    (OUT / "models.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(found)} models with audio or speech output -> {OUT / 'models.json'}")


def cmd_plan():
    price = prices()
    missing = sorted(m for m in set(MODEL_OF.values()) if m not in price)
    if missing:
        sys.exit(f"not in models.json any more: {missing}")
    total, rows = 0.0, []
    for P, lines in PASSAGES.items():
        n_spoken = sum(m == "speak" for m, _ in blocks(lines))
        n_sung = sum(m == "sing" for m, _ in blocks(lines))
        chars = len(text_of(lines)) + 400
        for take in range(1, TAKES + 1):
            rows += [(f"{P}-r1-gemini-t{take}", 0.03), (f"{P}-r1-seed-t{take}", 0.30),
                     (f"{P}-r1-eleven-t{take}", chars * float(price[ELEVEN]["prompt"]) * 1.5 + 0.005)]
            rows += [(f"{P}-spoken-t{take} x{n_spoken}", 0.03 * n_spoken),
                     (f"{P}-r2-lyria-t{take} x{n_sung} (+ transcribe)", 0.09 * n_sung),
                     (f"{P}-r2-seed-t{take} x{n_sung}", 0.30 * n_sung),
                     (f"{P}-r2-fish-t{take} x{n_sung}", 0.01 * n_sung)]
    rows.append(("transcripts, 24 takes", 24 * 0.01))
    for label, usd in rows:
        total += usd
        print(f"  {label:40s} worst ${usd:.3f}")
    print(f"worst case total ${total:.2f} (cap ${CAP_USD:.2f}); each call is also checked against the cap")


def cmd_selftest():
    sr, ok = M.SR, True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")

    t = np.arange(int(sr * 1.2)) / sr

    def tone(f):
        ph = 2 * np.pi * np.cumsum(f) / sr
        return 0.1 * sum(np.sin(k * ph) / k for k in range(1, 7))

    f0, _ = M.yin(tone(np.full_like(t, 220.0)))
    check("steady 220 Hz tone", M.held_fraction(f0) > 0.9 and abs(np.nanmedian(f0) - 220) < 2,
          f"held {M.held_fraction(f0):.2f}, median {np.nanmedian(f0):.1f} Hz")
    f0, _ = M.yin(tone(np.geomspace(150, 300, len(t))))
    check("octave glide over 1.2 s", M.held_fraction(f0) < 0.1, f"held {M.held_fraction(f0):.2f}")
    n = int(sr * 0.4)
    vib = 2 ** (20 / 1200 * np.sin(2 * np.pi * 5 * np.arange(n) / sr))
    f0, _ = M.yin(np.concatenate([tone(np.full(n, f) * vib) for f in (196, 220, 247)]))
    check("three notes with ±20 cent vibrato", M.held_fraction(f0) > 0.8, f"held {M.held_fraction(f0):.2f}")
    f0, _ = M.yin(np.random.default_rng(0).normal(0, 0.05, len(t)))
    check("white noise", np.mean(~np.isnan(f0)) < 0.05, f"voiced {np.mean(~np.isnan(f0)):.2f}")
    check("WER", M.wer("It fits. That's all it does.", "it fits that's all it does") == 0
          and abs(M.wer("and I'm looking.", "and looking") - 1 / 3) < 1e-9, "punctuation ignored, deletion counted")

    x = M.load(M.fetch(M.TEST_SPEECH))
    s = M.pitch_summary(x)
    check("real speech (Mandarin, 4 speakers) is mostly not held", s["held"] < 0.4, f"held {s['held']}")
    seg = x[sr : 7 * sr]
    noisy = seg + np.random.default_rng(1).normal(0, 0.01, len(seg)).astype(np.float32)
    shifted = np.interp(np.arange(0, len(seg) - 1, 2 ** 0.5), np.arange(len(seg)), seg).astype(np.float32)
    a, b = M.similarity(seg, noisy), M.similarity(seg, shifted)
    check("speaker: same speech plus noise", a > 0.7, f"{a}")
    check("speaker: same speech sped up 6 semitones (formants move)", b < 0.4, f"{b}")
    sims, prev = [], None
    for k in range(0, len(x) - 2 * sr, sr):
        e = M.embed(x[k : k + 2 * sr]) if M.active_rms_db(x[k : k + 2 * sr]) > -40 else None
        if e is not None and prev is not None:
            sims.append(float(e @ prev))
        prev = e
    check("speaker: adjacent 2 s windows mostly the same speaker",
          np.median(sims) > 0.5 and min(sims) < 0.2,
          f"median {np.median(sims):.2f}, min {min(sims):.2f} (the dips are speaker changes)")
    print("all passed" if ok else "SOME CHECKS FAILED")
    return ok


def cmd_run():
    for d in (WAV / "parts", RAW, STT):
        d.mkdir(parents=True, exist_ok=True)
    gen = Generator()
    todo = [f"{P}-{rid}-t{k}" for rid in ROUTE1 + ROUTE2 for P in PASSAGES for k in range(1, TAKES + 1)]
    if any(not (WAV / f"{t}.wav").exists() or not (STT / f"{t}.json").exists() for t in todo):
        try:
            orr.check_auth()
        except orr.APIError as e:
            sys.exit(str(e))
    takes, failures = [], {}
    for rid in ROUTE1 + ROUTE2:
        for P in PASSAGES:
            for take in range(1, TAKES + 1):
                tid = f"{P}-{rid}-t{take}"
                try:
                    gen.take(rid, P, take)
                    takes.append((tid, P))
                    print(f"made {tid}  (spent ${gen.ledger.spent:.3f})")
                except orr.BudgetExceeded:
                    raise
                except Exception as e:  # one model failing shouldn't stop the others
                    failures[tid] = str(e)[:300]
                    print(f"FAILED {tid}: {str(e)[:300]}")
    results = []
    for tid, P in takes:
        try:
            r = measure_take(gen, tid, P)
            r["mp3"] = export_mp3(tid)
            r["notes"] = sum((v for k, v in gen.notes.items() if k == tid or k.startswith(tid + "-")), [])
            results.append(r)
        except orr.BudgetExceeded:
            raise
        except Exception as e:
            failures[tid] = f"measurement: {str(e)[:300]}"
    base = baselines(results)
    doc = {"takes": results, "baselines": base, "failures": failures, "generation_notes": gen.notes}
    (OUT / "results.json").write_text(json.dumps(doc, indent=1) + "\n")
    write_report(results, base, failures, gen.ledger)
    print()
    for r in results:
        print(f"{r['take']}: {describe(r['take'])}. WER {pct(r['wer'])}, held spoken {pct(r['held_spoken'])} "
              f"/ sung {pct(r['held_sung'])}, similarity {r['similarity_spoken_vs_sung']}")
    for label, notes in gen.notes.items():
        print(f"note {label}: {'; '.join(notes)}")
    print(f"\nspent ${gen.ledger.spent:.2f}; report in {OUT / 'README.md'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    commands = {"models": cmd_models, "plan": cmd_plan, "selftest": cmd_selftest, "run": cmd_run}
    if cmd not in commands:
        sys.exit(__doc__)
    result = commands[cmd]()
    sys.exit(0 if result is not False else 1)
