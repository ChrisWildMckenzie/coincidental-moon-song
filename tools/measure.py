"""Measurements for voice takes. They are proxies for Chris's ears, not verdicts.

librosa and resemblyzer could not be installed (PyPI is blocked in the cloud environment), so:

- Pitch is plain YIN (de Cheveigne & Kawahara, 2002) in numpy with an energy gate, not librosa's pYIN.
- Speaker similarity uses the WeSpeaker ResNet34 VoxCeleb model (ONNX, from the sherpa-onnx releases)
  on Kaldi-style 80-bin fbank features computed here, not resemblyzer. Run `voice_test.py selftest`
  to check both against known signals.

Everything works on 16 kHz mono float32 arrays.
"""

import hashlib
import re
import subprocess
import warnings
import wave
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SR = 16000
HOP = 160  # 10 ms
CACHE = Path(__file__).resolve().parent / ".cache"

SPEAKER_MODEL = {
    "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    "wespeaker_en_voxceleb_resnet34.onnx",
    "sha256": "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94",
}
# Real conversational speech with four speakers, used only by the self-test.
TEST_SPEECH = {
    "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
    "0-four-speakers-zh.wav",
    "sha256": "bedf036caed208386c67b4ef4b11f83d74dd0d420b102163a1c33cd09cde7010",
}


# ---------------------------------------------------------------- audio I/O


def load(path, sr=SR):
    """Decode any audio file to mono float32 at `sr` with ffmpeg."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def save_wav(path, x, sr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def fetch(spec):
    """Download a pinned file into tools/.cache once, checking its sha256."""
    import requests

    CACHE.mkdir(exist_ok=True)
    path = CACHE / spec["url"].rsplit("/", 1)[1]
    if not path.exists():
        r = requests.get(spec["url"], timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        path.unlink()
        raise RuntimeError(f"{path.name}: sha256 mismatch ({digest})")
    return path


def active_rms_db(x, sr=SR):
    """RMS level (dBFS) of the frames within 35 dB of the loudest, i.e. ignoring silences."""
    n = len(x) // (sr // 100)
    if n == 0:
        return -120.0
    frames = x[: n * (sr // 100)].reshape(n, -1)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    keep = db > db.max() - 35
    return float(10 * np.log10(np.mean(frames[keep] ** 2) + 1e-12))


def level(x, sr=SR, target_db=-23.0, peak_db=-1.0):
    """Scale to a target active-speech RMS, then pull down if the peak would clip."""
    gain = 10 ** ((target_db - active_rms_db(x, sr)) / 20)
    y = x * gain
    peak = np.max(np.abs(y)) if len(y) else 0
    limit = 10 ** (peak_db / 20)
    return y * (limit / peak) if peak > limit else y


def trim_silence(x, sr=SR, floor_db=40, pad=0.05):
    n = len(x) // (sr // 100)
    if n == 0:
        return x
    frames = x[: n * (sr // 100)].reshape(n, -1)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    on = np.flatnonzero(db > db.max() - floor_db)
    if len(on) == 0:
        return x[:0]
    a = max(0, int((on[0] / 100 - pad) * sr))
    b = min(len(x), int(((on[-1] + 1) / 100 + pad) * sr))
    return x[a:b]


# ---------------------------------------------------------------- words


def words(text):
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def align(ref, hyp):
    """Levenshtein alignment. Returns (errors, pairs) where pairs are (ref_i or None, hyp_j or None)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return int(d[n, m]), pairs[::-1]


def fit_span(ref, hyp):
    """Best-matching stretch of `hyp` for the whole of `ref` (free start and end in `hyp`).

    Returns (errors, j_start, j_end) with j_end exclusive. Used to find one sung occurrence of the
    lyrics inside a longer music-model clip.
    """
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    end = int(np.argmin(d[n]))
    i, j = n, end
    while i:
        if j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return int(d[n, end]), j, end


def wer(ref_text, hyp_text):
    ref = words(ref_text)
    errors, _ = align(ref, words(hyp_text))
    return errors / max(1, len(ref))


def line_times(lines, stt_words):
    """Times for each reference line from word timestamps, by aligning transcript words to the text.

    `stt_words` is the transcriber's list of {"word", "start", "end"}. A line's span runs from its
    first to its last reference word that has an aligned transcript word (matches and substitutions
    both count). Lines with no aligned words get None.
    """
    ref, owner = [], []
    for k, line in enumerate(lines):
        for w in words(line):
            ref.append(w)
            owner.append(k)
    hyp, times = [], []
    for w in stt_words:
        for piece in words(w["word"]):
            hyp.append(piece)
            times.append((float(w["start"]), float(w["end"])))
    _, pairs = align(ref, hyp)
    spans = [None] * len(lines)
    for i, j in pairs:
        if i is None or j is None:
            continue
        k = owner[i]
        s, e = times[j]
        spans[k] = (s, e) if spans[k] is None else (min(spans[k][0], s), max(spans[k][1], e))
    return spans


# ---------------------------------------------------------------- pitch


def yin(x, sr=SR, fmin=70.0, fmax=900.0, win=480, hop=HOP, threshold=0.15, gate_db=35.0):
    """F0 per 10 ms frame in Hz (nan where unvoiced) and the YIN aperiodicity of each frame."""
    tmin, tmax = int(sr / fmax), int(np.ceil(sr / fmin))
    span = win + tmax
    x = np.concatenate([x.astype(np.float64), np.zeros(span)])
    frames = sliding_window_view(x, span)[::hop]
    frames = frames[: max(0, (len(x) - span) // hop + 1)]
    if len(frames) == 0:
        return np.array([]), np.array([])

    nfft = 1 << int(np.ceil(np.log2(win + span)))
    a = np.fft.rfft(frames[:, :win], nfft)
    b = np.fft.rfft(frames, nfft)
    corr = np.fft.irfft(np.conj(a) * b, nfft)[:, : tmax + 1]
    cs = np.concatenate([np.zeros((len(frames), 1)), np.cumsum(frames**2, axis=1)], axis=1)
    e1 = cs[:, win : win + 1]
    e2 = cs[:, win : win + tmax + 1] - cs[:, : tmax + 1]
    diff = np.maximum(e1 + e2 - 2 * corr, 0)
    diff[:, 0] = 0
    running = np.cumsum(diff[:, 1:], axis=1)
    cmnd = np.ones_like(diff)
    cmnd[:, 1:] = diff[:, 1:] * np.arange(1, tmax + 1) / np.maximum(running, 1e-12)
    cmnd[:, :tmin] = 1.0

    rows = np.arange(len(frames))
    below = cmnd < threshold
    has = below.any(axis=1)
    tau = np.where(has, below.argmax(axis=1), np.argmin(cmnd[:, tmin:], axis=1) + tmin)
    for _ in range(tmax):  # walk down to the local minimum
        step = (tau < tmax) & (cmnd[rows, np.minimum(tau + 1, tmax)] < cmnd[rows, tau])
        if not step.any():
            break
        tau = tau + step
    aper = cmnd[rows, tau]

    # Parabolic interpolation around the chosen lag.
    t0, t2 = np.maximum(tau - 1, 0), np.minimum(tau + 1, tmax)
    y0, y1, y2 = cmnd[rows, t0], cmnd[rows, tau], cmnd[rows, t2]
    den = y0 - 2 * y1 + y2
    shift = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0)
    f0 = sr / (tau + np.clip(shift, -1, 1))

    energy = 10 * np.log10(np.mean(frames[:, :win] ** 2, axis=1) + 1e-12)
    voiced = (aper < 0.3) & (energy > energy.max() - gate_db) & (energy > -60)
    f0 = np.where(voiced, f0, np.nan)

    # Three-frame median to remove isolated octave slips.
    padded = np.concatenate([[np.nan], f0, [np.nan]])
    win3 = sliding_window_view(padded, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(win3, axis=1)
    f0 = np.where(np.isnan(f0), np.nan, med)
    return f0, aper


def held_fraction(f0, min_dur=0.15, tol_cents=50, hop_s=HOP / SR):
    """Fraction of voiced frames that sit inside a held note.

    A held note is a stretch of at least `min_dur` seconds, voiced throughout, whose pitch stays
    within +/- `tol_cents` of its middle. Sung notes should mostly count; speech glides, so
    should mostly not.
    """
    voiced = ~np.isnan(f0)
    if voiced.sum() < 10:
        return float("nan")
    cents = 1200 * np.log2(np.where(voiced, f0, 1.0) / 55.0)
    cents[~voiced] = np.nan
    L = max(2, int(round(min_dur / hop_s)))
    if len(cents) < L:
        return 0.0
    w = sliding_window_view(cents, L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ok = ~np.isnan(w).any(axis=1) & ((np.nanmax(w, axis=1) - np.nanmin(w, axis=1)) <= 2 * tol_cents)
    held = np.convolve(ok.astype(int), np.ones(L, dtype=int))[: len(cents)] > 0
    return float(held[voiced].mean())


def pitch_summary(x, sr=SR):
    f0, _ = yin(x, sr)
    voiced = ~np.isnan(f0)
    return {
        "voiced_s": round(float(voiced.sum()) * HOP / sr, 2),
        "median_f0_hz": round(float(np.nanmedian(f0)), 1) if voiced.any() else None,
        "held": None if np.isnan(h := held_fraction(f0)) else round(h, 3),
    }


# ---------------------------------------------------------------- speaker


def fbank(x, sr=SR, n_mels=80):
    """Kaldi-style log mel filterbank (25 ms Hamming, 10 ms hop, preemphasis 0.97), mean-normalised."""
    x = x.astype(np.float64) * 32768.0
    win, hop, nfft = 400, 160, 512
    if len(x) < win:
        return np.zeros((0, n_mels), dtype=np.float32)
    frames = sliding_window_view(x, win)[::hop].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1]
    frames[:, 0] -= 0.97 * frames[:, 0]
    frames *= np.hamming(win)
    power = np.abs(np.fft.rfft(frames, nfft)) ** 2

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    lo, hi = mel(20.0), mel(sr / 2)
    centres = lo + (hi - lo) * np.arange(n_mels + 2) / (n_mels + 1)
    fft_mel = mel(np.arange(nfft // 2 + 1) * sr / nfft)
    banks = np.zeros((n_mels, nfft // 2 + 1))
    for m in range(n_mels):
        l, c, r = centres[m], centres[m + 1], centres[m + 2]
        up = (fft_mel - l) / (c - l)
        down = (r - fft_mel) / (r - c)
        banks[m] = np.maximum(0, np.minimum(up, down))
    banks[:, -1] = 0  # Kaldi leaves out the Nyquist bin
    feats = np.log(np.maximum(power @ banks.T, np.finfo(np.float32).eps))
    feats -= feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float32)


_session = None


def embed(x, sr=SR):
    """L2-normalised 256-d speaker embedding of the non-silent part of `x`."""
    global _session
    if _session is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _session = ort.InferenceSession(str(fetch(SPEAKER_MODEL)), opts, providers=["CPUExecutionProvider"])
    feats = fbank(trim_silence(x, sr), sr)
    if len(feats) < 50:
        return None
    e = _session.run(None, {"feats": feats[None]})[0][0]
    return e / np.linalg.norm(e)


def similarity(a, b, sr=SR):
    ea, eb = embed(a, sr), embed(b, sr)
    if ea is None or eb is None:
        return None
    return round(float(ea @ eb), 3)
