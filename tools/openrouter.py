"""Small OpenRouter client. Key from the environment, every paid call logged with its cost.

The key is read from OPENROUTER_API_KEY and only ever goes into the Authorization header. It is
never printed, logged or written to disk.
"""

import base64
import json
import os
import time
from pathlib import Path

import requests

BASE = "https://openrouter.ai/api/v1"


class APIError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def _headers():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise APIError("OPENROUTER_API_KEY is not set")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": "coincidental-moon-song",
    }


def check_auth():
    """One free authenticated call, so a bad key fails once instead of on every take."""
    r = requests.get(f"{BASE}/key", headers=_headers(), timeout=30)
    if r.status_code != 200:
        proxy = r.headers.get("X-Proxy-Error")
        raise APIError(f"OpenRouter auth failed: HTTP {r.status_code}: {r.text[:200]}"
                       + (f" [proxy: {proxy}]" if proxy else ""))


def models(output_modalities="all"):
    r = requests.get(f"{BASE}/models", params={"output_modalities": output_modalities}, timeout=60)
    r.raise_for_status()
    return r.json()["data"]


def endpoints(model_id):
    r = requests.get(f"{BASE}/models/{model_id}/endpoints", timeout=60)
    r.raise_for_status()
    return r.json()["data"]["endpoints"]


class Ledger:
    """Spend log in a JSON file, with a hard cap checked against each call's worst case."""

    def __init__(self, path, cap_usd):
        self.path = Path(path)
        self.cap = cap_usd
        self.calls = json.loads(self.path.read_text())["calls"] if self.path.exists() else []

    @property
    def spent(self):
        return sum(c["cost_usd"] for c in self.calls)

    def check(self, worst_case, label):
        if self.spent + worst_case > self.cap:
            raise BudgetExceeded(
                f"{label}: worst case ${worst_case:.3f} on top of ${self.spent:.3f} spent "
                f"would pass the ${self.cap:.2f} cap"
            )

    def record(self, **call):
        call["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.calls.append(call)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        summary = {"cap_usd": self.cap, "spent_usd": round(self.spent, 6), "calls": self.calls}
        self.path.write_text(json.dumps(summary, indent=1) + "\n")


class Client:
    def __init__(self, ledger, timeout=300):
        self.ledger = ledger
        self.timeout = timeout

    def _post(self, path, body, stream=False):
        r = requests.post(BASE + path, headers=_headers(), json=body, timeout=self.timeout, stream=stream)
        if r.status_code != 200:
            proxy = r.headers.get("X-Proxy-Error")
            raise APIError(
                f"{path} {body.get('model')}: HTTP {r.status_code}: {r.text[:400]}"
                + (f" [proxy: {proxy}]" if proxy else "")
            )
        return r

    def generation_cost(self, gen_id, tries=6):
        """Actual cost of a call, from OpenRouter's generation record (it can take a moment to appear)."""
        for i in range(tries):
            try:
                r = requests.get(f"{BASE}/generation", params={"id": gen_id}, headers=_headers(), timeout=30)
                if r.status_code == 200:
                    cost = (r.json().get("data") or {}).get("total_cost")
                    if cost is not None:
                        return float(cost)
            except requests.RequestException:
                pass
            time.sleep(1.5 * (i + 1))
        return None

    def _paid(self, label, model, endpoint, worst_case, estimate, units, call):
        """Run `call` under the cap, then log it. `estimate(result)` is the fallback cost."""
        self.ledger.check(worst_case, label)
        try:
            result, gen_id, actual = call()
        except Exception as e:
            self.ledger.record(
                label=label, model=model, endpoint=endpoint, **units, cost_usd=0.0,
                cost_source="failed, not billed", error=str(e)[:300],
            )
            raise
        if actual is None and gen_id:
            actual = self.generation_cost(gen_id)
        cost, source = (actual, "openrouter") if actual is not None else (estimate(result), "estimate")
        self.ledger.record(
            label=label, model=model, endpoint=endpoint, **units, **result.get("units", {}),
            cost_usd=round(cost, 6), cost_source=source, generation_id=gen_id,
            provider=result.get("provider"),
        )
        return result

    def speech(self, label, worst_case, estimate, **body):
        """POST /audio/speech. Returns {"audio": bytes, "content_type": str, ...}."""
        body = {k: v for k, v in body.items() if v is not None}
        text = body["input"] if isinstance(body["input"], str) else " ".join(t["text"] for t in body["input"])
        chars = len(text) + len(body.get("instructions") or "")

        def call():
            r = self._post("/audio/speech", body)
            res = {"audio": r.content, "content_type": r.headers.get("Content-Type", ""),
                   "provider": r.headers.get("X-Provider-Name")}
            return res, r.headers.get("X-Generation-Id"), None

        return self._paid(label, body["model"], "audio/speech", worst_case, estimate, {"chars": chars}, call)

    def chat_audio(self, label, worst_case, estimate, model, prompt, audio=None):
        """Chat completion with audio output (streamed). Returns {"audio": bytes, "text": str, "raw": [...]}.

        The response shape for music models is not documented, so this collects any base64 audio it
        finds in the stream and keeps the chunks (payloads elided) for inspection.
        """
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["text", "audio"],
            "stream": True,
            "usage": {"include": True},
        }
        if audio:
            body["audio"] = audio

        def call():
            r = self._post("/chat/completions", body, stream=True)
            pieces, text, raw, cost, gen_id = [], [], [], None, None
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                gen_id = gen_id or chunk.get("id")
                if chunk.get("usage", {}).get("cost") is not None:
                    cost = float(chunk["usage"]["cost"])
                for choice in chunk.get("choices", []):
                    for part in (choice.get("delta") or {}, choice.get("message") or {}):
                        if isinstance(part.get("content"), str):
                            text.append(part["content"])
                        _collect_audio(part, pieces)
                raw.append(_elide(chunk))
            res = {"audio": b"".join(pieces), "text": "".join(text), "raw": raw}
            return res, gen_id, cost

        return self._paid(label, model, "chat/completions", worst_case, estimate, {"chars": len(prompt)}, call)

    def transcribe(self, label, wav_bytes, model="openai/whisper-large-v3", worst_case=0.01):
        """POST /audio/transcriptions with word timestamps. Returns the verbose JSON."""
        body = {
            "model": model,
            "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"},
            "language": "en",
            "temperature": 0,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment", "word"],
        }

        def call():
            d = self._post("/audio/transcriptions", body).json()
            d["units"] = {"seconds": (d.get("usage") or {}).get("seconds")}
            return d, None, (d.get("usage") or {}).get("cost")

        return self._paid(label, model, "audio/transcriptions", worst_case, lambda _: worst_case, {}, call)


def _collect_audio(obj, out):
    """Append decoded base64 audio from any {"audio": {"data": ...}}-like or input/output_audio part."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in ("audio", "output_audio", "input_audio") and isinstance(val, dict):
                data = val.get("data")
                if isinstance(data, str) and data:
                    if data.startswith("data:"):
                        data = data.split(",", 1)[1]
                    out.append(base64.b64decode(data))
            elif isinstance(val, (dict, list)):
                _collect_audio(val, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_audio(v, out)


def _elide(obj):
    if isinstance(obj, dict):
        return {k: (f"<{len(v)} chars>" if k == "data" and isinstance(v, str) else _elide(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_elide(v) for v in obj]
    return obj
