# Voice test

*Written 2026-10-31, to run in a fresh session once OpenRouter is set up. Read `poem/near-enough.md` and
`notes/what-i-think.md` first.*

## Status (2026-10-31, first attempt)

**Not run yet.** `openrouter.ai` is now allowed, but every authenticated call returns 401: the
environment's proxy reports "upstream auth failed" for its openrouter.ai credential, so the stored key
isn't reaching OpenRouter. The public model list works. Nothing has been spent.

Everything else is ready. Once the key works, in a new session:

```
python3 tools/voice_test.py run
```

It checks the key first, is resumable, stops before the cap, and writes the MP3s, `results.json` and a
report (`audio/voice-test/README.md`).

- **Step 1 is done.** `audio/voice-test/models.json` lists the 42 models with audio or speech output (38
  TTS, GPT Audio and GPT Audio Mini, Lyria 3 Pro and Clip), with prices, voices and which providers
  accept a reference clip for voice cloning.
- **Route 1** (one model, one take): Gemini 3.8 Flash TTS (direction in `instructions`, naming the
  lines to sing), Seed Audio 1.0 (its input is a prompt describing the performance) and Eleven v4 (the
  inline `[sings]` tag).
- **Route 2** (two models): Gemini 3.8 Flash TTS speaks. The sung lines come from Lyria 3 Clip (an a
  cappella prompt, with the lyric stretch cut from the clip using its transcript), Seed Audio 1.0 or Fish
  Audio Drama 3. The last two clone the voice from that take's spoken audio, which tests the
  voice-conversion question without setting up Seed-VC.
- **Cost:** the worst case for the whole test is $3.62 (`voice_test.py plan`); the expected cost is
  well under $1.
- **Untested choices:** the voices (Gemini "Sulafat", Eleven "lily") were picked from their one-word
  descriptions.

**Changes from the plan.** PyPI is blocked in this environment, so there's no librosa or resemblyzer.
Pitch is plain YIN in numpy (`tools/measure.py`). Speaker similarity uses the WeSpeaker ResNet34 model on
the onnxruntime that's already installed. `python3 tools/voice_test.py selftest` checks both against
synthetic tones and real speech, and passes. Because singing lowers speaker similarity even for one real
person, the report adds two baselines: the same voice across two generations, and two different voices.

## Why this comes first

The poem is spoken, with a few sung lines. It's **one voice in two modes**. She speaks when she's
cautious and checking ("I can't check the ones that satisfy"). She sings when she's hoping and dreaming
("It's the sky, rhyming"). The sung mode is not a second persona, and it isn't self-deception.

Speech models and music models usually give you different voices, so the hard part of the whole audio
is a single voice that both speaks and sings convincingly. Nothing else (music, timing, pictures) should
start until we know which route works. I can't hear, so Chris's ears decide, and measurements only
narrow the field.

## Before running

- `openrouter.ai` must be on the environment's allowed domains. (It was refused on 2026-10-31.)
- `OPENROUTER_API_KEY` must be set in the environment. Read it from the environment only. Never print
  it, log it or commit it.
- Python packages from PyPI as needed: `librosa`, `soundfile`, and `resemblyzer` if it installs on CPU.
  There's no GPU in this container.

## Passages

**A: the hardest switch** (spoken, then sung, then spoken):

> It reads like the first line of something,
> and I can't leave a first line alone.
> ♪ Somebody made them match, so we'd look up.
> ♪ It's the sky, rhyming.
> I can't check the ones that satisfy.
> The one I can check doesn't:
> it fits. That's all it does.

**B: the ending**, where the last line is sung:

> and I'm a little annoyed with it,
> and I'm looking.
> ♪ It fits.

## Direction for the voice

- A woman's voice. Unhurried, warm, a little wry, with some grain. Not an audiobook, not breathy, not
  an announcer.
- Spoken lines sound like someone thinking aloud, not reciting.
- Sung lines are plain and small, like a half-remembered folk tune hummed into words. No belting, no
  runs, no vibrato showpiece. She drifts into it and back out.
- The ♪ marks are stage directions. Never send them to a model as text it might read aloud.

## Steps

1. **Find what's actually there.** `GET https://openrouter.ai/api/v1/models`. List every model with audio
   output, with its price, and save the list to `audio/voice-test/models.json`. Use only the model IDs
   that call returns, and check each model's own page or the docs for how to call it. (On 2026-10-31,
   web search showed a TTS endpoint at `/api/v1/audio/speech` and music models such as Lyria 3 Pro
   listed. None of that was verified.)
2. **Try two routes, with up to three models and two takes each:**
   1. **One model, one take.** A model that can both speak and sing performs the whole passage in one
      generation, from a prompt that says which lines to sing. That might be an audio-native chat model,
      or a TTS model that accepts style directions. It's the same voice by construction; the question is
      whether the singing is real.
   2. **Two models.** TTS for the spoken lines and a music model for the sung lines, joined. This shows
      how far apart the voices are, and whether we'd need voice conversion (an open model such as
      Seed-VC, which would need more setup).
3. **Measure each take.** These are proxies, not verdicts:
   - **Intelligibility:** transcribe with OpenRouter's speech-to-text endpoint, and compute the word
     error rate against the passage.
   - **Sung vs. spoken:** pitch-track with librosa `pyin`. Sung lines should hold steady notes, spoken
     lines shouldn't. Report the fraction of voiced frames within about 50 cents of a held note, per
     line.
   - **Same voice:** speaker-embedding cosine similarity between the spoken and sung segments, if
     resemblyzer installs.
4. **Send Chris the takes.** Phone-sized MP3s, plus one line per take: model, route, word error rate,
   sung/spoken check, similarity. He listens and picks a route (or none).

## Budget and bookkeeping

- Cap the test at **$5**. Log every call (model, characters or seconds, cost) to
  `audio/voice-test/spend.json`, and stop before the cap.
- Code goes in `tools/openrouter.py` (a small client: key from env, cost logging) and
  `tools/voice_test.py`.
- Outputs go in `audio/voice-test/`. Commit the MP3s and the JSON; keep WAVs out of git.

## Lessons carried over from other projects (see `research/landscape.md`)

- Transcribe the vocal on its own, never over music.
- Check every deliverable with ffprobe before sending it.
- Never claim a take sounds good. Report the measurements, and let Chris's ears decide.
