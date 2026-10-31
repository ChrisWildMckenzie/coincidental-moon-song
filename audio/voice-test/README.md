# Voice test results

*Generated 2026-10-31 23:02 UTC by `tools/voice_test.py run`. See `notes/voice-test.md` for the plan.*

These numbers are proxies, not verdicts. Chris's ears decide.

- **WER**: word error rate of a Whisper large-v3 transcript against the passage.
- **Held**: share of voiced frames inside a held note (at least 150 ms within ±50 cents). Sung lines should be high, spoken lines low. Pitch is plain YIN, not librosa's pYIN.
- **Similarity**: cosine similarity of WeSpeaker speaker embeddings, spoken lines vs sung lines of the same take. Singing lowers this even for one real person, so compare takes with each other and with the baselines below, not with 1.0. Under ~2 s of singing it is noisy.

| Take | How | WER | Held, spoken | Held, sung | Similarity | Length |
|---|---|---|---|---|---|---|
| [A-r1-gemini-t1](A-r1-gemini-t1.mp3) | google/gemini-3.8-flash-tts, one take | 2% | 28% | 62% | 0.774 | 26 s |
| [A-r1-gemini-t2](A-r1-gemini-t2.mp3) | google/gemini-3.8-flash-tts, one take | 0% | 18% | 24% | 0.809 | 25 s |
| [B-r1-gemini-t1](B-r1-gemini-t1.mp3) | google/gemini-3.8-flash-tts, one take | 0% | 9% | 0% | 0.225 (short) | 7 s |
| [B-r1-gemini-t2](B-r1-gemini-t2.mp3) | google/gemini-3.8-flash-tts, one take | 0% | 25% | 98% | 0.197 (short) | 8 s |
| [A-r1-seed-t1](A-r1-seed-t1.mp3) | bytedance-seed/seed-audio-1-0, one take | 0% | 40% | 87% | 0.552 | 29 s |
| [A-r1-seed-t2](A-r1-seed-t2.mp3) | bytedance-seed/seed-audio-1-0, one take | 0% | 51% | 83% | 0.747 | 31 s |
| [B-r1-seed-t1](B-r1-seed-t1.mp3) | bytedance-seed/seed-audio-1-0, one take | 0% | 64% | 0% | 0.194 (short) | 10 s |
| [B-r1-seed-t2](B-r1-seed-t2.mp3) | bytedance-seed/seed-audio-1-0, one take | 0% | 64% | 62% | 0.13 (short) | 10 s |
| [A-r1-eleven-t1](A-r1-eleven-t1.mp3) | elevenlabs/eleven-v4, one take | 2% | 0% | 53% | 0.733 | 18 s |
| [A-r1-eleven-t2](A-r1-eleven-t2.mp3) | elevenlabs/eleven-v4, one take | 2% | 4% | 53% | 0.696 | 18 s |
| [B-r1-eleven-t1](B-r1-eleven-t1.mp3) | elevenlabs/eleven-v4, one take | 0% | 32% | 0% | 0.301 (short) | 5 s |
| [B-r1-eleven-t2](B-r1-eleven-t2.mp3) | elevenlabs/eleven-v4, one take | 0% | 50% | 0% | 0.173 (short) | 5 s |
| [A-r2-lyria-t1](A-r2-lyria-t1.mp3) | google/gemini-3.8-flash-tts speaks, google/lyria-3-clip-preview sings | 0% | 25% | 94% | 0.362 | 35 s |
| [A-r2-lyria-t2](A-r2-lyria-t2.mp3) | google/gemini-3.8-flash-tts speaks, google/lyria-3-clip-preview sings | 2% | 38% | 94% | 0.347 | 34 s |
| [B-r2-lyria-t1](B-r2-lyria-t1.mp3) | google/gemini-3.8-flash-tts speaks, google/lyria-3-clip-preview sings | 0% | 51% | 62% | 0.14 (short) | 8 s |
| [B-r2-lyria-t2](B-r2-lyria-t2.mp3) | google/gemini-3.8-flash-tts speaks, google/lyria-3-clip-preview sings | 17% | 32% | n/a | n/a | 7 s |
| [A-r2-seed-t1](A-r2-seed-t1.mp3) | google/gemini-3.8-flash-tts speaks, bytedance-seed/seed-audio-1-0 sings (voice cloned from the spoken part) | 0% | 27% | 63% | 0.69 | 26 s |
| [A-r2-seed-t2](A-r2-seed-t2.mp3) | google/gemini-3.8-flash-tts speaks, bytedance-seed/seed-audio-1-0 sings (voice cloned from the spoken part) | 2% | 38% | 54% | 0.708 | 23 s |
| [B-r2-seed-t1](B-r2-seed-t1.mp3) | google/gemini-3.8-flash-tts speaks, bytedance-seed/seed-audio-1-0 sings (voice cloned from the spoken part) | 0% | 51% | 64% | 0.215 (short) | 7 s |
| [B-r2-seed-t2](B-r2-seed-t2.mp3) | google/gemini-3.8-flash-tts speaks, bytedance-seed/seed-audio-1-0 sings (voice cloned from the spoken part) | 0% | 32% | 0% | 0.132 (short) | 7 s |
| [A-r2-fish-t1](A-r2-fish-t1.mp3) | google/gemini-3.8-flash-tts speaks, fish-audio/drama-3 sings (voice cloned from the spoken part) | 0% | 27% | 43% | 0.595 | 23 s |
| [A-r2-fish-t2](A-r2-fish-t2.mp3) | google/gemini-3.8-flash-tts speaks, fish-audio/drama-3 sings (voice cloned from the spoken part) | 0% | 38% | 55% | 0.7 | 23 s |
| [B-r2-fish-t1](B-r2-fish-t1.mp3) | google/gemini-3.8-flash-tts speaks, fish-audio/drama-3 sings (voice cloned from the spoken part) | 0% | 51% | 0% | 0.304 (short) | 7 s |
| [B-r2-fish-t2](B-r2-fish-t2.mp3) | google/gemini-3.8-flash-tts speaks, fish-audio/drama-3 sings (voice cloned from the spoken part) | 0% | 32% | 0% | 0.311 (short) | 6 s |

**Similarity baselines** (spoken audio only):

- same voice, two generations (Gemini A t1 vs t2, spoken): 0.842
- different voices (Gemini vs Eleven, A t1, spoken): 0.211

Spend: $0.51 of the $5.00 cap (`spend.json`).
