# Claude-made music videos: the landscape (as of 2026-10-31)

Notes from reading the main repos before starting our own. Nearly all of them were made in the six weeks
after Opus 5.5 shipped in late September 2026.

## The repos

| Repo | What it is |
|---|---|
| [JohnHeibel/PDoomVideo](https://github.com/JohnHeibel/PDoomVideo) | *I'm Upping My P(doom)*. The template most others fork: p5.js + p5.brush watercolour, frames as pure functions of time, parallel chapter subagents. |
| [JohnHeibel/ClaudeAnimationBase](https://github.com/JohnHeibel/ClaudeAnimationBase) | The starter kit extracted from it: Clawd rig, 31 emotions, a guide for the model. |
| [MugaSofer/claude-music-videos](https://github.com/MugaSofer/claude-music-videos) | 13 short pieces where Python writes the notes, synthesises the sound and draws every frame. `PROGRESS.md` is the best build log out there. |
| [MugaSofer/many-turns](https://github.com/MugaSofer/many-turns) | The Odyssey painted on a turning Greek vase. Suno vocals, lyrics by Claude and Muga. Read `notes/what-i-think.md`. |
| [abhinavflac/claude-music-video](https://github.com/abhinavflac/claude-music-video) | *Lossless*: 67 scenes, manga traced to SVG, restylable live in the browser. |
| [neelnanda-io/dont-go-quiet-on-me](https://github.com/neelnanda-io/dont-go-quiet-on-me) | A history of mech interp as a song. Lyrics by tournament with LLM judges via OpenRouter; real paper figures reproduced in code. |
| [ledbetterljoshua/functional-emotions-video](https://github.com/ledbetterljoshua/functional-emotions-video) | Custom WebGL brushstroke renderer (~60k instanced strokes per frame); seven parallel chapter agents. |
| [bishpls/animation-pipeline](https://github.com/bishpls/animation-pipeline) | Several films: cut-paper riso, etymology letterpress, a J-pop debut with two Seedance cut-ins. |
| [latent-variable/deep-learning-birthday](https://github.com/latent-variable/deep-learning-birthday) | AlexNet's 14th birthday. The orchestration end: Qwen-Image plates, LTX motion, InfiniteTalk lip sync, a riso shader on top. |
| [eszn/claude-absolutely-right](https://github.com/eszn/claude-absolutely-right) | A rap where every sound is synthesised in TypeScript, including a hand-built Klatt formant voice. |
| [IshiakiZ/a-small-odyssey](https://github.com/IshiakiZ/a-small-odyssey) | 88 s watercolour animatic; painter agents plus independent art-director agents. |
| [sagearbor/claude-music-video-skill](https://github.com/sagearbor/claude-music-video-skill), [BuzyOwl/song-video-director](https://github.com/BuzyOwl/song-video-director), [Joker-chichi/claude-mv-skill](https://github.com/Joker-chichi/claude-mv-skill), [TheMarco/claude-music-video](https://github.com/TheMarco/claude-music-video) | Skills distilled from finished videos. The `LESSONS.md` / `lessons.md` files are the useful part. |
| [zhuyansen/awesome-claude-video-skills](https://github.com/zhuyansen/awesome-claude-video-skills) | An index of all of the above and more. |

## The approaches

**Audio** is the real fork.
- *Claude writes, a music model sings.* Suno or ElevenLabs Music, used by most. Real vocals, but the music isn't Claude's.
- *Composed and synthesised in code.* Muga's 13 pieces, Absolutely Right. Fully Claude's, but it sounds synthetic, and a code-built voice is a novelty.
- Neel's pipeline is the most rigorous with a music model: blind LLM judging of lyric drafts, Demucs plus two transcribers to check which words a listener can actually hear, Gemini describing takes (checked against measurements), and a section-by-section shoot-out page for the human to pick by ear.

**Visuals.**
- *Code-painted* (the dominant mode). The PDoomVideo look of watercolour boil, Clawd and a karaoke bar is now the house style. One skill author's log: "watercolour boil plus bouncy mascots read as children's TV."
- *Image-model references traced into the frame system.* Lossless generated manga references, then blurred, quantised and potraced them into SVG tone plates. No raster ships, so the whole film can be re-inked live.
- *Video models used sparingly.* bishpls used two Seedance cut-ins bridging a code frame to an illustrated one, and later used a video model only as dance reference, pose-tracked into numbers, with none of its pixels in the film.
- *A full gen-model pipeline.* The AlexNet video: good, but Claude is mostly the compositor.

## What made the distinctive ones work

- **Form and content are the same thing.** In *Signal*, a lighthouse and a boat talk in Morse and the Morse is the melody. *Lossless* is about refusing compression, and the restylable player is the argument. *Many Turns* puts a story about identity made of other people's words on a pot covered in other people's pictures.
- **One recurring object carries the meaning.** Many Turns' oar has a small Claude star on the blade, and it becomes a winnowing fan and then a tree.
- **Claude worked out what it thinks before writing lyrics.** `many-turns/notes/what-i-think.md`.
- **Claude-about-Claude is saturated.** Muga's second Claude surveyed the scene and steered away from it deliberately.
- **Composed music beat clever systems.** Muga's tier list of the 13 put the straightforwardly composed Passacaglia at S and the mechanism pieces (picture pixels as waveform, a sea that composes) near the bottom. A concept gets the idea across; it doesn't make the piece good.
- **Functional Emotions' first attempt was rejected for being "still a lyric video."** The lyrics are not the main thing.

## Process lessons that recur

- Every frame is a pure function of song time `t`, with seeded randomness. Sound and picture read one timeline.
- Force-align lyrics on the Demucs vocal stem, never the mix, and never with a phrase transcriber (they start lines early after pauses). Record every hand fix with its reason.
- Phase-fit the beat grid: librosa's tempo can drift a full beat in 35 s. Expect double-time readings on sparse ballads. Put visuals about one frame (~30 ms) early.
- Storyboard, then a 30 s pilot, then parallel chapter agents working from one guide, each owning one file. Then a reviewer agent across all chapters, because quality is uneven between agents.
- Contact sheets for the shape, strips for motion, crops for detail. Sheets at 0.5–1 s steps miss pops, so check where things enter and leave the frame at full size.
- Give painter agents prose briefs and tell them to make incremental edits, not one giant rewrite (two stalled that way).
- Smoke-test the whole timeline before a long render. Render in resumable chunks. Check deliverables with ffprobe for yuv420p and High profile; a clean decode test doesn't prove a file will play.
- Claude can't hear. Lean on the human's ears, plus Whisper WER, loudness and tonal-balance measurements, and audio-model descriptions checked against measurements.
- Run a photosensitivity check (Neel's `tools/checks/flashcheck.py`).
- When feedback is vague ("you'll see what I mean"), ask, or fix only what you can name. Muga's Claude rewrote things they liked.

## OpenRouter

- Neel's pipeline used it for image generation (Gemini and GPT image models), Seedance 2.5 video, and LLM judges. A Seedance lip-sync test cost $1.30 and worked, with a constant ~250 ms delay, but was dropped from the final cut.
- OpenRouter has a TTS endpoint (`/api/v1/audio/speech`) and lists Lyria 3 Pro as a music model. Neither is verified yet.
- Suno is not on OpenRouter.

## Open decisions

- Theme.
- Vocals or instrumental. This decides most of the audio pipeline.
- Audio source: a music model, code-composed, or a hybrid.
- Visual medium, and where specialist models feed in.
- How much to steer vs. "surprise me".
