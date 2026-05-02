AceJAM Studio
ACE-Step 1.5
Create
Simple
Custom
Edit
Cover
Repaint
Extract
Lego
Complete
Produce
Album
Trainer
Manage
Library
Settings
✕ AlbumPayloadQualityGate failed for track 1: duplicate_section_tags:[Verse 2],[Break],[Bridge],[Final Chorus],[Outro]
Adapter

No adapter
Adapter scale

 
Use adapter
PEFT LoRA adapters trained in Trainer can be applied to Simple, Custom, Cover, Repaint, Lego and Complete.
Production Team Brief
Describe the album. Your local LLM plans tracks with agents, then AceJAM renders every model album.
Prompt mode

Current mode
Provider

Ollama
AI Fill model

charaf/Huihui-Qwen3.6-35B-A3B-abliterated-mlx:thinking
 
Fill All Fields
Describe what you want
A 12-track award-level album about pressure, loyalty, money and rebirth...
AI Fill uses the selected local LLM and directly updates the current mode. You can edit every field before generating.
Production Team
Album
Plan once with local agents, edit every track, then render the album across the selected ACE-Step model strategy.

7-model studio
Concept
Album: You Buried the Wrong Man
Track 1: Concrete Canyons (Prod. Dr. Dre)

Vibe: Low-end rumble, sirens, West Coast weight

Verse:

They paved them blocks just to hide what’s real,
Boardroom smiles while they cut them deals.
Built ten towers off a fallen name,
Turned a man’s life to a numbers game.
Said it was peace, but I peeped that lie,
Same hands shaking be the ones that try—
To bury a king under concrete floors,
But ghosts don’t sleep when they settle scores.

Naming Drop Style:
“Death Row… East Coast… Closed doors…”
Genre prompt for agents
West Coast rap, clear hook, polished modern mix
Mood/vibe
dark, tense, triumphant, cinematic
Vocal type
male rap lead, clear chorus response
Audience/platform
streaming release, high quality album tracks
Tracks
1
Language

en
Album planner

Ollama
Planner model

charaf/Huihui-Qwen3.6-35B-A3B-abliterated-mlx:thinking
Embedding provider

Ollama
Embedding model

nomic-embed-text:latest
Model

XL SFT (highest quality)
Quality target

Hit single standard
 Planner thinking
 Print full agent I/O
Generate Album
Refresh local LLMs
Default: 7 full albums, one per ACE-Step model. Missing models auto-download, then generation resumes.
1
tracks
7
model albums
7
renders
3
downloads needed
7 model albums · 7 planned render(s)
Turbo
installed
Turbo Shift3
download required
SFT
installed
Base
download required
XL Turbo
download required
XL SFT
installed
XL Base
installed
Payload Review
Album payload will appear after planning.
Agent Runtime
Ollama planner charaf/Huihui-Qwen3.6-35B-A3B-abliterated-mlx:thinking, Ollama embedding nomic-embed-text:latest
Agent Tools
4 tag packs
Lyrics Craft
dense lyrics, varied BPM, related keys
Generation
1 variant(s), wav32, acestep-v15-xl-sft
Saved Album Tracks
Refresh
Album family ef15825001c1: 1 saved track(s) across 1 model album(s).
Download all model albums ZIP
Download selected album ZIP
Delete album family
Delete selected album

Concrete Signal - Concrete Canyons
text2music · xl-sft · album Album: You Buried the Wrong Ma · #1 · 240s · 2026-05-01

4:00


Download
✕ Job dacf0eeb56cd: FAILED · Album failed or incomplete · 5% · 2m 27s
Finished: 1-5-2026, 14:22:29
Engine: acejam_agents · AceJAM Agents used
Prompt Kit: ace-step-multilingual-hit-kit-2026-04-26
Input Contract: applied · Repairs: 0
Planner: ollama / charaf/Huihui-Qwen3.6-35B-A3B-abliterated-mlx:thinking · Embedding: ollama / nomic-embed-text:latest · Memory: off
Agent debug dir: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd
Generated: 0/1
ERROR: AlbumPayloadQualityGate failed for track 1: duplicate_section_tags:[Verse 2],[Break],[Bridge],[Final Chorus],[Outro]
---
Agent engine: AceJAM Agents (acejam-album-director-prompt-first-2026-05-01).
Agent debug log dir: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd
Agent raw prompts JSONL: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd/03_agent_prompts.jsonl
Agent raw responses JSONL: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd/04_agent_responses.jsonl
Agent track state JSONL: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd/05_track_state.jsonl
Agent gate reports JSONL: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd/06_gate_reports.jsonl
Agent final payloads JSONL: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd/07_final_payloads.jsonl
Ollama preflight: planner chat=True.
AceJAM Agent call: Album Intake Agent attempt 1 via Ollama (prompt_chars=6267, system=3656, user=2611).
AceJAM Agent response: Album Intake Agent 628 chars in 8.635s (parse pending).
AceJAM Agent parsed JSON: Album Intake Agent attempt 1 ok.
AceJAM Agent call: Track Concept Agent attempt 1 via Ollama (prompt_chars=5720, system=3721, user=1999).
AceJAM Agent response: Track Concept Agent 620 chars in 6.398s (parse pending).
AceJAM Agent parsed JSON: Track Concept Agent attempt 1 ok.
AceJAM Agent call: Tag Agent attempt 1 via Ollama (prompt_chars=6761, system=3978, user=2783).
AceJAM Agent response: Tag Agent 794 chars in 7.336s (parse pending).
AceJAM Agent parsed JSON: Tag Agent attempt 1 ok.
AceJAM Agent call: BPM Agent attempt 1 via Ollama (prompt_chars=6906, system=3510, user=3396).
AceJAM Agent response: BPM Agent 10 chars in 2.715s (parse pending).
AceJAM Agent parsed JSON: BPM Agent attempt 1 ok.
AceJAM Agent call: Key Agent attempt 1 via Ollama (prompt_chars=6931, system=3529, user=3402).
AceJAM Agent response: Key Agent 23 chars in 2.739s (parse pending).
AceJAM Agent parsed JSON: Key Agent attempt 1 ok.
AceJAM Agent call: Time Signature Agent attempt 1 via Ollama (prompt_chars=6979, system=3555, user=3424).
AceJAM Agent response: Time Signature Agent 22 chars in 2.734s (parse pending).
AceJAM Agent parsed JSON: Time Signature Agent attempt 1 ok.
AceJAM Agent call: Duration Agent attempt 1 via Ollama (prompt_chars=6995, system=3531, user=3464).
AceJAM Agent response: Duration Agent 16 chars in 2.763s (parse pending).
AceJAM Agent parsed JSON: Duration Agent attempt 1 ok.
AceJAM Agent call: Section Map Agent attempt 1 via Ollama (prompt_chars=7433, system=3654, user=3779).
AceJAM Agent response: Section Map Agent 443 chars in 5.916s (parse pending).
AceJAM Agent parsed JSON: Section Map Agent attempt 1 ok.
AceJAM Agent call: Hook Agent attempt 1 via Ollama (prompt_chars=8606, system=3580, user=5026).
AceJAM Agent response: Hook Agent 755 chars in 8.479s (parse pending).
AceJAM Agent parsed JSON: Hook Agent attempt 1 ok.
AceJAM Agent call: Track Lyrics Agent Part 1 attempt 1 via Ollama (prompt_chars=10897, system=3863, user=7034).
AceJAM Agent response: Track Lyrics Agent Part 1 1877 chars in 19.178s (parse pending).
AceJAM Agent parsed JSON: Track Lyrics Agent Part 1 attempt 1 ok.
AceJAM Agent call: Track Lyrics Agent Part 2 attempt 1 via Ollama (prompt_chars=11793, system=3863, user=7930).
AceJAM Agent response: Track Lyrics Agent Part 2 942 chars in 11.914s (parse pending).
AceJAM Agent parsed JSON: Track Lyrics Agent Part 2 attempt 1 ok.
AceJAM Agent call: Track Lyrics Agent Part 3 attempt 1 via Ollama (prompt_chars=11798, system=3863, user=7935).
AceJAM Agent response: Track Lyrics Agent Part 3 784 chars in 10.776s (parse pending).
AceJAM Agent parsed JSON: Track Lyrics Agent Part 3 attempt 1 ok.
AceJAM Agent call: Caption Agent attempt 1 via Ollama (prompt_chars=7467, system=3848, user=3619).
AceJAM Agent response: Caption Agent 313 chars in 4.55s (parse pending).
AceJAM Agent parsed JSON: Caption Agent attempt 1 ok.
AceJAM Agent call: Performance Agent attempt 1 via Ollama (prompt_chars=8760, system=3628, user=5132).
AceJAM Agent response: Performance Agent 1034 chars in 9.222s (parse pending).
AceJAM Agent parsed JSON: Performance Agent attempt 1 ok.
AceJAM Agent call: Final Payload Assembler attempt 1 via Ollama (prompt_chars=16988, system=3892, user=13096).
AceJAM Agent response: Final Payload Assembler 5038 chars in 40.396s (parse pending).
AceJAM Agent parsed JSON: Final Payload Assembler attempt 1 ok.
AceJAM Director planning failed loudly: AlbumPayloadQualityGate failed for track 1: duplicate_section_tags:[Verse 2],[Break],[Bridge],[Final Chorus],[Outro]
Album debug log dir: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd
Phase 1: Planning album with AceJAM Agents and deterministic gates...
Concept preview: Album: You Buried the Wrong Man Track 1: Concrete Canyons (Prod. Dr. Dre) Vibe: Low-end rumble, sirens, West Coast weight Verse: They paved them blocks just to hide what’s real, Boardroom smiles while they cut them de...
Language: English
Prompt Kit: ace-step-multilingual-hit-kit-2026-04-26
Prompt Kit routing: language_preset=en; genre_modules=pop.
Input Contract: applied; album_title=You Buried the Wrong Man; locked_tracks=1; blocked_unsafe=0.
Tracks: 1 x 240s
Planner LM: Ollama (charaf/Huihui-Qwen3.6-35B-A3B-abliterated-mlx:thinking); ACE-Step LM is disabled for album-agent payloads.
Embedding: Ollama (nomic-embed-text:latest); not used by the pure sequential director.
AceJAM Director runtime: json_retries=2, temperature=0.25, print_agent_io=True, planner_thinking=False.
Agent engine: AceJAM Agents (acejam-album-director-prompt-first-2026-05-01).
Agent debug log dir: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd
Agent raw prompts JSONL: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd/03_agent_prompts.jsonl
Agent raw responses JSONL: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd/04_agent_responses.jsonl
Agent track state JSONL: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd/05_track_state.jsonl
Agent gate reports JSONL: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd/06_gate_reports.jsonl
Agent final payloads JSONL: /Users/charafchnioune/Desktop/code/pinokio/api/acejam.pinokio.git/app/data/debug/album_runs/dacf0eeb56cd/07_final_payloads.jsonl
Ollama preflight: planner chat=True.
AceJAM Agent call: Album Intake Agent attempt 1 via Ollama (prompt_chars=6267, system=3656, user=2611).
AceJAM Agent response: Album Intake Agent 628 chars in 8.635s (parse pending).
AceJAM Agent parsed JSON: Album Intake Agent attempt 1 ok.
AceJAM Agent call: Track Concept Agent attempt 1 via Ollama (prompt_chars=5720, system=3721, user=1999).
AceJAM Agent response: Track Concept Agent 620 chars in 6.398s (parse pending).
AceJAM Agent parsed JSON: Track Concept Agent attempt 1 ok.
AceJAM Agent call: Tag Agent attempt 1 via Ollama (prompt_chars=6761, system=3978, user=2783).
AceJAM Agent response: Tag Agent 794 chars in 7.336s (parse pending).
AceJAM Agent parsed JSON: Tag Agent attempt 1 ok.
AceJAM Agent call: BPM Agent attempt 1 via Ollama (prompt_chars=6906, system=3510, user=3396).
AceJAM Agent response: BPM Agent 10 chars in 2.715s (parse pending).
AceJAM Agent parsed JSON: BPM Agent attempt 1 ok.
AceJAM Agent call: Key Agent attempt 1 via Ollama (prompt_chars=6931, system=3529, user=3402).
AceJAM Agent response: Key Agent 23 chars in 2.739s (parse pending).
AceJAM Agent parsed JSON: Key Agent attempt 1 ok.
AceJAM Agent call: Time Signature Agent attempt 1 via Ollama (prompt_chars=6979, system=3555, user=3424).
AceJAM Agent response: Time Signature Agent 22 chars in 2.734s (parse pending).
AceJAM Agent parsed JSON: Time Signature Agent attempt 1 ok.
AceJAM Agent call: Duration Agent attempt 1 via Ollama (prompt_chars=6995, system=3531, user=3464).
AceJAM Agent response: Duration Agent 16 chars in 2.763s (parse pending).
AceJAM Agent parsed JSON: Duration Agent attempt 1 ok.
AceJAM Agent call: Section Map Agent attempt 1 via Ollama (prompt_chars=7433, system=3654, user=3779).
AceJAM Agent response: Section Map Agent 443 chars in 5.916s (parse pending).
AceJAM Agent parsed JSON: Section Map Agent attempt 1 ok.
AceJAM Agent call: Hook Agent attempt 1 via Ollama (prompt_chars=8606, system=3580, user=5026).
AceJAM Agent response: Hook Agent 755 chars in 8.479s (parse pending).
AceJAM Agent parsed JSON: Hook Agent attempt 1 ok.
AceJAM Agent call: Track Lyrics Agent Part 1 attempt 1 via Ollama (prompt_chars=10897, system=3863, user=7034).
AceJAM Agent response: Track Lyrics Agent Part 1 1877 chars in 19.178s (parse pending).
AceJAM Agent parsed JSON: Track Lyrics Agent Part 1 attempt 1 ok.
AceJAM Agent call: Track Lyrics Agent Part 2 attempt 1 via Ollama (prompt_chars=11793, system=3863, user=7930).
AceJAM Agent response: Track Lyrics Agent Part 2 942 chars in 11.914s (parse pending).
AceJAM Agent parsed JSON: Track Lyrics Agent Part 2 attempt 1 ok.
AceJAM Agent call: Track Lyrics Agent Part 3 attempt 1 via Ollama (prompt_chars=11798, system=3863, user=7935).
AceJAM Agent response: Track Lyrics Agent Part 3 784 chars in 10.776s (parse pending).
AceJAM Agent parsed JSON: Track Lyrics Agent Part 3 attempt 1 ok.
AceJAM Agent call: Caption Agent attempt 1 via Ollama (prompt_chars=7467, system=3848, user=3619).
AceJAM Agent response: Caption Agent 313 chars in 4.55s (parse pending).
AceJAM Agent parsed JSON: Caption Agent attempt 1 ok.
AceJAM Agent call: Performance Agent attempt 1 via Ollama (prompt_chars=8760, system=3628, user=5132).
AceJAM Agent response: Performance Agent 1034 chars in 9.222s (parse pending).
AceJAM Agent parsed JSON: Performance Agent attempt 1 ok.
AceJAM Agent call: Final Payload Assembler attempt 1 via Ollama (prompt_chars=16988, system=3892, user=13096).
AceJAM Agent response: Final Payload Assembler 5038 chars in 40.396s (parse pending).
AceJAM Agent parsed JSON: Final Payload Assembler attempt 1 ok.
AceJAM Director planning failed loudly: AlbumPayloadQualityGate failed for track 1: duplicate_section_tags:[Verse 2],[Break],[Bridge],[Final Chorus],[Outro]
ERROR: Album planning failed
Results
Generated takes appear here with player, download, retry, source/reference, score, LRC and audio-code actions.