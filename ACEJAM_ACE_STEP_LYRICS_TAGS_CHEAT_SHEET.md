# AceJAM ACE-Step Lyrics Tags Cheat Sheet

Use this file as the repo source of truth for what we tell models they can do
inside the ACE-Step `lyrics` field.

This cheat sheet is intentionally split into trust levels:

- `Officially documented`
- `Observed / likely supported in practice`
- `Do not rely on this`

The goal is to be complete without overstating what ACE-Step guarantees.

Official source anchors checked for this sheet:

- `https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md`
- `https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/ace_step_musicians_guide.md`
- `https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md`
- `https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md`
- `https://github.com/ace-step/ACE-Step-1.5/issues/1095`

Repo-local implementation anchors used for the observed list:

- `app/songwriting_toolkit.py`
- `app/prompt_kit.py`

---

## Hard rules before the tag list

1. `caption` is sound-only. Put style, drums, bass, timbre, mood, era,
   production, and vocal character there.
2. `lyrics` is the temporal script. Use it for sections, performance shifts,
   echo lines, and the actual sung or rapped words.
3. BPM, key, time signature, duration, model names, seed, or metadata do not
   belong in the lyric lines.
4. Prefer square-bracket section tags at the start of a block, then lyric lines.
5. Use one modifier max in `[Section - modifier]`.
6. Use parentheses inside lyric lines for echoes, ad-libs, and backing-vocal
   responses.
7. No HTML, no markdown styling, no colored words, no nested formatting tricks.

---

## Officially documented

These are the safest patterns because ACE-Step explicitly documents them or
shows them as canonical examples.

### Structure tags

- `[Intro]`
  - Opening section.
  - Best for scene setup, a chant entrance, or sparse opening bars.
- `[Verse]`
  - General verse section when numbering is unnecessary.
- `[Verse 1]`
  - First verse.
- `[Verse 2]`
  - Second verse.
- `[Pre-Chorus]`
  - Lift section that increases tension before the chorus.
- `[Chorus]`
  - Main hook or chorus section.
- `[Bridge]`
  - Contrast section with a new emotional or melodic angle.
- `[Outro]`
  - Ending section.

### Dynamic / arrangement tags

- `[Build]`
  - Energy build-up before a release.
- `[Drop]`
  - Release point, especially for electronic or hybrid production.
- `[Breakdown]`
  - Stripped section with less density and more space.
- `[Fade Out]`
  - Signals a fading ending.
- `[Silence]`
  - Hard stop or dramatic pause.

### Instrumental / special section tags

- `[Instrumental]`
  - No singing or rapping; instrumental passage only.
- `[Guitar Solo]`
  - Guitar solo section.
- `[Piano Interlude]`
  - Piano-led interlude.

### Vocal / delivery tags documented by ACE-Step examples

These are documented as valid performance ideas, but ACE-Step works best when
they are used as modifiers inside a section tag rather than as standalone
bracket lines.

- `whispered`
  - Soft confidential delivery.
- `spoken`
  - Spoken or recited delivery.
- `falsetto`
  - Light high-register lead.
- `harmonies`
  - Layered harmonized support.
- `call and response`
  - Lead line with backing answer.
- `ad-lib`
  - Accent phrase, reaction, or supporting exclamation.
- `raspy vocal`
  - Rougher vocal timbre.
- `powerful belting`
  - Strong full-voice emphasis.
- `high energy`
  - More impact and intensity.
- `low energy`
  - Pulled-back, restrained delivery.
- `building energy`
  - Intensifying over the section.
- `explosive`
  - Abrupt high-impact push.
- `melancholic`
  - Sad or heavy emotional shading.
- `euphoric`
  - Release and uplift.
- `dreamy`
  - Soft blurred emotional or textural mood.
- `aggressive`
  - Hard attack, punch, or hostile pressure.

### Official syntax patterns

- `[Section - modifier]`
  - Example: `[Verse - whispered]`
  - What it does: keeps the section identity while steering delivery.
  - Safe usage: one modifier, one dash, no long stacks.

- `main lyric (echo lyric)`
  - Example: `We run the city (run the city)`
  - What it does: the parenthesized text is treated like backing vocals,
    echo, ad-lib, or harmony support.

- `ALL CAPS`
  - Example: `WE WON'T FALL`
  - What it does: adds shouted emphasis or hard accent.
  - Safe usage: accents, refrains, one-word hits; not whole verses.

- vowel stretching
  - Example: `so aliiive`
  - What it does: suggests sustained vowels.
  - Important: ACE-Step docs imply this can work, but not always reliably.

---

## Observed / likely supported in practice

These are not the same as “officially guaranteed,” but they are part of the
AceJAM toolkit and are treated as likely-supported semantic tags.

### Additional structure tags used by AceJAM

- `[Verse 3]`
  - Third verse.
- `[Post-Chorus]`
  - Follow-up section after a chorus.
- `[Hook]`
  - Hook section label.
- `[Hook/Chorus]`
  - Combined hook-and-chorus label.
- `[Refrain]`
  - Repeating refrain section.
- `[Final Chorus]`
  - Last chorus pass, often bigger or more emotional.
- `[Interlude]`
  - Short transitional section.
- `[Build-Up]`
  - Alternate build label.
- `[Final Drop]`
  - Endgame drop.
- `[Climax]`
  - Peak-emotion or peak-energy section.
- `[Beat Switch]`
  - Indicates a production shift in the arrangement.

### Additional instrumental / sectional labels used in AceJAM

- `[inst]`
  - Shorthand instrumental label.
- `[Instrumental Break]`
  - Instrumental gap or reset.
- `[Synth Solo]`
  - Synth lead feature.
- `[Piano Solo]`
  - Piano solo.
- `[Brass Break]`
  - Brass accent section.
- `[Saxophone Solo]`
  - Saxophone feature.
- `[Violin Solo]`
  - Violin feature.
- `[Drum Break]`
  - Drum-only or drum-led section.

### Additional modifiers used in AceJAM

- `[Verse - rap]`
  - Reliable rap-mode cue.
- `[Verse - melodic rap]`
  - Rap with more melody in the pocket.
- `[Verse - double time rap]`
  - Faster syllabic density.
- `[Verse - spoken]`
  - Spoken verse delivery.
- `[Verse - shouted]`
  - Harder attack.
- `[Verse - powerful]`
  - Strong forceful lead.
- `[Verse - crooned]`
  - Smooth melodic phrasing.
- `[Chorus - anthemic]`
  - Big chorus lift.
- `[Chorus - rap]`
  - Chorus performed as rap.
- `[Chorus - layered vocals]`
  - Stacked chorus.
- `[Chorus - chant]`
  - Group-chant energy.
- `[Chorus - whispered]`
  - Intimate or eerie chorus.
- `[Chorus - call and response]`
  - Lead/backing volley.
- `[Bridge - emotional]`
  - Emotional contrast.
- `[Intro - dreamy]`
  - Dreamy entry.
- `[Intro - dark]`
  - Dark setup.
- `[Intro - spoken]`
  - Spoken opening.
- `[Intro - ambient]`
  - Texture-first opening.
- `[Intro - piano]`
  - Piano-led opening.
- `[Intro - talkbox]`
  - Talkbox intro.
- `[Outro - spoken]`
  - Spoken ending.
- `[Outro - acapella]`
  - Voice-only ending.
- `[Outro - talkbox]`
  - Talkbox ending.
- `[Climax - powerful]`
  - Peak-force section.
- `[Hook - sung]`
  - Sung hook.
- `[Hook - chant]`
  - Chanted hook.

### What these observed tags are good for

- Better planning clarity for AceJAM prompts.
- Stronger alignment with our rap, boom bap, G-funk, trap, and drill workflows.
- More explicit control over hook style and section pressure.

### What to remember

- Use these as “high-confidence practical tags,” not as guaranteed ACE-Step
  parser features.
- Keep them simple and musically obvious.
- Avoid inventing long custom modifiers if one of these already expresses the
  intent.

---

## Do not rely on this

These patterns are either unsupported, unclear, or too unstable to promise in a
prompt contract.

- HTML in lyrics
  - Example: `<span style="color:red">word</span>`
  - Do not use.
- Markdown styling in lyrics
  - Example: `**line**`, `_line_`
  - Do not use.
- Colored-word annotation
  - Example: Genius-style word-color breakdowns.
  - Do not use in `lyrics`.
- Speaker scripts as a hard syntax
  - Example: `[Male:]`, `[Female:]`, `[Rapper 1:]`
  - ACE-Step has no reliable public standard for this.
- Multiple stacked modifiers
  - Example: `[Verse - whispered - aggressive - doubled]`
  - Too unstable; tag text may be sung.
- Metadata embedded in lyric lines
  - Example: `[120 BPM]`, `[F minor]`
  - Keep metadata in fields, not lyrics.
- Nested parentheses tricks
  - Example: `line ((echo))`
  - Unclear behavior.
- Prompt commentary inside lyrics
  - Example: `[make this more emotional]`
  - Never do this.
- Long prose directions inside square brackets
  - Example: `[Verse - cinematic but also gritty with modern trap details and huge strings]`
  - Too verbose; higher risk of literal singing.

---

## Best-practice formatting rules

- One section tag, then the lines for that section.
- Leave a blank line between sections.
- Keep lyric lines performable.
- Use ad-libs in parentheses on the same line or the next short line.
- Prefer:
  - `[Verse - rap]`
  - lyric lines
- Avoid:
  - standalone bracket lines for every tiny vocal effect

Safe example:

```text
[Verse - rap]
They paved them blocks just to hide what's real (for real)
Boardroom smiles while they cut them deals

[Chorus - chant]
We came back loud (came back loud)
We don't bow down
```

Unsafe example:

```text
[aggressive]
[high energy]
[male vocal]
[120 BPM]
[Verse - whispered - powerful - dark]
```

---

## Per-genre quick examples

### Rap

```text
[Intro]
Yeah

[Verse - rap]
I learned the code from the cracks in the wall (uh)
Now every number on the file feels small

[Hook - chant]
We don't fold
We don't break
```

Why it works:
- clear rap cue
- ad-libs in parentheses
- hook has chant energy without extra formatting noise

### Drill

```text
[Intro - dark]
Look

[Verse - rap]
Cold on the glide with the weight in my chest (grrt)
Back of the block where the sirens don't rest

[Chorus - chant]
Still outside
Still on ten
```

Why it works:
- dark intro modifier
- tight short drill bars
- chant chorus

### Pop

```text
[Verse]
You left your coat by the doorway light
I kept the silence on all night

[Pre-Chorus]
Say it slow
Don't let go

[Chorus - anthemic]
If you stay, I can breathe again
```

Why it works:
- classic verse / pre / chorus structure
- anthemic modifier used only where it matters

### R&B

```text
[Verse - whispered]
Hands on the glass, your name in the steam
Slow little sparks in the edge of the dream

[Chorus - layered vocals]
Stay for the night (stay)
Stay in the light
```

Why it works:
- whisper modifier for intimacy
- layered-vocal chorus
- echoes in parentheses

### Singer-songwriter

```text
[Verse]
The kettle shook at a quarter to two
I heard the hallway before I heard you

[Bridge]
Maybe the truth is small
Maybe it lives in the pause
```

Why it works:
- no over-tagging
- scene writing carries the section

### Rock and roll

```text
[Intro]
One, two, three

[Verse - powerful]
Red lights jumping off the chrome tonight
Boot heels kicking up the dust just right

[Chorus]
Come on back
Come on back to me
```

Why it works:
- simple strong structure
- forceful verse modifier

### Afro-Caribbean

```text
[Intro]
Yeah yeah

[Verse]
Waistline roll when the moon sits low
Whole block warm when the riddim gets close

[Chorus - call and response]
Move with me (move with me)
Stay with me (stay with me)
```

Why it works:
- movement language
- repetition discipline
- call/response done with both modifier and parentheses

---

## Prompt-writing policy for AceJAM

When AceJAM prompts mention lyrics tags, they should:

- present official tags as the safe default
- allow observed tags only when clearly framed as likely-supported
- forbid HTML, color markup, and prose instructions inside `lyrics`
- keep vocal-character and energy concepts mostly in `caption` and comma tags
- use square brackets only for sections and short section modifiers

If there is any doubt, prefer simpler tags and stronger lyric writing over more
formatting.
