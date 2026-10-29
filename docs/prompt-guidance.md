# Pika H3 Flex prompt guidance

Pika H3 Flex prompt-writing guidance. Write scene descriptions in English; keep dialogue and on-screen text in their intended language.

Choose T2VA (text-to-video-and-audio) for text-only generation, FL2VA (first/last-frame-to-video-and-audio) when using a first frame, last frame or both, and Ref2VA (reference-to-video-and-audio) for reference-conditioned generation and editing.

## 1. T2VA: Text to Video

Use three fields: `integrated_multimodal_description` (visuals, action, camera and synchronized sound), `overall_soundscape` (overall sound), and `non_diegetic_music` (background score).

### Prompt template

```txt
integrated_multimodal_description:
[Shot 1] ...

overall_soundscape:
...

non_diegetic_music:
...
```

### Example 1: T2VA — Action and a Visible Result

Input: no assets. Show the moving watch hand and the character's reaction after the lid opens.

```txt
integrated_multimodal_description: [Shot 1] A live-action close-up shows an elderly watchmaker holding a closed pocket watch above his workbench. He presses the crown with his thumb. The lid clicks open, revealing a moving second hand. He watches it tick, smiles and says in a quiet, raspy voice (S1), <d>[English] It works again.</d> The camera remains steady, keeping the open watch and his expression visible.

overall_soundscape: Quiet workshop room tone, a small metal click and steady watch ticking.

non_diegetic_music: N/A
```

## 2. FL2VA: Image-Driven Video

Specify the first frame, last frame or both, according to the inputs. The first frame establishes the opening state; the last frame establishes the ending state. Describe the action connecting them. Each prompt begins with an image-alignment instruction, followed by the same three fields as T2VA.

### First frame only

Use the first image as the video's opening state and develop the action forward from it.

```txt
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description:
[Shot 1] Begin from <Picture 1>. ...

overall_soundscape:
...

non_diegetic_music:
...
```

### Last frame only

Describe the action that reaches the supplied ending image. Replace `{duration}` with the actual video duration using two decimal places.

```txt
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the {duration}-second mark of the target video.

integrated_multimodal_description:
[Shot 1] ... End with the subjects and framing aligned with <Picture 1>.

overall_soundscape:
...

non_diegetic_music:
...
```

### First and last frames

Begin from the first image and describe the continuous action that reaches the last image. Replace `{duration}` with the actual video duration using two decimal places.

```txt
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 0.00-second mark of the target video; <Picture 2> (from [Shot 1]) aligns with the {duration}-second mark of the target video.

integrated_multimodal_description:
[Shot 1] Begin aligned with <Picture 1>. ... End aligned with <Picture 2>.

overall_soundscape:
...

non_diegetic_music:
...
```

### Example 2: FL2VA — Connecting the First and Last Frames

Input: an open tool pouch as the first frame and the same pouch rolled and fastened as the last frame. Duration: eight seconds. The first line specifies alignment; the body describes the continuous change.

```txt
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] An overhead close-up begins with the open fabric tool pouch and hands positioned as in Picture 1. The hands fold the side flap inward, then roll the fabric steadily around the tools. One hand holds the roll while the other wraps and fastens the strap. End with the pouch, hands and framing aligned with Picture 2. Keep the same table, fabric and lighting throughout, with no cuts.

overall_soundscape: Soft fabric rubbing, gentle tool contact and a small fastening click.

non_diegetic_music: N/A
```

Put the alignment instruction first, followed by a blank line before the three fields. For multiple shots, assign the ending image to the final shot. Prefer one continuous shot between first and last frames; add cuts when explicitly needed.

## 3. Ref2VA: Reference Generation and Editing

Use these six fields in order:

1. `subject_definitions`: define referenced content, sources and roles.
2. `summary`: summarize the target, beginning with a task marker such as `[reference generation]`, `[video editing]` or `[video continuation]`.
3. `retention_analysis`: state what is preserved, changed, transferred or referenced.
4. `detailed_description`: describe the visuals and sound in playback order.
5. `overall_soundscape`: overall ambience and physical sounds.
6. `non_diegetic_music`: background score.

### Prompt template

```txt
subject_definitions:
...

summary:
[reference generation], [video editing] or [video continuation] ...

retention_analysis:
...

detailed_description:
[Shot 1] ...

overall_soundscape:
...

non_diegetic_music:
...
```

Use `<Video N>` for an original video asset. Use `<Subject N>` for the specific person, object, action, style or other content referenced from any asset. Use `<Picture N>` for concrete frame anchors and `<Audio N>` for supplied, enabled audio. Keep each label consistent throughout.

These markers express prompt intent, not deterministic editing, literal waveform copying, or exact voice reproduction. The `[video continuation]` marker describes a prompting intent; the released CLI has no dedicated seamless-extension mode.

### Preservation and Changes in Visual References

Relationships apply to **the content and features already defined**. If a character definition includes only face and hair, changing the background does not change whether those features are fully preserved. All four markers appear in the complete examples below.

| Marker | Meaning | Example |
| --- | --- | --- |
| fully_preserved | Keep all defined features. | Example 3 keeps the character's face, hair and outfit. |
| partially_preserved | Keep the referenced content while changing part of it. | Example 4 removes one person while retaining the rest of the scene and timing. |
| attribute_transfer | Apply referenced characteristics to a new subject or scene. | Example 3 applies the action to the target character and the camera movement to the new scene. |
| weak_reference | Use only broad characteristics as inspiration. | Example 5 borrows a dim, quiet atmosphere while redesigning the location and composition. |

### Explicit Edits and Selective Reference Use

To remove an object or person, identify it by appearance or position in `subject_definitions` and write `removed` in its `retention_analysis` entry. Describe the people, actions and environment to retain, and the background that fills the removed area. Use `partially_preserved` for a local edit that keeps the rest of the source video.

For replacement, bind the original entity to the replacement identity and source. State the changed attributes separately from the action, camera, timing and environment that must remain. Preserve only the defined features that are actually unchanged. Instantiate the replacement in the target shots and keep the original entity out of those shots.

To borrow composition, specify the angle, framing and subject placement. To borrow style, identify the image that supplies the visual treatment. To place a character-sheet identity in a new scene, describe the character's location and action and state, for example, 'Use only the character appearance; show one character in the scene, without the sheet panels, labels or alternative views.'

Visuals and sound can be edited separately. To retain source audio, specify how much of the recording to copy. To remove or replace a sound, name that sound and describe the change using the corresponding audio relationship marker.

### Copying and Referencing Audio

Reusing a recording and borrowing a voice's timbre serve different goals. All four markers also appear in the complete examples.

| Marker | Meaning | Example |
| --- | --- | --- |
| fully_copy | Use the entire source audio as the complete final soundtrack. | Example 4 retains speech and ambience unchanged, without adding sounds. |
| partially_copy | Reuse an excerpt or layer, or add or remove sounds after copying. | Example 6 copies one second of a bell recording and adds new footsteps. |
| reference | Generate new audio guided by specific source characteristics. | Example 5 uses a voice's timbre for a new line without copying the recording. |
| weak_reference | Borrow only the broad sound category or atmosphere. | Example 6 uses a relaxed-jazz feel with a new melody, rhythm and arrangement. |

### Example 3: Separate Character, Action and Camera References

Input: a character image, a cycling video and a camera-motion video. Uses `fully_preserved` and `attribute_transfer`.

```txt
subject_definitions:
<Subject 1> is the woman from Picture 1, retaining her face, hair and outfit.
<Subject 2> is the pedaling action from Video 1.
<Video 2> provides the forward camera trajectory.

summary:
[reference generation] <Subject 1> performs the cycling action from <Subject 2> as the camera follows the trajectory of <Video 2>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - retain her defined appearance.
<Subject 2> (appears in [Shot 1]): attribute_transfer - apply only the pedaling rhythm and posture to <Subject 1>; keep the source rider and unrelated scene content out of the target.
<Video 2>: attribute_transfer - apply its forward camera movement in a way that fits the target scene; retain the target character and setting.

detailed_description:
The video is live-action.
[Shot 1] <Subject 1> pedals a stationary exercise bike indoors. Her legs follow the cadence and posture of <Subject 2>, while both hands hold the handlebars. The camera moves forward along the trajectory supplied by <Video 2>, gradually closing the distance. She continues pedaling through the final frame.

overall_soundscape:
Rhythmic pedal clicks and quiet indoor room tone.

non_diegetic_music:
N/A
```

### Example 4: A Local Video Edit with the Original Soundtrack

Input: a source video showing two men and its enabled soundtrack. Remove only the left-foreground man visually; retain the other man's actions, environment, camera and complete original audio. Uses `partially_preserved`, `fully_preserved`, the explicit edit directive `removed`, and `fully_copy`.

```txt
subject_definitions:
<Video 1> is the source video showing two men in a desert setting.
<Subject 1> is the young man in the red hoodie from Video 1.
<Subject 2> is the man in the black jacket in the left foreground of Video 1, selected for removal.
<Audio 1> is the enabled original soundtrack of Video 1, containing speech and outdoor ambience.

summary:
[video editing + audio reuse] Remove <Subject 2> from <Video 1>. Retain <Subject 1>, his original actions, the desert environment, the source camera and timing, and <Audio 1>.

retention_analysis:
<Video 1>: partially_preserved - remove only <Subject 2>; preserve the remaining scene, desert background, timing and camera movement.
<Subject 1> (appears in [Shot 1]): fully_preserved - retain his identity, clothing and original actions.
<Subject 2>: removed - completely remove the man in the black jacket from every target frame.
<Audio 1>: fully_copy - use the complete source soundtrack as the complete final audio.

detailed_description:
The target video retains the source's live-action appearance.
[Shot 1] Follow <Video 1>'s original framing, camera movement and action timing. Only <Subject 1> remains visible. He continues his original gestures and speech. The area previously occupied by <Subject 2> shows continuous desert background matching the surrounding ground, lighting and camera movement. Keep <Audio 1> unchanged and synchronize <Subject 1>'s mouth movements with his source speech.

overall_soundscape:
Reuse the speech and outdoor ambience contained in <Audio 1>; add no new sounds.

non_diegetic_music:
N/A
```

### Example 5: Broad Visual Inspiration and a Specific Voice Reference

Input: character Picture 1, atmosphere Picture 2 and voice reference Audio 1. Uses visual `weak_reference` and audio `reference`.

```txt
subject_definitions:
<Subject 1> is the man from Picture 1, retaining his face, hair and dark jacket.
<Subject 2> is the dim, quiet atmosphere loosely inspired by Picture 2; its people and layout are not reused.
<Audio 1> is the low, calm voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] <Subject 1> looks toward the camera and delivers a new Chinese line using the vocal character of <Audio 1>, in a new setting loosely inspired by <Subject 2>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - retain the defined appearance.
<Subject 2> (appears in [Shot 1]): weak_reference - retain only the broad dim, quiet atmosphere; redesign the location and composition.
<Audio 1>: reference - use its timbre and measured delivery, without copying its recording or dialogue.

detailed_description:
The video is live-action in a dim interior with soft amber light.
[Shot 1] A medium close-up shows <Subject 1> standing beside an open doorway in a newly designed room with <Subject 2>'s dim, quiet atmosphere. <Subject 1> (S1) turns toward the camera, pauses and says in a low, calm voice, <d>[Chinese] 我们出发吧。</d> His lips follow the new line naturally. He then looks toward the doorway. The camera remains fixed.

overall_soundscape:
Quiet room tone and faint outdoor wind.

non_diegetic_music:
N/A
```

### Example 6: Reusing an Audio Excerpt with a New Score

Input: a cafe image, bell recording Audio 1 and jazz reference Audio 2. Uses `partially_copy` and audio `weak_reference`.

```txt
subject_definitions:
<Subject 1> is the cafe interior from Picture 1, retaining its counter, doorway and warm lighting.
<Audio 1> supplies a doorbell recording; use only its 2.00–3.00 second excerpt.
<Audio 2> is loose inspiration for a relaxed jazz score.

summary:
[reference generation + audio reuse + audio reference] A visitor enters <Subject 1>. Reuse the selected bell excerpt from <Audio 1> and generate footsteps, room ambience and a new jazz score.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - retain the defined cafe features.
<Audio 1>: partially_copy - reuse only the selected bell excerpt, mixed with new sound.
<Audio 2>: weak_reference - use only the broad relaxed-jazz character; compose a different melody, rhythm and arrangement.

detailed_description:
The video is live-action.
[Shot 1] A fixed wide shot shows <Subject 1>'s doorway and counter. A visitor pushes the door open. At that moment, play the selected excerpt from <Audio 1> once. The visitor walks to the counter with clearly audible footsteps and stops. Keep the doorway and counter visible throughout.

overall_soundscape:
Newly generated quiet cafe ambience and footsteps, with the copied bell excerpt synchronized to the door opening.

non_diegetic_music:
A newly composed, relaxed jazz score loosely inspired by <Audio 2>, with sparse piano and soft brushed drums. Do not copy the source melody or recording.
```

### Example 7: CG Style and Identity Transfer from a Character Sheet

Input: Picture 1 supplies a stylized 3D CG treatment. Picture 2 is a character sheet with several views of one wolf. The target is one wolf guarding a vial in a new alley scene, using Picture 1's style and Picture 2's identity. Show the character in the alley, without displaying the sheet layout.

```txt
subject_definitions:
<Subject 1> is the single wolf identity shown in the different views of Picture 2, retaining its recognizable face, coat markings and build.
<Subject 2> is the stylized 3D CG treatment of Picture 1; its depicted cast and layout are not target content.

summary:
[reference generation] One instance of <Subject 1>, rendered with <Subject 2>, guards a small vial in a coherent alley scene.

retention_analysis:
<Subject 1> (appears in [Shot 1]): attribute_transfer - transfer the selected wolf identity into the target scene; use the different views as identity evidence only.
<Subject 2> (appears in [Shot 1]): attribute_transfer - apply the reference's stylized form, shading and colour treatment to the target scene.

detailed_description:
The video uses the stylized 3D CG treatment of <Subject 2>.
[Shot 1] A medium-wide shot establishes a single alley space with exactly one wolf, <Subject 1>, and a small vial beside its front paws. <Subject 1> lowers its head toward the vial and shifts its body protectively around it, keeping the vial close while watching the alley entrance. Keep the face and vial readable against the background. The camera holds this scene throughout. The character-sheet panels, alternate views, labels and borders are not displayed.

overall_soundscape:
Quiet alley ambience, light paw contact and soft breathing.

non_diegetic_music:
N/A
```

The protective stance shown is one example. You can also describe the wolf holding the vial securely in its mouth or shielding it with a front paw.

### Example 8: Replace an Identity While Preserving the Action and Scene

Input: Video 1 is one continuous 3D CG shot of a man in a blue jacket walking across a stone courtyard. Picture 1 supplies a woman in a green coat. Replace the man with the woman while keeping the walking route, timing, camera, environment and CG treatment. This example generates new footsteps and courtyard ambience.

```txt
subject_definitions:
<Video 1> supplies the courtyard, walking route, action timing, continuous camera movement and 3D CG treatment.
<Subject 1> is the original man in the blue jacket from Video 1, selected for replacement.
<Subject 2> is the woman from Picture 1, retaining her face, hair and green coat.

summary:
[video editing] Replace <Subject 1> with <Subject 2> while preserving the original walk, courtyard, camera and visual treatment of <Video 1>.

retention_analysis:
<Video 1>: partially_preserved - keep the environment, walking route, timing, continuous camera movement and CG treatment; change only the selected character identity and outfit.
<Subject 1>: removed - replace the original man with <Subject 2> throughout the target shot.
<Subject 2> (appears in [Shot 1]): attribute_transfer - apply her defined appearance to the original character's walking performance; the background of Picture 1 is not transferred.

detailed_description:
Retain the 3D CG appearance of Video 1.
[Shot 1] <Subject 2> occupies the original walker's starting position and walks along the same route at the same pace. She wears the green coat from Picture 1. Follow the source camera's continuous movement and retain the stone courtyard and its lighting. Show one walker throughout, ending at the original walk's final position.

overall_soundscape:
Generate footsteps synchronized with the walk and quiet courtyard ambience.

non_diegetic_music:
N/A
```

## 4. Writing Visuals, Camera and Sound

These rules apply to T2VA, FL2VA and Ref2VA. The main description field is `integrated_multimodal_description` for basic tasks and `detailed_description` for Ref2VA. Shot, camera, speech, text and sound conventions are shared.

### Develop Visuals and Action in Playback Order

Describe visible and audible content. Establish style, subjects, positions and composition, then describe actions, changes and results. For image-driven tasks, begin from the state established by the input image.

For a handoff, name the giver and receiver and state who subsequently holds and uses the object. For example: she takes out the stylus and gives it to her colleague; the colleague receives it and signs the document. To show stabilization, describe the object visibly swaying before the character reaches out to steady it. Establish a track, path or support before describing movement that depends on it.

State the number of people and their positions directly, for example, 'Four men stand around the car.' Alternate views, aliases and props do not add characters. In each shot, say who appears, where they are and what they do. A prop-only shot can omit the cast.

### Keep Shot Numbers and Cut Times in Order

Begin with `[Shot 1]`, without a timestamp. Introduce later shots with increasing cut times within the video duration, such as `[Shot 2] At 00:03.000, ...`. Cuts introduce new information; prefer camera movement for small changes in distance or angle.

For an uninterrupted take, write `one continuous shot`. For a fixed viewpoint, write `The camera remains fixed.` Describe multiple shots separately as `[Shot 1]`, `[Shot 2]` and so on, with cut times inside the video duration. Small changes in distance or angle can be described as camera movement.

### Specify Camera Motion, Amplitude and Speed

Write camera movement as a natural sentence, adding amplitude and speed when useful. Common movements include pushing in or pulling out, panning, trucking, tilting, moving up or down, orbiting, tracking and holding a static shot. A `push in` moves the camera; a `zoom in` changes focal length.

Describe the camera's position relative to the subject and its starting and ending views, for example, 'The camera slowly pushes in from in front of the character, ending on a facial close-up.' When borrowing camera movement, specify that only the camera motion is transferred and describe the target character and scene. For a fixed composition, write `The camera remains fixed.`

### Keep Speakers and Dialogue Consistent

Assign stable IDs such as `(S1)` and `(S2)` to actual vocal sources; use `(S1,S2)` for shared speech. Keep identity and delivery outside the dialogue tags. Inside `<d>[Chinese] 我们出发吧。</d>`, include only the language and original words. For narration, use `says in an off-screen voiceover`, followed by a statement that the corresponding on-screen character's lips remain closed. Use `<scenetrans>` for dialogue crossing a cut and `<cutoff>` for speech truncated by the video ending.

### Put On-Screen Text in Double Quotation Marks

Preserve the original wording and punctuation of signs, subtitles and labels. Use English double quotation marks, for example `a sign reading "营业中"`.

### Summarize Overall Sound in overall_soundscape

Use 1–4 sentences for ambience, physical sounds and non-verbal human sounds. Dialogue, singing and diegetic music belong in the shot description and should not be repeated here. Use `N/A` only when the entire video is explicitly intended to be silent.

### Describe Audience-Only Score in non_diegetic_music

Use 1–3 sentences to describe instrumentation, speed, rhythm and volume changes. Write `N/A` when there is no score. A radio or live performance audible to characters belongs in the shot description.

## 5. Common fixes

- **Unclear action endpoint:** Describe the actions in order and state the final visible result. For example, change “Stir the coffee” to “After stirring, place the spoon on the saucer and release it.”
- **Unclear shot breakdown:** Put different scenes in separate shots and specify each view and cut time. For example, change “Study a map in a cafe, then arrive at the coast” to “`[Shot 1]` Study the map in the cafe; `[Shot 2]` At 00:03.000, cut to holding the map on the coast.”
- **Mixed reference roles:** Specify what each asset supplies and limit the character, composition or style features being referenced. For example, change “Use Pictures 1 and 2” to “Use Picture 1 only for composition and Picture 2 only for character appearance.”
- **Incomplete deletion or replacement:** Identify the edited object and what stays. Mark a deleted object `removed` in `retention_analysis`; for replacement, identify the replacement's reference source. For example, change “Remove the person on the left” to “Left-side man in the black jacket: `removed`; retain the other person, actions and background.”
- **The entire reference image appears:** Limit the appearance features to use and describe the character's new scene, keeping the reference layout out of the shot. For example, change “Use this character sheet” to “Use only the character appearance; show one character in a cafe, without the panels, labels or alternative views.”
- **Wrong action owner or causal order:** Name who performs each action and describe events in sequence. For example, change “Pass the stylus and sign” to “She hands the stylus to her colleague; the colleague receives it and signs.” Change “Steady the vase” to “The vase wobbles first, and the character then reaches out to steady it.”

## 6. Running the examples with the CLI

The commands below run from the repository root. Set `MODEL_REPO` and the immutable `MODEL_REVISION` as described in the README. The `inputs/` paths are placeholders for your own local media; those inputs are not bundled. Adapt the prompt descriptions to your actual media. These examples illustrate how to make requests, rather than reproduce the gallery videos.

Run the bundled text-only example (a red cup beside an open window):

```bash
pika-h3-pro --model-repo "$MODEL_REPO" --revision "$MODEL_REVISION" --task t2va --prompt-file examples/prompts/t2va.txt --resolution 480p --aspect 16:9 --duration 5 --seed 42 --output outputs/text-only.mp4
```

### Endpoint images

Complete five-second examples are included for each mode:

| Mode | Prompt file | Local media to supply |
|---|---|---|
| First frame (`i2va`) | [i2va.txt](../examples/prompts/i2va.txt) | A red cup on a wooden table beside an open window. |
| Last frame (`l2va`) | [l2va.txt](../examples/prompts/l2va.txt) | A blue bowl resting on a workbench, with the potter's hands withdrawn. |
| Both endpoints (`fl2va`) | [fl2va.txt](../examples/prompts/fl2va.txt) | A closed pocket watch and the same watch open, held in the same hand above the same workbench. |

```bash
pika-h3-pro --model-repo "$MODEL_REPO" --revision "$MODEL_REVISION" --task i2va --prompt-file examples/prompts/i2va.txt --first-image inputs/cup-first.png --resolution 480p --aspect 16:9 --duration 5 --seed 42 --output outputs/first-frame.mp4

pika-h3-pro --model-repo "$MODEL_REPO" --revision "$MODEL_REVISION" --task l2va --prompt-file examples/prompts/l2va.txt --last-image inputs/bowl-last.png --resolution 480p --aspect 16:9 --duration 5 --seed 42 --output outputs/last-frame.mp4

pika-h3-pro --model-repo "$MODEL_REPO" --revision "$MODEL_REVISION" --task fl2va --prompt-file examples/prompts/fl2va.txt --first-image inputs/watch-closed.png --last-image inputs/watch-open.png --resolution 480p --aspect 16:9 --duration 5 --seed 42 --output outputs/both-frames.mp4
```

For `i2va` and `l2va`, the single supplied image is `<Picture 1>`. For `fl2va`, the first image is `<Picture 1>` and the last image is `<Picture 2>`. These endpoint examples each describe one continuous shot.

The encoded clip duration follows the model's frame grid; endpoint instructions express the intended opening and ending rather than frame-exact timing guarantees.

### Reference inputs and labels

Keep labels consistent: `<Subject N>` identifies content you define in the prompt, `<Picture N>` identifies supplied images, `<Video N>` identifies supplied videos, and `<Audio N>` identifies supplied audio. Repeated `--ref` options preserve their command-line order, but image, video and audio labels each have their own counter starting at 1. A video with a soundtrack consumes an audio label as well as a video label. Do not refer to audio that was not supplied.

For a reference request, include at least one image or video. Audio can accompany visual references. Give each asset a distinct role—for example character appearance, action, scene layout, or camera motion—rather than asking all references to control everything.

A simple image-reference example uses [ref2va.txt](../examples/prompts/ref2va.txt) to request a portrait in a softly lit room:

```bash
pika-h3-pro --model-repo "$MODEL_REPO" --revision "$MODEL_REVISION" --task ref2va --prompt-file examples/prompts/ref2va.txt --ref image:inputs/portrait.png --resolution 480p --aspect 16:9 --duration 5 --seed 42 --output outputs/reference-image.mp4
```

For an image, a video with sound, and separate audio, use [ref2va-mixed.txt](../examples/prompts/ref2va-mixed.txt). Supply a dancer portrait, a short dance video **with an audio track**, and a percussion recording. Use 2–5-second video/audio inputs for this five-second example.

```bash
pika-h3-pro --model-repo "$MODEL_REPO" --revision "$MODEL_REVISION" --task ref2va --prompt-file examples/prompts/ref2va-mixed.txt --ref image:inputs/dancer.png --ref video:inputs/dance-with-sound.mp4 --ref audio:inputs/percussion.wav --resolution 480p --aspect 16:9 --duration 5 --seed 42 --output outputs/reference-mixed.mp4
```

The exact label mapping for that command is:

| Input | Prompt label |
|---|---|
| `inputs/dancer.png` | `<Picture 1>`; the prompt names its dancer `<Subject 1>`. |
| Visual content of `inputs/dance-with-sound.mp4` | `<Video 1>` |
| Soundtrack of `inputs/dance-with-sound.mp4` | `<Audio 1>` |
| `inputs/percussion.wav` | `<Audio 2>` |

If the video has no audio track, the standalone recording becomes `<Audio 1>`: update the prompt accordingly. The CLI does not have a per-video audio-disable flag. Changing the order of the references can also change audio numbering. Audio intent markers are descriptive instructions; they do not implement exact audio copying or source separation.

The complete prompts above describe reference media that you supply; they do not bundle those inputs. Match each description and label to your actual files. For example, a source video's soundtrack receives its own audio label automatically; use a video without an audio track if you do not want to supply that soundtrack.

This guidance is usable without a prompt-rewriter service. All command examples use distinct output filenames; choose a new filename when rerunning because the CLI refuses to overwrite existing outputs. Language tags express requested speech; they are not a validated language-support list.
