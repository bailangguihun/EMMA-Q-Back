from pathlib import Path
import pretty_midi

OUT = Path("web_app/samples")
OUT.mkdir(parents=True, exist_ok=True)

def write_midi(pm: pretty_midi.PrettyMIDI, name: str):
    pm.write(str(OUT / name))
    print("Wrote", OUT / name)

def demo_scale():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    t = 0.0
    dur = 0.35
    # C major up + down
    pitches = [60,62,64,65,67,69,71,72, 71,69,67,65,64,62,60]
    for p in pitches:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=t, end=t+dur))
        t += dur
    pm.instruments.append(inst)
    write_midi(pm, "demo_scale_C_major.mid")

def demo_chords():
    pm = pretty_midi.PrettyMIDI(initial_tempo=100)
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    dur = 1.6
    # C - Am - F - G
    chords = [
        [60,64,67],      # C
        [57,60,64],      # Am
        [53,57,60],      # F
        [55,59,62],      # G
    ]
    for ch in chords * 2:
        for p in ch:
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=t, end=t+dur))
        t += dur
    pm.instruments.append(inst)
    write_midi(pm, "demo_chord_progression.mid")

def demo_melody_with_drum():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)

    piano = pretty_midi.Instrument(program=0)
    drum = pretty_midi.Instrument(program=0, is_drum=True)

    # melody (8 bars feel)
    t = 0.0
    step = 0.25
    melody = [60,62,64,67, 64,62,60,62, 64,65,67,69, 67,65,64,62]
    for p in melody:
        piano.notes.append(pretty_midi.Note(velocity=92, pitch=p, start=t, end=t+step*0.9))
        t += step

    # simple drum: kick(36) on 1&3, snare(38) on 2&4, hihat(42) every 1/8
    t = 0.0
    total = 4.0  # seconds
    while t < total:
        # hihat
        drum.notes.append(pretty_midi.Note(velocity=55, pitch=42, start=t, end=t+0.05))
        # kick/snare on beats (assuming 120bpm -> beat=0.5s)
        if abs((t % 0.5) - 0.0) < 1e-6:
            beat_index = int(t / 0.5) % 4
            if beat_index in (0, 2):
                drum.notes.append(pretty_midi.Note(velocity=80, pitch=36, start=t, end=t+0.05))
            if beat_index in (1, 3):
                drum.notes.append(pretty_midi.Note(velocity=75, pitch=38, start=t, end=t+0.05))
        t += 0.25

    pm.instruments.extend([piano, drum])
    write_midi(pm, "demo_melody_with_drum.mid")

if __name__ == "__main__":
    demo_scale()
    demo_chords()
    demo_melody_with_drum()