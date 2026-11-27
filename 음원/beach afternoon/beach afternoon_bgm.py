"""Original BGM for beach afternoon.mp4 (20 s): a sunlit beach at midday, small waves breaking.

Clear, refreshing summer pop in E major: chiming clean-guitar arpeggios, a bright pluck-lead that
opens into long, soaring notes, driving 8th-note bass, pop drums and sparkling sun-glint bells.
The chorus runs on the bright IV-V-iii-vi "summer" progression and climbs to a long B5 over B
before landing on E.

Sync (the HUD's "breaking" flag): waves break at 0.50, 4.77, 9.20, 14.10 and 19.17 s. At 108.35 BPM
the 4.77 s wave opens the chorus on a downbeat, the biggest wave (9.20 s) opens the lifted second
half of the chorus (crash, wide pad, higher melody), the 14.10 s wave is met by a falling sparkle
cascade, and the 19.17 s wave lands exactly on beat 3 of the last bar (final sparkles).
No noise beds and no wave sounds.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261121)

BREAKS = [0.50, 4.77, 9.20, 14.10, 19.17]
BEAT = (9.20 - 4.77) / 8          # 0.55375 s = 108.35 BPM
BAR = 4 * BEAT
B0 = 4.77 - 2 * BAR               # bar 0 starts at 0.34 s; bar 2 = 4.77 s; bar 4 = 9.20 s; bar 8 = 18.06 s


def tb(bar, pos=0.0):
    return B0 + bar * BAR + pos * BEAT


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def tt(T):
    return np.arange(int(T * SR)) / SR


def fft_filter(x, lo=None, hi=None, order=2):
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n, 1 / SR)
    H = np.ones_like(f)
    if hi is not None:
        H /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo is not None:
        H /= np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(X * H, n, axis=-1)


def noise(n):
    return rng.standard_normal(n)


STEMS = {}
GAIN = dict(lead=0.30, pluck=0.16, octave=0.06, gtr=0.19, piano=0.13, bass=0.16, pad=0.055, bells=0.12,
            sparkle=0.21, swell=0.08, kick=0.30, snare=0.36, hat=0.10, crash=0.07)
SEND = dict(lead=.30, pluck=.30, octave=.45, gtr=.30, piano=.22, bass=.03, pad=.40, bells=.50, sparkle=.50,
            swell=.40, kick=.04, snare=.18, hat=.08, crash=.20)


def place(name, sig, t, pan=0.0, gain=1.0):
    buf = STEMS.setdefault(name, np.zeros((2, N), np.float32))
    i = int(round(t * SR))
    if sig.ndim == 1:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * np.sqrt(2)
    else:
        st = sig.copy()
    k = min(st.shape[1], max(int(0.01 * SR), int(0.04 * st.shape[1])))
    st[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    if i < 0:
        st, i = st[:, -i:], 0
    n = min(st.shape[1], N - i)
    if n > 0:
        buf[:, i:i + n] += (st[:, :n] * gain).astype(np.float32)


def echo(name, sig, t, pan, gain, delay=0.75 * BEAT, fb=0.35, taps=2):
    place(name, sig, t, pan, gain)
    for k in range(1, taps + 1):
        place(name, fft_filter(sig, hi=5000 / k), t + delay * k, -0.6 if k % 2 else 0.6, gain * fb ** k)


# ---------------------------------------------------------------- instruments
def eguitar(m, dur, vel=0.5):
    """Clean electric guitar note: bright plucked partials (pickup near the bridge), ringing."""
    f = float(mtof(m)); T = dur + 0.5; t = tt(T)
    tau1 = float(np.interp(m, [40, 64, 88], [2.6, 1.6, 0.8]))
    out = np.zeros_like(t)
    for n in range(1, 19):
        if f * n > 9000:
            break
        a = abs(np.sin(n * np.pi * 0.13)) / n ** 0.95
        out += a * np.sin(2 * np.pi * f * n * t + rng.uniform(0, 0.3)) * np.exp(-t / (tau1 / (1 + 0.28 * (n - 1) ** 1.15)))
    env = np.minimum(1, t / 0.0015) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.08))
    return out * env * vel


def piano(m, dur, vel=0.5):
    f = float(mtof(m)); T = dur + 0.9; t = tt(T)
    t60 = float(np.interp(m, [36, 60, 84, 100], [7.0, 4.0, 1.8, 0.8]))
    out = np.zeros_like(t)
    for n in range(1, 14):
        fn = f * n * np.sqrt(1 + 0.0004 * n * n)
        if fn > 12000:
            break
        a = (1.0 / n ** 1.15) * ((0.30 + 0.55 * vel) ** (n - 1))
        tau = t60 / 6.9 / (1 + 0.45 * (n - 1))
        out += a * np.sin(2 * np.pi * fn * t) * (0.65 * np.exp(-t / (tau * 0.35)) + 0.35 * np.exp(-t / tau))
    out *= np.minimum(1, t / 0.002) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.12))
    return out * vel


def ebass(m, dur, vel=0.7):
    f = float(mtof(m)); T = dur + 0.15; t = tt(T)
    out = sum((1.0 / k ** 1.3) * np.sin(2 * np.pi * f * k * t) * ((1.0 if k <= 2 else 0.3 / k) + (1 - (1.0 if k <= 2 else 0.3 / k)) * np.exp(-t * (6 + 5 * k))) for k in range(1, 12) if f * k < 3500)
    env = np.minimum(1, t / 0.004) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.04))
    return np.tanh(1.2 * out) * env * vel


def bell(m, vel=0.3):
    """Sun glint: a small, clear bell (celesta-like)."""
    f = float(mtof(m)); t = tt(1.6)
    out = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.7) + 0.18 * np.sin(2 * np.pi * 2.0 * f * t) * np.exp(-t / 0.18)
    out += 0.06 * np.sin(2 * np.pi * 3.0 * f * t) * np.exp(-t / 0.06)
    return out * np.minimum(1, t / 0.001) * vel


def supersaw(notes, dur, vel=0.5, attack=0.25):
    T = dur + 1.2; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det in (-14, -6, 0, 6, 14):
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = np.zeros_like(t); ph = rng.uniform(0, 6.28)
            for k in range(1, 30):
                if f * k > 7000:
                    break
                s += np.sin(2 * np.pi * f * k * t + ph * k) / k
            w = 0.5 + det / 34
            L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.35))
    return np.vstack([L * env, R * env]) * vel / len(notes)


def lead_line(notes, glide=0.06):
    """Bright, clear synth voice: soft-saw harmonics that are brighter at each onset, legato glides,
    delayed vibrato. notes = (on, off, midi, vel)."""
    L = int(DUR * SR) + SR
    pitch = np.full(L, np.nan); amp = np.zeros(L); vibe = np.zeros(L); bright = np.zeros(L)
    for i, (on, off, m, v) in enumerate(notes):
        i0 = int(on * SR)
        nxt = int(notes[i + 1][0] * SR) if i + 1 < len(notes) else L
        end = min(nxt, int((off + 0.5) * SR), L)
        tl = np.arange(end - i0) / SR
        prev = notes[i - 1] if i > 0 else None
        legato = prev is not None and on - prev[1] < 0.05
        p = np.full(end - i0, float(m))
        if legato:
            g = min(len(p), int(glide * SR))
            p[:g] = m + (prev[2] - m) * (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, g)))
        pitch[i0:end] = p
        start = 0.8 if legato else 0.0
        env = start + (1 - start) * np.sin(np.minimum(1, tl / 0.025) * np.pi / 2) ** 2
        env *= 0.88 + 0.12 * np.exp(-tl / 0.15)
        env *= np.where(tl < off - on, 1.0, np.exp(-(tl - (off - on)) / 0.14))
        amp[i0:end] = env * v
        vibe[i0:end] = np.clip((tl - 0.22) / 0.35, 0, 1)
        bright[i0:end] = 0.62 + 0.38 * np.exp(-tl / 0.22)
    idx = np.where(~np.isnan(pitch), np.arange(L), 0); np.maximum.accumulate(idx, out=idx)
    pitch = pitch[idx]; pitch[np.isnan(pitch)] = 69
    t = np.arange(L) / SR
    fr = mtof(pitch) * 2 ** (vibe * 17 * np.sin(2 * np.pi * 5.4 * t) / 1200)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    tone = np.zeros(L)
    for k in range(1, 9):
        tone += (1 / k ** 1.25) * bright ** (k - 1) * np.sin(k * ph)
    return (tone * amp)[:int(DUR * SR)]


def kick(vel=1.0):
    t = tt(0.4)
    ph = 2 * np.pi * np.cumsum(52 + 95 * np.exp(-t / 0.022)) / SR
    return np.tanh(1.3 * np.sin(ph) * np.exp(-t / 0.19)) * np.minimum(1, t / 0.0015) * vel


def snare(vel=0.8):
    t = tt(0.45)
    tone = np.sin(2 * np.pi * np.cumsum(190 * (1 + 0.35 * np.exp(-t / 0.01))) / SR) * np.exp(-t / 0.07)
    nz = fft_filter(noise(len(t)), lo=1200, hi=8000); nz /= nz.std()
    clap = sum(np.where((t >= d) & (t < d + 0.006), 1.0, 0.0) for d in (0.0, 0.009, 0.018))
    body = 0.8 * tone + 0.5 * np.tanh(0.7 * nz) * (np.exp(-t / 0.12) + 0.6 * clap)
    return body * np.minimum(1, t / 0.0008) * vel


def hat(vel=0.3, open_=False):
    T = 0.3 if open_ else 0.06; t = tt(T)
    nz = fft_filter(noise(len(t)), lo=7000, hi=14000); nz /= nz.std()
    return nz * np.exp(-t / (0.09 if open_ else 0.014)) * np.minimum(1, t / 0.0008) * vel


def crash(vel=0.6):
    t = tt(2.6)
    nz = fft_filter(noise(len(t)), lo=3500, hi=13000); nz /= nz.std()
    metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in [3100, 4500, 6200]) / 3
    return (np.tanh(0.6 * nz) / 0.6 + 0.25 * metal) * np.exp(-t / 0.9) * np.minimum(1, t / 0.002) * vel


# ---------------------------------------------------------------- harmony
VO = {   # bass root, guitar arpeggio notes (low -> high), piano/pad voicing
    'Eadd9':  (40, [52, 59, 66, 68, 71], [64, 66, 68, 71]),
    'C#m7':   (37, [49, 56, 64, 68, 71], [64, 68, 71, 73]),
    'Bsus4':  (35, [47, 54, 64, 66, 71], [64, 66, 71, 76]),
    'B':      (35, [47, 54, 63, 66, 71], [63, 66, 71, 75]),
    'Amaj7':  (33, [45, 52, 61, 64, 68], [61, 64, 68, 71]),
    'G#m7':   (44, [44, 51, 59, 63, 66], [63, 66, 68, 71]),
    'F#m7':   (42, [42, 49, 57, 64, 69], [61, 64, 69, 73]),
    'Eend':   (40, [52, 59, 64, 68, 71, 76], [64, 68, 71, 76, 78]),
}
PROG = [(0, 0, 'Eadd9'), (1, 0, 'C#m7'), (1, 2, 'Bsus4'), (1, 3, 'B'),
        (2, 0, 'Amaj7'), (2, 2, 'B'), (3, 0, 'G#m7'), (3, 2, 'C#m7'),
        (4, 0, 'Amaj7'), (4, 2, 'B'), (5, 0, 'G#m7'), (5, 2, 'C#m7'),
        (6, 0, 'F#m7'), (6, 2, 'G#m7'), (7, 0, 'Amaj7'), (7, 2, 'Bsus4'), (7, 3, 'B'), (8, 0, 'Eend')]


def segs():
    for i, (bar, pos, c) in enumerate(PROG):
        nb, npos = (PROG[i + 1][0], PROG[i + 1][1]) if i + 1 < len(PROG) else (9, 0)
        yield bar, pos, c, (nb - bar) * 4 + (npos - pos)


E4, F4s, G4s, A4, B4, C5s, D5s, E5, F5s, G5s, A5, B5 = 64, 66, 68, 69, 71, 73, 75, 76, 78, 80, 81, 83

# ---------------------------------------------------------------- melody (bar, beat, beats, midi)
MEL = [
    (1, 3.0, 0.5, B4), (1, 3.5, 0.5, D5s),
    # chorus A (from the 4.77 s wave): Amaj7 B | G#m7 C#m7
    (2, 0, 0.5, E5), (2, 0.5, 1.0, G5s), (2, 1.5, 1.0, F5s), (2, 2.5, 0.5, E5), (2, 3, 0.5, D5s), (2, 3.5, 0.5, E5),
    (3, 0, 1.5, B4), (3, 1.5, 0.5, C5s), (3, 2, 0.5, E5), (3, 2.5, 1.0, G5s), (3, 3.5, 0.5, F5s),
    # chorus A' (from the biggest wave, 9.20 s): reaches higher
    (4, 0, 0.5, E5), (4, 0.5, 1.0, B5), (4, 1.5, 1.0, A5), (4, 2.5, 0.5, G5s), (4, 3, 0.5, F5s), (4, 3.5, 0.5, G5s),
    (5, 0, 1.5, D5s), (5, 1.5, 0.5, E5), (5, 2, 1.0, G5s), (5, 3, 0.5, F5s), (5, 3.5, 0.5, E5),
    # climb: F#m7 G#m7 | Amaj7 B -> the long B5
    (6, 0, 0.5, C5s), (6, 0.5, 0.5, E5), (6, 1, 0.5, F5s), (6, 1.5, 0.5, A5), (6, 2, 1.0, G5s), (6, 3, 0.5, F5s), (6, 3.5, 0.5, G5s),
    (7, 0, 1.0, A5), (7, 1, 0.5, G5s), (7, 1.5, 0.5, A5), (7, 2, 2.0, B5),
    (8, 0, 3.2, G5s),
]
notes = []
for bar, p, d, m in MEL:
    on = tb(bar, p); off = on + d * BEAT * (0.92 if d <= 0.5 else 0.97)
    v = 0.50 if bar <= 3 else 0.58
    notes.append((on, min(off, DUR - 0.2), m, v))
    place('pluck', eguitar(m, min(d * BEAT, 0.5), 0.55), on, pan=0.05)         # clear attack layer
    if bar >= 4:
        place('octave', bell(m + 12, 0.5), on, pan=0.15)
lead = lead_line(notes)
place('lead', lead, 0.0, pan=0.0)
place('lead', fft_filter(lead, hi=3500), 0.75 * BEAT, pan=-0.55, gain=0.22)   # dotted-8th ping-pong
place('lead', fft_filter(lead, hi=2500), 1.5 * BEAT, pan=0.55, gain=0.10)

# guitar arpeggios: 8ths (16ths in the climb), chorus effect at mix
PAT = [0, 1, 2, 3, 4, 3, 2, 1]
for bar, pos, c, span in segs():
    if bar == 8:
        for j, m in enumerate(VO[c][1]):
            place('gtr', eguitar(m, 2.4, 0.55), tb(8, 0) + j * 0.035, pan=float(-0.4 + 0.16 * j))
        continue
    arp = VO[c][1]
    step = 0.25 if bar in (6, 7) else 0.5
    for k in range(int(round(span / step))):
        p = pos + k * step
        idx = PAT[k % 8] if step == 0.5 else [0, 2, 3, 4, 2, 3, 1, 3][k % 8]
        vel = (0.62 if abs(p % 1) < 1e-6 else 0.48) * (0.85 if bar <= 1 else 1.0)
        place('gtr', eguitar(arp[idx], step * BEAT * 1.9, vel), tb(bar, p), pan=float(-0.35 + 0.12 * idx))

# piano: syncopated pop stabs in the chorus
for bar, pos, c, span in segs():
    if bar < 2:
        continue
    v = VO[c][2]
    if bar == 8:
        for j, m in enumerate(v + [83]):
            place('piano', piano(m, 2.8, 0.45), tb(8, 0) + j * 0.02, pan=float(-0.2 + 0.1 * j))
        place('piano', piano(VO[c][0] + 12, 2.8, 0.5), tb(8, 0))
        continue
    hits = [(0, 0.45, 0.5), (1.5, 0.35, 0.4)] if span >= 2 else [(0, 0.45, 0.5)]
    for hp, d, vel in hits:
        if hp < span:
            for m in v:
                place('piano', piano(m, d * BEAT * 2, vel), tb(bar, pos + hp), pan=0.2)

# bass: whole notes in the intro, driving 8ths in the chorus
for bar, pos, c, span in segs():
    r = VO[c][0]
    r = r if r >= 33 else r + 12
    if bar == 0:
        place('bass', ebass(r, span * BEAT * 0.98, 0.6), tb(bar, pos)); continue
    if bar == 1:
        place('bass', ebass(r, span * BEAT * 0.95, 0.65), tb(bar, pos)); continue
    if bar == 8:
        place('bass', ebass(r, 1.9, 0.8), tb(8, 0)); continue
    for k in range(int(span * 2)):
        p = pos + k * 0.5
        m = r + 12 if (k == int(span * 2) - 1 and bar >= 4) else r
        place('bass', ebass(m, BEAT * 0.45, 0.75 if k % 2 == 0 else 0.6), tb(bar, p))

# pad: airy in the intro, wide and open from the biggest wave on
for bar, pos, c, span in segs():
    v = VO[c][2]
    if bar in (0, 1):
        place('pad', supersaw(v, span * BEAT + 0.05, 0.35, attack=0.8 if bar == 0 else 0.3), tb(bar, pos))
    elif bar >= 4:
        place('pad', supersaw(v + [v[0] + 12], span * BEAT + 0.05, 0.75, attack=0.12), tb(bar, pos))
    elif bar >= 2:
        place('pad', supersaw(v, span * BEAT + 0.05, 0.45, attack=0.2), tb(bar, pos))

# sun glints: sparse little bells on a 16th grid with ping-pong echoes (E major, not pentatonic)
SCALE = [76, 78, 80, 81, 83, 85, 87, 88, 90, 92]
dens = lambda t: float(np.interp(t, [0, 4.7, 4.8, 9.1, 9.2, 17.9, 18.1, 20], [0.40, 0.40, 0.15, 0.15, 0.12, 0.12, 0.45, 0.45]))
for k in range(int((DUR - 0.6) / (BEAT / 4))):
    t = tb(0) + k * BEAT / 4
    if 0.2 < t < 19.6 and rng.random() < dens(t):
        echo('bells', bell(int(rng.choice(SCALE)), rng.uniform(0.18, 0.34)), t + rng.uniform(0, 0.02),
             pan=float(rng.uniform(-.8, .8)), gain=1.0, delay=0.75 * BEAT, fb=0.33, taps=2)

# wave accents: a rising sparkle into the first, a falling cascade at 14.10, final sparkles at 19.17
for j, m in enumerate([64, 68, 71, 76, 80, 83]):
    place('sparkle', bell(m, 0.30), 0.03 + j * 0.05, pan=float(-0.6 + 0.24 * j))
for j, m in enumerate([95, 92, 90, 88, 85, 83, 80, 78, 76]):
    place('sparkle', bell(m, 0.34 - 0.02 * j), BREAKS[3] + j * 0.065, pan=float(0.7 - 0.16 * j))
for j, m in enumerate([83, 88, 92, 95, 100]):
    place('sparkle', bell(m, 0.30 - 0.03 * j), BREAKS[4] + j * 0.07, pan=float(-0.5 + 0.25 * j))


# reversed chord swells into the 4.77 s and 9.20 s waves (tonal, no noise)
def rev_swell(c, T=1.3):
    sig = sum(piano(m, T, 0.5) for m in VO[c][2])
    sig = sig[:int(T * SR)][::-1]
    return sig * np.linspace(0, 1, len(sig)) ** 1.5


place('swell', rev_swell('Amaj7', 1.2), BREAKS[1] - 1.2, pan=-0.2)
place('swell', rev_swell('Amaj7', 1.6), BREAKS[2] - 1.6, pan=0.2, gain=1.3)

# drums
for bar in range(1, 8):
    lv = 0.8 if bar == 1 else 1.0
    if bar == 1:
        for pos in (0, 2):
            place('kick', kick(0.7), tb(1, pos))
        for k in range(8):
            place('hat', hat(0.35 if k % 2 == 0 else 0.22), tb(1, k * 0.5) + 0.01, pan=0.3)
        for k, pos in enumerate(np.arange(3.0, 4.0, 0.25)):
            place('snare', snare(0.35 + 0.12 * k), tb(1, pos), pan=-0.05)
        continue
    for pos in ((0, 1.5, 2.5) if bar < 6 else (0, 1, 1.5, 2, 2.5, 3)):
        place('kick', kick(0.95 if pos in (0, 2) else 0.75), tb(bar, pos))
    for pos in (1, 3):
        if bar == 7 and pos == 3:
            continue
        place('snare', snare(0.85), tb(bar, pos) + 0.005, pan=-0.05)
    step = 0.5 if bar < 4 else 0.25
    for k in range(int(4 / step)):
        p = k * step
        if bar == 7 and p >= 3:
            continue
        if p % 1 == 0.5 and bar >= 4 and p == 3.5:
            place('hat', hat(0.5, open_=True), tb(bar, p) + 0.01, pan=0.3)
            continue
        place('hat', hat((0.42 if p % 0.5 == 0 else 0.26) * lv), tb(bar, p) + 0.01, pan=0.3)
for k, pos in enumerate(np.arange(3.0, 4.0, 0.25)):
    place('snare', snare(0.45 + 0.13 * k), tb(7, pos), pan=-0.05)
for t0, v in ((tb(2), 0.7), (tb(4), 1.0), (tb(8), 0.9)):
    place('crash', crash(v), t0, pan=0.25)
place('kick', kick(1.0), tb(8))


# ---------------------------------------------------------------- mix
def reverb(x2, T=2.4):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.4)
        ir = (fft_filter(nz, hi=8000) * w + fft_filter(nz, hi=3500) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.025 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


def chorus(x2, depth_ms=2.2, rate=0.9):
    n = x2.shape[1]; idx = np.arange(n, dtype=float)
    out = np.empty_like(x2)
    for c, ph in ((0, 0.0), (1, np.pi / 2)):
        d = (7 + depth_ms * (1 + np.sin(2 * np.pi * rate * idx / SR + ph)) / 2) / 1000 * SR
        out[c] = 0.7 * x2[c] + 0.5 * np.interp(idx - d, idx, x2[c])
    return out


STEMS['gtr'] = chorus(fft_filter(STEMS['gtr'].astype(float), lo=160)).astype(np.float32)
STEMS['pad'] = fft_filter(fft_filter(STEMS['pad'].astype(float), hi=4200), lo=260).astype(np.float32)
STEMS['piano'] = fft_filter(STEMS['piano'].astype(float), lo=140).astype(np.float32)
STEMS['swell'] = fft_filter(STEMS['swell'].astype(float), lo=200).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, tb(2)), chorusA=(tb(2), tb(4)), chorusB=(tb(4), tb(6)), climb=(tb(6), tb(8)), end=(tb(8), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.28 * reverb(snd)
mix = fft_filter(mix, lo=32)
_X = np.fft.rfft(mix, axis=-1); _f = np.fft.rfftfreq(mix.shape[-1], 1 / SR); _r = (_f / 6000) ** 2
mix = np.fft.irfft(_X * (1 + 0.33 * _r / (1 + _r)), mix.shape[-1], axis=-1)        # a little air
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.0, 1.0, np.cos(np.clip((t - 19.0) / 1.0, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.02)
mix /= np.abs(mix).max()
drive = 1.3
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
