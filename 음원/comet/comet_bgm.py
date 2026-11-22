"""Original BGM for comet.mp4 (20 s), v6: a piano melody over an orchestral bed.

v5's lead (violins + clarinet in unison) read as an organ, so the tune is now a piano: single
notes in the verse; octaves plus a chord tone under the long notes in the highlight, over flowing
left-hand 8ths. The orchestra (strings, horns, choir, contrabass) sits underneath as a soft,
slow-attack bed, and only the cellos keep a gentle counter-line. The form is unchanged: the tune from
1.2 s, the build from 3.8 s, the highlight from fireball 1 (6.47 s) to the end, a lift from D to E
major at 11.76 s (now a harp glissando), and fireball 3 at 17.05 s. The fireball peaks land on beats
(90.75 BPM). Original music in the spirit of the hopeful, swelling climax of "Sparkle".
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261204)

BPM = 90.75
BEAT = 60.0 / BPM
BAR = 4 * BEAT
K0 = 6.47
FIREBALLS = [(5.63, 6.47), (11.43, 12.40), (16.33, 17.07)]


def tb(bar, pos=0.0):
    return K0 + bar * BAR + pos * BEAT


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
GAIN = dict(melody=0.60, lh=0.28, vln=0.08, cello=0.070, clarinet=0.18, horns=0.085, violas=0.050, contrabass=0.055,
            ebass=0.12, harp=0.22, piano=0.16, choir=0.040, timpani=0.11, kick=0.30, snare=0.46, cymbal=0.13,
            swell=0.07, celesta=0.15)
SEND = dict(melody=.26, lh=.22, vln=.40, cello=.40, clarinet=.35, horns=.45, violas=.50, contrabass=.20, ebass=.05,
            harp=.40, piano=.28, choir=.55, timpani=.30, kick=.08, snare=.30, cymbal=.30, swell=.45, celesta=.45)


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


# ---------------------------------------------------------------- instruments
def bowed(m, dur, vel=0.6, attack=0.12, voices=6, vib_c=13, bright=7000, spread=1.0):
    """A string section: detuned saw voices with vibrato, spread in stereo."""
    f0 = float(mtof(m)); T = dur + 0.9; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for v in range(voices):
        vib = vib_c * np.sin(2 * np.pi * rng.uniform(4.8, 6.0) * t + rng.uniform(0, 6)) * np.clip(t / 0.35, 0, 1) + rng.normal(0, 6)
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** (vib / 1200)) / SR
        s = sum(np.sin(k * ph) / k / np.sqrt(1 + (f0 * k / bright) ** 4) for k in range(1, 16) if f0 * k < 10000)
        w = 0.5 + spread * ((v + 0.5) / voices - 0.5)
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.3))
    return np.vstack([L * env, R * env]) * vel


def horns(notes, dur, vel=0.6, attack=0.25):
    """French horns: warm, round brass (saw stack through a soft filter that opens with the swell)."""
    T = dur + 0.8; t = tt(T)
    fc = 500 + 900 * np.clip(t / (attack * 2), 0, 1)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-5, .7), (5, .3)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = sum(np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6)) / k / np.sqrt(1 + (f * k / fc) ** 4) for k in range(1, 18) if f * k < 6000)
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.3))
    return np.vstack([np.tanh(0.8 * L), np.tanh(0.8 * R)]) * env * vel


def clarinet(m, dur, vel=0.5):
    """Warm reed: mostly odd harmonics, gentle vibrato, soft attack."""
    f = float(mtof(m)); T = dur + 0.4; t = tt(T)
    vib = 8 * np.sin(2 * np.pi * 5.0 * t) * np.clip((t - 0.25) / 0.3, 0, 1)
    ph = 2 * np.pi * np.cumsum(f * 2 ** (vib / 1200)) / SR
    tone = sum(a * np.sin(k * ph) for k, a in [(1, 1), (3, .45), (5, .22), (7, .1), (2, .06), (9, .04)] if f * k < 8000)
    env = np.sin(np.minimum(1, t / 0.05) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.08))
    return tone * env * vel


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


def harp(m, vel=0.4):
    f = float(mtof(m)); T = float(np.interp(m, [40, 60, 84], [4.0, 3.0, 1.6])); t = tt(T)
    out = np.zeros_like(t)
    for n in range(1, 11):
        if f * n > 9000:
            break
        out += abs(np.sin(n * np.pi * 0.3)) / n ** 1.6 * np.sin(2 * np.pi * f * n * t) * np.exp(-t * (0.8 + 0.6 * n) * (f / 440) ** 0.3)
    return out * np.minimum(1, t / 0.003) * vel


def ebass(m, dur, vel=0.7):
    f = float(mtof(m)); T = dur + 0.15; t = tt(T)
    out = sum((1.0 / k ** 1.4) * np.sin(2 * np.pi * f * k * t) * ((1.0 if k <= 2 else 0.3 / k) + (1 - (1.0 if k <= 2 else 0.3 / k)) * np.exp(-t * (5 + 4 * k))) for k in range(1, 10) if f * k < 3000)
    env = np.minimum(1, t / 0.006) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.05))
    return np.tanh(1.1 * out) * env * vel


VOWEL_AH = [(700, 110, 1.0), (1150, 130, 0.45), (2600, 200, 0.08)]


def choir(m, dur, attack=0.5):
    f0 = float(mtof(m)); T = dur + 1.5; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    form = lambda f: sum(a / (1 + ((f - F) / bw) ** 2) for F, bw, a in VOWEL_AH) + 0.01
    for v in range(4):
        vib = 12 * np.sin(2 * np.pi * rng.uniform(4.5, 5.3) * t + rng.uniform(0, 6)) + rng.normal(0, 6)
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** (vib / 1200)) / SR
        s = sum(form(f0 * k) / k ** 0.6 * np.sin(k * ph + rng.uniform(0, 6)) for k in range(1, 30) if f0 * k < 4500)
        w = (v + 0.5) / 4
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.6))
    return np.vstack([L * env, R * env])


def timpani(m, vel=0.8):
    f = float(mtof(m)); t = tt(2.5)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.04 * np.exp(-t / 0.05))) / SR
    body = np.sin(ph) * np.exp(-t / 0.9) + 0.4 * np.sin(1.5 * ph) * np.exp(-t / 0.4) + 0.2 * np.sin(1.99 * ph) * np.exp(-t / 0.3)
    nz = fft_filter(noise(len(t)), hi=700); nz /= nz.std()
    return (body + 0.25 * nz * np.exp(-t / 0.02)) * np.minimum(1, t / 0.002) * vel


def kick(vel=1.0):
    t = tt(0.45)
    ph = 2 * np.pi * np.cumsum(50 + 85 * np.exp(-t / 0.025)) / SR
    return np.tanh(1.1 * np.sin(ph) * np.exp(-t / 0.22)) * np.minimum(1, t / 0.002) * vel


def snare(vel=0.8):
    t = tt(0.5)
    tone = np.sin(2 * np.pi * np.cumsum(170 * (1 + 0.3 * np.exp(-t / 0.01))) / SR) * np.exp(-t / 0.09)
    nz = fft_filter(noise(len(t)), lo=900, hi=6000); nz /= nz.std()
    return (0.9 * tone + 0.45 * np.tanh(0.7 * nz) * np.exp(-t / 0.15)) * np.minimum(1, t / 0.001) * vel


def cymbal(vel=0.6, swell=0.0):
    """Suspended cymbal: a soft crash, or a mallet swell that crescendos into the hit."""
    T = 2.8 + swell; t = tt(T)
    nz = fft_filter(noise(len(t)), lo=3000, hi=11000); nz /= nz.std()
    metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in [2900, 4100, 5300]) / 3
    x = np.tanh(0.6 * nz) / 0.6 + 0.3 * metal
    if swell:
        env = np.where(t < swell, (t / swell) ** 2.5, np.exp(-(t - swell) / 0.9))
    else:
        env = np.exp(-t / 1.0) * np.minimum(1, t / 0.002)
    return x * env * vel


def celesta(m, vel=0.3):
    f = float(mtof(m)); t = tt(1.8)
    return (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.8) + 0.2 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.25)) * np.minimum(1, t / 0.002) * vel


# ---------------------------------------------------------------- harmony
VO = {   # (bass root, mid voicing for violas/horns/choir, harp tones under the pedal)
    'Dadd9': (38, [62, 66, 69, 76], [62, 66, 64]),
    'A/C#':  (37, [61, 64, 69, 73], [61, 64, 61]),
    'Bm7':   (35, [62, 66, 69, 74], [62, 66, 59]),
    'Gmaj7': (43, [62, 66, 71, 74], [59, 62, 66]),
    'D':     (38, [62, 66, 69, 74], [62, 66, 62]),
    'Gmaj9': (43, [62, 66, 69, 71], [59, 62, 66]),
    'Bsus4': (35, [59, 64, 66, 71], [64, 66, 59]),
    'B':     (35, [59, 63, 66, 71], [63, 66, 59]),
    # after the lift into E major
    'E':     (40, [59, 64, 68, 71], [64, 68, 64]),
    'B/D#':  (39, [59, 63, 66, 71], [63, 66, 63]),
    'C#m7':  (37, [59, 64, 68, 71], [61, 64, 68]),
    'A':     (33, [57, 61, 64, 69], [61, 64, 69]),
    'Eend':  (40, [59, 64, 68, 71, 76], [64, 68, 64]),
}
BARS = {-3: [(0, 'Dadd9'), (2, 'Dadd9')], -2: [(0, 'Dadd9'), (2, 'A/C#')], -1: [(0, 'Bm7'), (2, 'Gmaj7')],
        0: [(0, 'D'), (2, 'A/C#')], 1: [(0, 'Bm7'), (2, 'Gmaj9'), (3, 'Bsus4'), (3.5, 'B')],
        2: [(0, 'E'), (2, 'B/D#')], 3: [(0, 'C#m7'), (2, 'A'), (3, 'B')], 4: [(0, 'Eend')]}


def pedal(bar):
    return 69 if bar < 2 else 71          # A4 in D major, B4 after the lift to E


def chord_at(bar, pos):
    c = BARS[bar][0][1]
    for p, name in BARS[bar]:
        if pos >= p:
            c = name
    return c


def spans():
    for bar in range(-3, 5):
        for p, c in BARS[bar]:
            nxt = [pp for pp, _ in BARS[bar] if pp > p]
            span = (nxt[0] if nxt else 4) - p
            yield bar, p, c, span


def seg_dur(bar, span):
    return span * BEAT if bar < 4 else (DUR - tb(4)) + 0.3


LVL = {-3: 0.45, -2: 0.6, -1: 0.8, 0: 1.0, 1: 1.0, 2: 1.1, 3: 1.1, 4: 1.0}

# harp sparkle: 16ths through the build, 8ths under the highlight (pedal on top, chord tones below)
for bar in range(-3, 4):
    step = 0.25 if bar < 0 else 0.5
    for k in range(int(4 / step)):
        pos = k * step; t0 = tb(bar, pos)
        if t0 < 0.02:
            continue
        under = VO[chord_at(bar, pos)][2]
        m = pedal(bar) if k % 2 == 1 else under[(k // 2) % 3]
        place('harp', harp(m, (0.42 if k % 4 == 0 else 0.34) * (0.95 if bar < 0 else 0.8)), t0,
              pan=float(0.35 if k % 2 else -0.2))
for j, m in enumerate([83, 80, 76, 71, 68, 64, 59, 56]):           # closing cascade in E
    place('harp', harp(m, 0.40 - 0.03 * j), tb(4) + 0.35 + j * 0.12, pan=float(0.4 - 0.1 * j))

# strings on the chords from the very start (soft), contrabass + bass under the highlight
for bar, p, c, span in spans():
    root, v, _ = VO[c]
    dur = seg_dur(bar, span)
    t0 = max(tb(bar, p), 0.0)
    if tb(bar, p) + dur <= 0:
        continue
    for m in v[:4]:
        place('violas', bowed(m, dur + 0.03, 0.5, attack=1.0 if bar == -3 else 0.45, voices=4, bright=3200), t0, gain=LVL[bar])
    if bar >= -1:
        place('contrabass', bowed(root - 12 if root >= 40 else root, dur + 0.03, 0.8, attack=0.12, voices=3, vib_c=8,
                                  bright=1500, spread=0.3), t0, gain=LVL[bar])
    if bar >= 0:
        for k in range(max(1, int(span * 2))):
            if bar == 4:
                place('ebass', ebass(root, dur * 0.95, 0.75), t0); break
            place('ebass', ebass(root, BEAT * 0.45, 0.75 if k % 2 == 0 else 0.6), tb(bar, p + k * 0.5))

# horns: from the build on, carrying the chords; choir 'ah' under the whole highlight
for bar, p, c, span in spans():
    if bar < -1:
        continue
    root, v, _ = VO[c]
    dur = seg_dur(bar, span)
    place('horns', horns([n - 12 for n in v[:3]], dur + 0.03, 0.5 if bar < 0 else 0.7, attack=0.5), tb(bar, p), pan=-0.3)
    if bar >= 0:
        for m in v[:3]:
            place('choir', choir(m - 12, dur, attack=0.3), tb(bar, p), gain=LVL[bar])

# the tune: it starts at 1.2 s, builds from 3.8 s, and pours out from the first fireball to the end
MEL = {-2: [(0, 66, 1), (1, 69, 1), (2, 64, 1.5), (3.5, 66, .5)],
       -1: [(0, 69, 1), (1, 71, 1), (2, 74, 1), (3, 76, 1)],
       0: [(0, 78, 1.5), (1.5, 76, .5), (2, 76, .5), (2.5, 78, .5), (3, 81, 1)],
       1: [(0, 78, 1), (1, 78, .5), (1.5, 81, .5), (2, 83, 1), (3, 76, .5), (3.5, 75, .5)],
       2: [(0, 76, .5), (0.5, 78, .5), (1, 80, 2), (3, 78, .5), (3.5, 83, .5)],
       3: [(0, 80, 1), (1, 83, 1), (2, 85, 1), (3, 83, .5), (3.5, 78, .5)],
       4: [(0, 80, 4.2)]}
def under_tone(bar, pos, m):
    """The chord tone just below the melody note (for a fuller right hand on long notes)."""
    cand = [n + 12 * k for n in VO[chord_at(bar, pos)][1] for k in (-1, 0, 1)]
    below = sorted(n for n in cand if 2 < m - n <= 9)
    return below[-1] if below else None


for bar, ns in MEL.items():
    for p, m, d in ns:
        t0 = tb(bar, p); dur = d * BEAT * (0.97 if bar < 4 else 1.0)
        if bar < 0:
            place('melody', piano(m, dur, 0.52 if bar == -2 else 0.58), t0, pan=0.05)
            continue
        vel = 0.68 if (p % 1 == 0 or d >= 1) else 0.6
        place('melody', piano(m, dur, vel), t0, pan=0.05)
        place('melody', piano(m - 12, dur, vel * 0.62), t0, pan=-0.05)
        if d >= 1:
            u = under_tone(bar, p, m)
            if u:
                place('melody', piano(u, dur, vel * 0.42), t0, pan=0.0)

# piano left hand under the highlight: flowing 8th-note broken chords (root, fifth, octave, tenth)
for bar, p, c, span in spans():
    if bar < 0:
        continue
    root, v, _ = VO[c]
    r = root if root >= 36 else root + 12
    figure = [r, r + 7, r + 12, r + 16 if c in ('D', 'E', 'A', 'B', 'Eend', 'Gmaj9', 'Gmaj7') else r + 15, r + 12, r + 7]
    steps = int(span * 2) if bar < 4 else 6
    for k in range(steps):
        place('lh', piano(figure[k % len(figure)], BEAT * 0.9, 0.42 if k % 2 == 0 else 0.34), tb(bar, p + k * 0.5), pan=-0.2)

# counter-melody on cellos + horns, moving against the tune through the highlight
COUNTER = [(0, 0, 62, 2), (0, 2, 61, 2), (1, 0, 59, 2), (1, 2, 59, 1), (1, 3, 64, .5), (1, 3.5, 63, .5),
           (2, 0, 64, 2), (2, 2, 63, 2), (3, 0, 61, 2), (3, 2, 64, 1), (3, 3, 63, 1), (4, 0, 64, 4.2)]
for bar, p, m, d in COUNTER:
    dur = d * BEAT * 0.98
    place('cello', bowed(m, dur, 0.45, attack=0.3, voices=4, bright=3000), tb(bar, p), pan=-0.3)

# fireballs: timpani roll + mallet-cymbal swell + low-string swell through the brightening, then the hit
for t_start, t_peak in FIREBALLS:
    T = t_peak - t_start
    nroll = int(T / 0.055)
    for k in range(nroll):
        place('timpani', timpani(38, 0.06 + 0.30 * (k / nroll) ** 1.8), t_start + k * 0.055, pan=-0.1)
    tp = 40 if t_peak > tb(2) - 0.01 else 38
    place('timpani', timpani(tp, 1.0), t_peak, pan=-0.1)
    place('timpani', timpani(tp - 5, 0.7), t_peak + 0.01, pan=0.1)
    place('cymbal', cymbal(0.7, swell=T), t_start, pan=float(rng.uniform(-.3, .3)))
    bar_pk = int(np.floor((t_peak - K0 + 0.01) / BAR)); pos_pk = ((t_peak - K0 + 0.01) % BAR) / BEAT
    r = VO[chord_at(bar_pk, pos_pk)][0] % 12 + 48
    place('swell', bowed(r, T + 0.05, 0.6, attack=T, voices=4, bright=4000), t_start)
    place('swell', bowed(r + 7, T + 0.05, 0.5, attack=T, voices=4, bright=4000), t_start)
# the lift into E major: a quick string run-up and timpani on G -> E
for j, m in enumerate([59, 62, 64, 66, 68, 69, 71, 73, 76]):
    place('harp', harp(m, 0.30 + 0.02 * j), tb(1, 3.5) + j * BEAT / 18, pan=float(-0.5 + 0.12 * j))

# drums: light from the build, full under the highlight
for bar in range(-1, 4):
    if bar == -1:
        place('kick', kick(0.7), tb(bar, 0)); place('snare', snare(0.5), tb(bar, 2), pan=-0.05)
        for k in range(8):
            place('snare', snare(0.18 + 0.05 * k), tb(bar, 2 + k * 0.25), pan=-0.05)
        continue
    for pos, v in [(0, 1.0), (1.5, .6), (2, .8), (2.5, .6)]:
        place('kick', kick(v), tb(bar, pos))
    for pos in (1, 3):
        place('snare', snare(0.8), tb(bar, pos), pan=-0.05)
for k in range(8):
    place('timpani', timpani(45 if k % 2 == 0 else 40, 0.4 + 0.06 * k), tb(3, 2 + k * 0.25), pan=float(0.3 - 0.08 * k))

# meteors: a few low celesta glints
for bar in range(-3, 4):
    GL = [74, 76, 78, 81, 83, 85, 86] if bar < 2 else [76, 78, 80, 83, 85, 87, 88]
    dens = np.interp(bar, [-3, 0, 3], [0.04, 0.08, 0.10])
    for e in range(8):
        t0 = tb(bar, e / 2)
        if t0 > 0.1 and rng.random() < dens:
            place('celesta', celesta(int(rng.choice(GL)), rng.uniform(0.2, 0.3)), t0, pan=float(rng.uniform(-.6, .6)))


# ---------------------------------------------------------------- mix
def reverb(x2, T=3.0):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.4)
        ir = (fft_filter(nz, hi=7000) * w + fft_filter(nz, hi=2500) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.025 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


for k, lo, hi in [('vln', 200, 7000), ('cello', 60, 4000), ('violas', 150, 5000), ('contrabass', 30, 1200),
                  ('horns', 70, 3000), ('choir', 120, 3500), ('swell', 90, 4000)]:
    if k in STEMS:
        STEMS[k] = fft_filter(STEMS[k].astype(float), lo=lo, hi=hi).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, tb(-2)), verse=(tb(-2), tb(0)), ch1=(tb(0), tb(2)), ch2=(tb(2), tb(4)), end=(tb(4), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:10s}                {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.30 * reverb(snd)
mix = fft_filter(mix, lo=28)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.0, 1.0, np.cos(np.clip((t - 19.0) / 1.0, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.01)
mix /= np.abs(mix).max()
drive = 1.4
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
