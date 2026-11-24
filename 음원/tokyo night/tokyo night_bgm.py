"""Original BGM for tokyo night.mp4 (20 s): Tokyo across the bay at dusk - a Diamond Fuji sunset, then the
city switching on: neon signs, the Rainbow Bridge, Tokyo Tower, the Skytree, and the bridge going rainbow.

Japanese city pop at 120 BPM in E-flat major (maj7/9 chords, the IVmaj7 - III7 - vim7 colour), moving up a
semitone to E major for the rainbow finale. A real-band sound: a warm, lightly driven electric guitar
lead (plucked-string model, oversampled amp, chorus and delay), 16th-note guitar cutting, Rhodes, piano,
a fingered bass and a 16-beat drum groove. No synth leads.

Sync (the event times are the constants in tokyo.py):
  0.0 s  Diamond Fuji: a shimmering piano/Rhodes chord as the sun sits on the summit (dusk intro)
  3.2 / 5.6 / 7.2 / 10.4 s  neon signs strike: a Rhodes note that stutters with the tube, then holds
  4.0 s  Rainbow Bridge lights sweep left -> right: the groove starts (bar 2) with a piano run panned L -> R
  8.5 s  Tokyo Tower lights from the base up: the lead guitar enters with a rising run
 12.0 s  Skytree lights: the chorus (bar 6)
 15.5 s  the bridge turns rainbow: a pick-up into a semitone lift (Eb -> E) at 16.0 s for the finale
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261130)

BEAT = 0.5                                  # 120 BPM
BAR = 4 * BEAT
FUJI, BRIDGE, TOWER, SKYTREE, RAINBOW, LIFT = 0.0, 4.0, 8.5, 12.0, 15.5, 16.0
NEON = [3.2, 5.6, 7.2, 10.4]
SW16 = 0.02                                 # a little 16th swing


def tb(bar, pos=0.0):
    frac = pos % 0.5
    if abs(frac - 0.25) < 1e-6:
        pos += SW16
    return bar * BAR + pos * BEAT


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


def peak_eq(x, f0, gain_db, q=1.0):
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = 10 ** (gain_db / 20) - 1
    H = 1 + g / (1 + (q * (f / f0 - f0 / np.maximum(f, 1e-3))) ** 2)
    return np.fft.irfft(X * H, n, axis=-1)


def noise(n):
    return rng.standard_normal(n)


STEMS = {}
GAIN = dict(lead=0.42, cut=0.32, rhodes=0.15, piano=0.22, neon=0.16, bass=0.16, strings=0.022,
            kick=0.30, snare=0.48, hat=0.19, ohat=0.10, crash=0.08, tom=0.18)
SEND = dict(lead=.25, cut=.10, rhodes=.25, piano=.30, neon=.40, bass=.03, strings=.45,
            kick=.04, snare=.16, hat=.06, ohat=.10, crash=.20, tom=.12)


def place(name, sig, t, pan=0.0, gain=1.0):
    buf = STEMS.setdefault(name, np.zeros((2, N), np.float32))
    i = int(round(t * SR))
    if sig.ndim == 1:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * np.sqrt(2)
    else:
        st = sig.copy()
    k = min(st.shape[1], max(int(0.004 * SR), int(0.03 * st.shape[1])))
    st[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    if i < 0:
        st, i = st[:, -i:], 0
    n = min(st.shape[1], N - i)
    if n > 0:
        buf[:, i:i + n] += (st[:, :n] * gain).astype(np.float32)


# ---------------------------------------------------------------- guitars: plucked strings + amp
def ks_string(m, dur, vel=0.7, bright=0.6, pos=0.16, T60=None, damp=0.025):
    """Karplus-Strong plucked string, damped after `dur`."""
    f = float(mtof(m)); P = SR / f
    P0 = int(P - 0.5); w = P - 0.5 - P0
    n = int((dur + 0.1) * SR)
    L = P0 + 2
    exc = fft_filter(noise(L * 2), hi=1000 + 6000 * bright * vel, order=1)[:L]
    d = max(1, int(pos * P))
    exc = exc - np.concatenate([np.zeros(d), exc[:-d]])
    exc /= np.abs(exc).max() + 1e-9
    x = np.zeros(n); x[:L] = exc * vel
    if T60 is None:
        T60 = float(np.interp(m, [40, 52, 64, 76, 88], [5.0, 4.0, 3.0, 2.2, 1.5]))
    a, b, c = 0.5 * (1 - w), 0.5, 0.5 * w
    om = 2 * np.pi * f / SR
    loss = abs(a + b * np.exp(-1j * om) + c * np.exp(-2j * om))
    g = min(0.9999, 10 ** (-3 * P / (T60 * SR)) / loss)
    y = np.zeros(n); y[:L] = x[:L]; s = L
    while s < n:
        e = min(n, s + P0)
        y[s:e] = x[s:e] + g * (a * y[s - P0:e - P0] + b * y[s - P0 - 1:e - P0 - 1] + c * y[s - P0 - 2:e - P0 - 2])
        s = e
    t = np.arange(n) / SR
    return y * np.where(t < dur, 1.0, np.exp(-(t - dur) / damp))


def bend(x, cents):
    rate = 2 ** (cents / 1200)
    pos = np.cumsum(rate) - rate[0]
    return np.interp(pos, np.arange(len(x)), x, right=0.0)


def oversampled(x, fn, factor=4):
    n = len(x)
    X = np.fft.rfft(x)
    up = np.zeros(n * factor // 2 + 1, complex); up[:len(X)] = X
    y = fn(np.fft.irfft(up, n * factor) * factor)
    return np.fft.irfft(np.fft.rfft(y)[:n // 2 + 1], n) / factor


def amp_cab(x, drive=2.5, mid=800, presence=2400, top=4800, pickup_hi=3000):
    """Warm two-stage amp (oversampled, light drive) into a speaker cabinet."""
    x = fft_filter(x, hi=pickup_hi, order=1)                 # neck pickup: warm
    x = fft_filter(x, lo=100, order=1)
    x = peak_eq(x, mid, 3.0, q=0.8)
    x = x / (np.abs(x).max() + 1e-9)
    y = oversampled(x, lambda u: np.tanh(drive * u + 0.15) - np.tanh(0.15))
    y = y / (np.abs(y).max() + 1e-9)
    y = oversampled(y, lambda u: np.tanh(1.4 * u))
    y = fft_filter(y, lo=80, order=2)
    y = peak_eq(y, 120, 2.0, q=1.8)
    y = peak_eq(y, presence, 2.5, q=1.2)
    y = fft_filter(y, hi=top, order=4)
    y = peak_eq(y, 7000, -8.0, q=0.8)
    return y


# ---------------------------------------------------------------- keys, bass, drums
def rhodes(m, dur, vel=0.45):
    f = float(mtof(m)); T = dur + 1.2; t = tt(T)
    idx = (0.5 + 1.3 * vel) * np.exp(-t / 0.22) + 0.18
    car = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
    tine = 0.08 * vel * np.sin(2 * np.pi * f * 7 * t) * np.exp(-t / 0.03)
    decay = float(np.interp(m, [48, 72, 96], [3.4, 2.2, 1.1]))
    env = np.minimum(1, t / 0.004) * np.exp(-t / decay) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.15))
    return (car + tine) * env * vel


def piano(m, dur, vel=0.5):
    f = float(mtof(m)); T = dur + 1.0; t = tt(T)
    t60 = float(np.interp(m, [36, 60, 84, 100], [7.5, 4.3, 1.9, 0.9]))
    out = np.zeros_like(t)
    for n in range(1, 13):
        fn = f * n * np.sqrt(1 + 0.0004 * n * n)
        if fn > 10000:
            break
        a = (1.0 / n ** 1.2) * ((0.25 + 0.5 * vel) ** (n - 1))
        tau = t60 / 6.9 / (1 + 0.45 * (n - 1))
        out += a * np.sin(2 * np.pi * fn * t + rng.uniform(0, 0.2)) * (0.6 * np.exp(-t / (tau * 0.35)) + 0.4 * np.exp(-t / tau))
    out *= np.sin(np.minimum(1, t / 0.003) * np.pi / 2) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.15))
    return out * vel


def ebass(m, dur, vel=0.7):
    """Fingered electric bass: round, punchy."""
    f = float(mtof(m)); T = dur + 0.12; t = tt(T)
    out = sum((1.0 / k ** 1.3) * np.sin(2 * np.pi * f * k * t) * ((1.0 if k <= 2 else 0.3 / k) + (1 - (1.0 if k <= 2 else 0.3 / k)) * np.exp(-t * (6 + 5 * k))) for k in range(1, 12) if f * k < 3500)
    env = np.sin(np.minimum(1, t / 0.004) * np.pi / 2) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.03))
    return np.tanh(1.3 * out) * env * vel


def bowed(m, dur, vel=0.5, attack=0.5, voices=5, bright=2600):
    f0 = float(mtof(m)); T = dur + 1.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for v in range(voices):
        vibr = 11 * np.sin(2 * np.pi * rng.uniform(4.8, 5.8) * t + rng.uniform(0, 6)) * np.clip(t / 0.4, 0, 1) + rng.normal(0, 6)
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** (vibr / 1200)) / SR
        s = sum(np.sin(k * ph) / k / np.sqrt(1 + (f0 * k / bright) ** 4) for k in range(1, 12) if f0 * k < 7000)
        w = (v + 0.5) / voices
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.4))
    return np.vstack([L * env, R * env]) * vel


def kick(vel=0.9):
    t = tt(0.4)
    ph = 2 * np.pi * np.cumsum(55 + 80 * np.exp(-t / 0.02)) / SR
    boom = np.sin(ph) * np.exp(-t / 0.15)
    click = fft_filter(noise(len(t)), lo=1800, hi=6000) * np.exp(-t / 0.0025)
    return np.tanh(1.3 * (boom + 0.2 * click / (np.abs(click).max() + 1e-9))) * np.minimum(1, t / 0.0012) * vel


def snare(vel=0.8):
    t = tt(0.4)
    body = np.sin(2 * np.pi * np.cumsum(205 * (1 + 0.3 * np.exp(-t / 0.012))) / SR) * np.exp(-t / 0.06)
    wires = fft_filter(noise(len(t)), lo=2000, hi=10000); wires /= wires.std()
    return np.tanh(1.2 * (0.7 * body + 0.55 * np.tanh(0.6 * wires) * np.exp(-t / 0.12))) * np.minimum(1, t / 0.0008) * vel


def hat(vel=0.3, open_=False):
    T = 0.3 if open_ else 0.06; t = tt(T)
    nz = fft_filter(noise(len(t)), lo=6500, hi=13000); nz /= nz.std()
    metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in (7100, 8300, 9600)) / 3
    return (nz + 0.4 * metal) * np.exp(-t / (0.10 if open_ else 0.015)) * np.minimum(1, t / 0.0008) * vel


def crash(vel=0.6):
    t = tt(2.6)
    nz = fft_filter(noise(len(t)), lo=3000, hi=12000); nz /= nz.std()
    metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in [3100, 4500, 6200, 7700]) / 4
    return (np.tanh(0.6 * nz) / 0.6 + 0.3 * metal) * np.exp(-t / 1.0) * np.minimum(1, t / 0.002) * vel


def tom(m, vel=0.7):
    f = float(mtof(m)); t = tt(0.5)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.2 * np.exp(-t / 0.03))) / SR
    return np.tanh(1.1 * (np.sin(ph) * np.exp(-t / 0.22) + 0.3 * np.sin(1.6 * ph) * np.exp(-t / 0.07))) * np.minimum(1, t / 0.0015) * vel


# ---------------------------------------------------------------- harmony
CH = {   # bass root, Rhodes voicing (rootless), guitar cutting triad (upper), strings
    'Abmaj9':  (44, [60, 63, 67, 70], [67, 72, 75], [63, 67, 72]),
    'Bb13sus': (46, [63, 67, 68, 72], [68, 72, 75], [63, 68, 72]),
    'Ebmaj9':  (39, [62, 65, 67, 70], [67, 70, 74], [62, 67, 70]),
    'Cm9':     (36, [58, 62, 63, 67], [63, 67, 70], [63, 67, 70]),
    'Fm9':     (41, [56, 60, 63, 67], [63, 68, 72], [60, 63, 68]),
    'Bb13':    (46, [62, 67, 68, 72], [68, 72, 74], [62, 68, 72]),
    'G7#5':    (43, [59, 63, 65, 71], [65, 71, 75], [59, 65, 71]),
    'Bbm7':    (46, [61, 65, 68, 72], [65, 68, 73], [61, 65, 68]),
    'Eb7':     (39, [61, 65, 67, 72], [67, 73, 77], [61, 67, 73]),
    'Bb/Ab':   (44, [62, 65, 70, 74], [65, 70, 74], [62, 65, 70]),
    'Gm7':     (43, [58, 62, 65, 69], [65, 70, 74], [62, 65, 70]),
    'C7':      (36, [58, 64, 67, 69], [64, 70, 72], [58, 64, 69]),
    'Bb7b9':   (46, [59, 62, 68, 71], [62, 68, 71], [62, 68, 71]),
    'Amaj9':   (45, [61, 64, 68, 71], [68, 73, 76], [64, 68, 73]),
    'B/A':     (45, [63, 66, 71, 75], [66, 71, 75], [63, 66, 71]),
    'Emaj9':   (40, [63, 66, 68, 71], [68, 71, 75], [63, 68, 71]),
}
PROG = [(0, 0, 'Abmaj9'), (1, 0, 'Bb13sus'),
        (2, 0, 'Ebmaj9'), (2, 2, 'Cm9'), (3, 0, 'Fm9'), (3, 2, 'Bb13'),
        (4, 0, 'Abmaj9'), (4, 2, 'G7#5'), (5, 0, 'Cm9'), (5, 2, 'Bbm7'), (5, 3, 'Eb7'),
        (6, 0, 'Abmaj9'), (6, 2, 'Bb/Ab'), (7, 0, 'Gm7'), (7, 1, 'C7'), (7, 2, 'Fm9'), (7, 3, 'Bb7b9'),
        (8, 0, 'Amaj9'), (8, 2, 'B/A'), (9, 0, 'Emaj9')]


def chord_at(bar, pos):
    c = PROG[0][2]
    for b, p, name in PROG:
        if (bar, pos) >= (b, p):
            c = name
    return c


def segs():
    for i, (bar, pos, c) in enumerate(PROG):
        nb, npos = (PROG[i + 1][0], PROG[i + 1][1]) if i + 1 < len(PROG) else (10, 0)
        yield bar, pos, c, (nb - bar) * 4 + (npos - pos)


# ---------------------------------------------------------------- the lead guitar (bar, beat, beats, midi)
G4, Ab4, Bb4, B4, C5, Db5, D5, Eb5, E5, F5 = 67, 68, 70, 71, 72, 73, 74, 75, 76, 77
Eb4, F4, Bb3, C4, G3 = 63, 65, 58, 60, 55
LEAD = [
    # a short hook in the groove
    (3, 2, .5, G4), (3, 2.5, .5, Bb4), (3, 3, .5, C5), (3, 3.5, .5, D5),
    # Tokyo Tower lights from the base up: a rising run, then the tune
    (4, 1, .25, Eb4), (4, 1.25, .25, G4), (4, 1.5, .25, Bb4), (4, 1.75, .25, C5),
    (4, 2, 1, Eb5), (4, 3, .5, B4), (4, 3.5, .5, C5),
    (5, 0, 1.5, D5), (5, 1.5, .5, C5), (5, 2, .5, Db5), (5, 2.5, .5, C5), (5, 3, .5, Bb4), (5, 3.5, .5, G4),
    # the chorus (Skytree)
    (6, 0, .5, G4), (6, .5, 1, C5), (6, 1.5, .5, Bb4), (6, 2, 1.5, D5), (6, 3.5, .5, C5),
    (7, 0, .5, Bb4), (7, .5, .5, D5), (7, 1, .5, E5), (7, 1.5, .5, D5), (7, 2, 1, C5), (7, 3, .5, D5), (7, 3.5, .5, F5),
    # rainbow: up a semitone to E major
    (8, 0, 1.5, 73), (8, 1.5, .5, 71), (8, 2, 1.5, 75), (8, 3.5, .5, 73),
    (9, 0, 3.5, 71),
]
OCT = -12                    # the guitar plays it an octave below the written line (warm, not too high)
for i, (bar, p, d, m) in enumerate(LEAD):
    on = tb(bar, p)
    nxt = tb(LEAD[i + 1][0], LEAD[i + 1][1]) if i + 1 < len(LEAD) else on + d * BEAT
    dur = min(d * BEAT, max(nxt - on, 0.05)) + 0.01
    v = 0.85 if p % 1 == 0 else 0.75
    x = ks_string(m + OCT, dur, v, bright=0.6, pos=0.2, T60=6.0 if d >= 1 else 4.0)
    tl = np.arange(len(x)) / SR
    x *= np.minimum(np.exp(tl / 1.0), 3.0)                       # sustain
    cents = 22 * np.clip((tl - 0.22) / 0.3, 0, 1) * np.sin(2 * np.pi * 5.4 * tl) if d >= 1 else np.zeros(len(x))
    place('lead', bend(x, cents), on)
    if bar >= 8:                                                  # rainbow finale: a twin guitar a third above
        h = {73: 76, 71: 75, 75: 78}[m]
        xh = ks_string(h + OCT, dur, v * 0.8, bright=0.6, pos=0.22, T60=6.0 if d >= 1 else 4.0)
        xh *= np.minimum(np.exp(np.arange(len(xh)) / SR / 1.0), 3.0)
        place('lead', bend(xh, cents[:len(xh)] * 0.8), on + 0.006)

# ---------------------------------------------------------------- the band
# Diamond Fuji: a shimmering chord as the sun sits on the summit
for j, m in enumerate([56, 63, 67, 70, 72, 75, 79, 84]):
    place('piano', piano(m, 3.0, 0.42 - 0.02 * j), FUJI + 0.05 + j * 0.07, pan=float(-0.5 + 0.14 * j))

# Rhodes: long chords in the intro, comping (on the beat and the "and of 2") in the groove
for bar, pos, c, span in segs():
    v = CH[c][1]
    if bar <= 1:
        for j, m in enumerate(v):
            place('rhodes', rhodes(m, span * BEAT * 0.98, 0.40), tb(bar, pos) + 0.02 * j, pan=-0.25)
        continue
    if bar == 9:
        for j, m in enumerate(v + [v[0] + 12]):
            place('rhodes', rhodes(m, 3.0, 0.42), tb(9, 0) + 0.025 * j, pan=-0.2)
        continue
    for hp, d, vel in ((0, 0.9, 0.42), (1.5, 0.4, 0.32)):
        if hp < span:
            for m in v:
                place('rhodes', rhodes(m, d * BEAT, vel), tb(bar, pos + hp) + 0.01, pan=-0.3)

# the bridge lights sweep left to right: a piano run panned with them
run = [63, 67, 70, 74, 75, 79, 82, 86, 87]
for j, m in enumerate(run):
    place('piano', piano(m, 0.8, 0.40), BRIDGE + j * 1.5 / len(run), pan=float(-0.9 + 1.8 * j / (len(run) - 1)))

# neon signs strike: a Rhodes note that stutters like the tube (18 flickers a second), then holds
for t0, m in zip(NEON, [79, 82, 77, 84]):
    gate = (rng.random(13) > 0.45).astype(float) * 0.8
    for k, g in enumerate(gate):
        if g:
            place('neon', rhodes(m, 1 / 18 * 0.9, 0.45 * g), t0 + k / 18, pan=0.55)
    place('neon', rhodes(m, 0.6, 0.38), t0 + 0.7, pan=0.55)

# guitar cutting: muted-ish 16th chops on the off-beats from the groove on
cut_pattern = [(0.5, .8), (0.75, .45), (1.5, .8), (2.5, .8), (2.75, .45), (3.5, .8)]
for bar in range(2, 9):
    for q, v in cut_pattern:
        c = chord_at(bar, q)
        for j, m in enumerate(CH[c][2]):
            place('cut', ks_string(m - 12, 0.09, v, bright=0.7, pos=0.15, damp=0.012), tb(bar, q) + j * 0.004 + rng.normal(0, 0.002))

# bass: whole notes in the intro, a syncopated 16-beat line from the groove
BASS_PAT = [(0, 0, .75, .9), (0.75, 12, .25, .6), (1.5, 0, .5, .75), (2, 0, .75, .9), (2.75, 7, .25, .6), (3.5, 12, .25, .65), (3.75, 10, .25, .6)]
for bar, pos, c, span in segs():
    r = CH[c][0]
    if bar <= 1:
        place('bass', ebass(r - 12 if r > 40 else r, span * BEAT * 0.98, 0.5), tb(bar, pos)); continue
    if bar == 9:
        place('bass', ebass(r, 2.5, 0.85), tb(9, 0)); continue
    for q, off, d, v in BASS_PAT:
        if pos <= q < pos + span:
            m = r + off - (12 if r > 43 else 0)
            place('bass', ebass(m, d * BEAT * 0.95, v), tb(bar, q))

# strings: a quiet pad in the chorus and the finale
for bar, pos, c, span in segs():
    if bar >= 6:
        d = span * BEAT + 0.05 if bar < 9 else 3.0
        for m in CH[c][3]:
            place('strings', bowed(m, d, 0.5, attack=0.4), tb(bar, pos), gain=1.7 if bar >= 8 else 1.0)

# drums: a fill into the groove, then a 16-beat city-pop groove
for j, (q, m) in enumerate([(3.0, 52), (3.25, 50), (3.5, 47), (3.75, 43)]):
    place('tom', tom(m, 0.55 + 0.08 * j), tb(1, q), pan=float(0.3 - 0.2 * j))
for bar in range(2, 9):
    for q, v in ((0, 1.0), (1.75, 0.55), (2, 0.9), (2.5, 0.6)):
        place('kick', kick(v), tb(bar, q))
    for q in (1, 3):
        place('snare', snare(0.85), tb(bar, q) + 0.004, pan=-0.05)
    for k in range(16):
        q = k * 0.25
        if q == 3.5:
            place('ohat', hat(0.6, open_=True), tb(bar, q), pan=0.3)
            continue
        place('hat', hat(0.42 if k % 4 == 0 else (0.3 if k % 2 == 0 else 0.2)), tb(bar, q), pan=0.3)
# the tower: an accent as it lights; the rainbow: a snare pick-up into the lift
place('crash', crash(0.6), TOWER, pan=-0.3)
for k, q in enumerate(np.arange(3.0, 4.0, 0.25)):
    place('snare', snare(0.45 + 0.12 * k), tb(7, q) + 0.004, pan=-0.05)
for t0, v in ((BRIDGE, 0.8), (SKYTREE, 1.0), (LIFT, 1.0), (tb(9), 0.8)):
    place('crash', crash(v), t0, pan=0.3)
place('kick', kick(1.0), tb(9)); place('snare', snare(0.8), tb(9) + 0.004)


# ---------------------------------------------------------------- amps, effects and mix
def reverb(x2, T=2.2):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.35)
        ir = (fft_filter(nz, hi=8000) * w + fft_filter(nz, hi=3500) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.02 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


def chorus_fx(x2, depth_ms=2.2, rate=0.7):
    n = x2.shape[1]; idx = np.arange(n, dtype=float)
    out = np.empty_like(x2)
    for c, ph in ((0, 0.0), (1, np.pi / 2)):
        d = (7 + depth_ms * (1 + np.sin(2 * np.pi * rate * idx / SR + ph)) / 2) / 1000 * SR
        out[c] = 0.7 * x2[c] + 0.5 * np.interp(idx - d, idx, x2[c])
    return out


def to_stereo(mono, pan):
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.vstack([mono * np.cos(th), mono * np.sin(th)]) * np.sqrt(2)


lead = amp_cab(STEMS['lead'].astype(float).mean(0), drive=2.6, mid=850, presence=2300, top=4800, pickup_hi=3000)
_s = 3.5 * np.sqrt((lead[lead != 0] ** 2).mean()); lead = _s * np.tanh(lead / _s)
dl = np.zeros_like(lead)
for dly, g, hi in ((0.75 * BEAT, 0.26, 3200), (1.5 * BEAT, 0.12, 2500)):
    i = int(dly * SR); dl[i:] += g * fft_filter(lead, hi=hi)[:-i]
STEMS['lead'] = chorus_fx(to_stereo(lead, 0.05) + to_stereo(dl, -0.55) * 0.6 + to_stereo(dl, 0.55) * 0.4, depth_ms=1.5).astype(np.float32)
cut = amp_cab(STEMS['cut'].astype(float).mean(0), drive=1.3, mid=1200, presence=3000, top=6000, pickup_hi=5000)
STEMS['cut'] = chorus_fx(to_stereo(cut, 0.55)).astype(np.float32)
STEMS['rhodes'] = chorus_fx(fft_filter(STEMS['rhodes'].astype(float), lo=120, hi=5000), depth_ms=2.5, rate=0.9).astype(np.float32)
STEMS['bass'] = peak_eq(STEMS['bass'].astype(float), 800, 3.0, q=0.9).astype(np.float32)
STEMS['strings'] = fft_filter(STEMS['strings'].astype(float), lo=200, hi=4000).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(dusk=(0, BRIDGE), groove=(BRIDGE, TOWER), tower=(TOWER, SKYTREE), chorus=(SKYTREE, LIFT), rainbow=(LIFT, 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:10s}                {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.22 * reverb(snd)
mix = fft_filter(mix, lo=32)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.0, 1.0, np.cos(np.clip((t - 19.0) / 1.0, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.01)
mix /= np.abs(mix).max()
drive = 1.3
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
