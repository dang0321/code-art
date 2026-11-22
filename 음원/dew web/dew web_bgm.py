"""Original BGM for dew web.mp4 (20 s): dew drops on a spider web in the early light.

A bright, bouncy tune on a high marimba in F major (written in G, transposed down a tone) (maj7 / #11 colours, no pentatonic licks).
Rolling three-note "또르르" figures stand for drops running along the silk; staccato 8ths and
16ths, off-beat marimba dyads and a pizzicato bass keep it bouncing.

Sync (measured from the video): the big drop shows up at ~7.2 s (glock sparkle), starts to slide
down its strand at ~10.8 s (bar 5: the tune turns into descending rolling figures), stretches from
14.9 s (marimba tremolo crescendo, drums out) and lets go at 15.62 s (bar 7 downbeat): a fast
marimba run falls with it, a water "plip" marks the release and a "plop" when it leaves the frame
(16.42 s). The web's recoil (a ~3.1 Hz damped vibration) is a vibraphone chord whose tremolo runs
at the same rate and dies away at the same speed. 100 BPM, so the slide and the release land on
bar downbeats.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261120)

RELEASE = 15.62            # the drop lets go of the strand
EXIT = 16.42               # the falling drop leaves the frame
RECOIL_HZ = 19.7 / (2 * np.pi)   # the web's damped vibration (3.14 Hz), decaying as exp(-2.2 t)
BEAT = 0.6                 # 100 BPM
BAR = 4 * BEAT
B1 = RELEASE - 6 * BAR     # bar 1 = 1.22 s; bar 5 (slide) = 10.82 s; bar 7 (release) = 15.62 s
SW16 = 0.035               # a light 16th-note swing for bounce
TRANSPOSE = -2             # written in G, sounds in F: high, but not piercing


def tb(bar, pos=0.0):
    if abs(pos % 0.5 - 0.25) < 1e-6:
        pos += SW16
    return B1 + (bar - 1) * BAR + pos * BEAT


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) + TRANSPOSE - 69.0) / 12.0)


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
GAIN = dict(lead=0.50, harmony=0.26, comp=0.20, roll=0.36, run=0.40, bass=0.36, pad=0.022, glock=0.16,
            vibes=0.13, drop=0.20, kick=0.26, snap=0.15, wood=0.20, tick=0.16)
SEND = dict(lead=.22, harmony=.25, comp=.20, roll=.28, run=.30, bass=.06, pad=.45, glock=.45, vibes=.40,
            drop=.35, kick=.04, snap=.18, wood=.20, tick=.10)


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
def marimba(m, vel=0.6, hard=0.6):
    """Rosewood bar over a resonator: tuned partials 1 : 3.98 : 9.85, short high notes, a woody click."""
    f = float(mtof(m))
    tau = float(np.interp(m, [43, 60, 72, 84, 96], [1.2, 0.75, 0.46, 0.28, 0.17]))
    T = tau * 6 + 0.05; t = tt(T)
    out = np.sin(2 * np.pi * f * t) * (0.7 * np.exp(-t / tau) + 0.3 * np.exp(-t / (tau * 0.25)))
    for r, a, k in [(3.98, 0.20 + 0.32 * vel * hard, 0.16), (9.85, 0.04 + 0.12 * vel * hard, 0.05)]:
        if f * r < 15000:
            out += a * np.sin(2 * np.pi * f * r * t + rng.uniform(0, 6)) * np.exp(-t / (tau * k))
    click = fft_filter(noise(int(0.012 * SR)), lo=1800, hi=7000)
    click /= click.std() + 1e-9
    out[:len(click)] += 0.06 * vel * hard * click * np.exp(-tt(0.012) / 0.0015)
    att = 0.0008 + 0.002 * (1 - hard)
    return out * np.sin(np.minimum(1, t / att) * np.pi / 2) * vel


def glock(m, vel=0.3):
    f = float(mtof(m)); t = tt(2.4)
    out = np.sin(2 * np.pi * f * t) * np.exp(-t / 1.0)
    for r, a, k in [(2.76, 0.22, 0.22), (5.40, 0.07, 0.07)]:
        if f * r < 16000:
            out += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / k)
    return out * np.minimum(1, t / 0.001) * vel


def vibe_note(m, T, vel, trem):
    f = float(mtof(m)); t = tt(T)
    tau = float(np.interp(m, [55, 79, 91], [2.6, 1.8, 1.1]))
    out = (np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * np.exp(-t / tau)
           + 0.10 * np.sin(2 * np.pi * 4.0 * f * t) * np.exp(-t / (tau * 0.12)))
    return out * trem(t) * np.minimum(1, t / 0.002) * vel


def pizz(m, vel=0.7):
    """Pizzicato double bass: plucked partials, fast decay, round body."""
    f = float(mtof(m)); t = tt(1.4)
    out = np.zeros_like(t)
    for n in range(1, 11):
        if f * n > 3500:
            break
        out += (abs(np.sin(n * np.pi * 0.22)) / n ** 1.2) * np.sin(2 * np.pi * f * n * t) * np.exp(-t * (2.4 + 1.8 * n))
    return out * np.sin(np.minimum(1, t / 0.004) * np.pi / 2) * vel


def pad(notes, dur, attack=0.8):
    T = dur + 2.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-8, 0.8), (0, 0.5), (8, 0.2)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = sum(np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28)) / k ** 1.4 for k in range(1, 7) if f * k < 6000)
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.6))
    return np.vstack([L * env, R * env])


def plip(m, vel=0.4, rise=1.0, decay=0.035):
    """A water drop: a sine whose pitch springs upward as it rings (the bubble resonance)."""
    f0 = float(mtof(m)); t = tt(0.3)
    fr = f0 * 2 ** (rise * (1 - np.exp(-t / 0.012)))
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / decay) * np.minimum(1, t / 0.0008) * vel


def kick(vel=0.8):
    t = tt(0.4)
    ph = 2 * np.pi * np.cumsum(55 + 70 * np.exp(-t / 0.025)) / SR
    return np.sin(ph) * np.exp(-t / 0.16) * np.minimum(1, t / 0.002) * vel


def snap(vel=0.5):
    """Finger snap: a very short band of noise plus a tiny knock."""
    t = tt(0.08)
    nz = fft_filter(noise(len(t)), lo=1300, hi=5000); nz /= nz.std()
    return (nz * np.exp(-t / 0.007) + 0.5 * np.sin(2 * np.pi * 2100 * t) * np.exp(-t / 0.004)) * np.minimum(1, t / 0.0005) * vel


def wood(vel=0.5, f=1180.0):
    t = tt(0.12)
    tone = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.020) + 0.4 * np.sin(2 * np.pi * 2.71 * f * t) * np.exp(-t / 0.007)
    return tone * np.minimum(1, t / 0.0005) * vel


def tick(vel=0.3):
    t = tt(0.04)
    nz = fft_filter(noise(len(t)), lo=7000, hi=13000); nz /= nz.std()
    return nz * np.exp(-t / 0.008) * np.minimum(1, t / 0.0008) * vel


# ---------------------------------------------------------------- harmony
CH = {   # bass root, mid voicing (comp dyads come from the top of it, pad uses all)
    'Gmaj7':    (43, [59, 62, 66, 69]),
    'Cmaj7#11': (36, [59, 64, 66, 67]),
    'Am9':      (45, [60, 64, 67, 71]),
    'D7sus4':   (38, [60, 62, 67, 69]),
    'Bm9':      (47, [61, 62, 66, 69]),
    'E7':       (40, [62, 64, 68, 71]),
    'Em7/G':    (43, [62, 64, 67, 71]),
    'Cmaj9/E':  (40, [62, 64, 67, 71]),
    'Gmaj9':    (43, [59, 62, 66, 69]),
    'Gend':     (43, [59, 62, 66, 69, 74]),
}
PROG = [(0, 2, 'Gmaj7'), (1, 0, 'Gmaj7'), (2, 0, 'Cmaj7#11'), (3, 0, 'Am9'), (3, 2, 'D7sus4'),
        (4, 0, 'Bm9'), (4, 2, 'E7'), (5, 0, 'Am9'), (5, 2, 'Em7/G'), (6, 0, 'Cmaj9/E'), (6, 2, 'D7sus4'),
        (7, 0, 'Gmaj9'), (7, 2, 'Cmaj7#11'), (7, 3, 'D7sus4'), (8, 0, 'Gend')]
CHORD_TONES = {   # pitch classes for the harmony voice
    'Gmaj7': [7, 11, 2, 6], 'Cmaj7#11': [0, 4, 7, 11, 6], 'Am9': [9, 0, 4, 7, 11], 'D7sus4': [2, 7, 9, 0],
    'Bm9': [11, 2, 6, 9, 1], 'E7': [4, 8, 11, 2], 'Em7/G': [4, 7, 11, 2], 'Cmaj9/E': [0, 4, 7, 11, 2],
    'Gmaj9': [7, 11, 2, 6, 9], 'Gend': [7, 11, 2, 6, 9]}


def chord_at(bar, pos):
    c = PROG[0][2]
    for b, p, name in PROG:
        if (bar, pos) >= (b, p):
            c = name
    return c


def segs():
    for i, (bar, pos, c) in enumerate(PROG):
        nb, npos = (PROG[i + 1][0], PROG[i + 1][1]) if i + 1 < len(PROG) else (9, 0)
        yield bar, pos, c, (nb - bar) * 4 + (npos - pos)


def below(bar, pos, m, lo=3, hi=9):
    pcs = CHORD_TONES[chord_at(bar, pos)]
    cand = [n for n in range(m - hi, m - lo + 1) if n % 12 in pcs]
    return cand[-1] if cand else None


# ---------------------------------------------------------------- melody (bar, beat, midi, accent)
G5, A5, B5, C6, D6, E6, F6s, G6, A6 = 79, 81, 83, 84, 86, 88, 90, 91, 93
D5, E5, F5s, G5s = 74, 76, 78, 80
MEL = [
    # pickup
    (0, 3.0, 74, 0), (0, 3.25, 79, 0), (0, 3.5, 81, 0), (0, 3.75, 83, 0),
    # bar 1  Gmaj7
    (1, 0, D6, 1), (1, .5, B5, 0), (1, .75, D6, 0), (1, 1, G6, 1), (1, 1.5, F6s, 0), (1, 1.75, D6, 0),
    (1, 2, E6, 1), (1, 2.5, D6, 0), (1, 3, B5, 1), (1, 3.5, A5, 0), (1, 3.75, B5, 0),
    # bar 2  Cmaj7#11
    (2, 0, E6, 1), (2, .5, C6, 0), (2, .75, E6, 0), (2, 1, F6s, 1), (2, 1.5, G6, 0), (2, 2, E6, 'roll1'),
    (2, 3, 'trip', [G6, F6s, E6]), (2, 3.25, D6, 0), (2, 3.5, B5, 0), (2, 3.75, D6, 0),
    # bar 3  Am9 | D7sus4
    (3, 0, E6, 1), (3, .5, C6, 0), (3, .75, E6, 0), (3, 1, A6, 1), (3, 1.5, G6, 0), (3, 1.75, E6, 0),
    (3, 2, G6, 1), (3, 2.5, D6, 0), (3, 3, C6, 1), (3, 3.5, A5, 0), (3, 3.75, C6, 0),
    # bar 4  Bm9 | E7
    (4, 0, D6, 'roll1'), (4, 1, F6s, 1), (4, 1.5, E6, 0), (4, 1.75, D6, 0),
    (4, 2, B5, 1), (4, 2.5, D6, 0), (4, 2.75, B5, 0), (4, 3, G5s, 1), (4, 3.5, B5, 0), (4, 3.75, D6, 0),
    # bar 5  the slide: rolling figures that bounce back up, each a step lower
    (5, 0, 'trip', [E6, D6, C6]), (5, .5, G6, 1), (5, .75, E6, 0),
    (5, 1, 'trip', [D6, C6, B5]), (5, 1.5, E6, 1), (5, 1.75, C6, 0),
    (5, 2, 'trip', [B5, A5, G5]), (5, 2.5, D6, 1), (5, 2.75, B5, 0),
    (5, 3, 'trip', [A5, G5, F5s]), (5, 3.5, B5, 1), (5, 3.75, G5, 0),
    # bar 6  Cmaj9/E, then the stretch (tremolo dyads, see below)
    (6, 0, 'trip', [G5, F5s, E5]), (6, .5, C6, 1), (6, .75, B5, 0),
    (6, 1, 'trip', [E6, D6, C6]), (6, 1.5, A5, 1), (6, 1.75, G5, 0),
    # bar 7  after the fall: the theme comes back
    (7, 1.5, B5, 0), (7, 1.75, D6, 0),
    (7, 2, E6, 1), (7, 2.5, G6, 0), (7, 2.75, E6, 0), (7, 3, A6, 1), (7, 3.25, G6, 0), (7, 3.5, E6, 0), (7, 3.75, F6s, 0),
    # bar 8  home
    (8, 0, G6, 'end'),
]

for ev in MEL:
    bar, pos = ev[0], ev[1]
    if ev[2] == 'trip':
        for j, m in enumerate(ev[3]):           # "또르르": 16th-note triplet run
            place('lead', marimba(m, [0.62, 0.50, 0.46][j], 0.6), B1 + (bar - 1) * BAR + (pos + j / 6) * BEAT,
                  pan=float(0.15 - 0.1 * j))
        continue
    m, acc = ev[2], ev[3]
    t0 = tb(bar, pos)
    if acc == 'roll1':                           # rolled long note (one beat), the marimba way to sustain
        k = 0
        while k * 0.075 < BEAT * 0.98:
            v = 0.62 if k == 0 else 0.30 + 0.06 * np.sin(k * 1.7)
            place('lead', marimba(m, v, 0.5), t0 + k * 0.075 + rng.normal(0, 0.003), pan=0.1 if k % 2 else 0.0)
            k += 1
        continue
    vel = 0.78 if acc == 'end' else (0.70 if acc else 0.54)
    vel *= 0.9 if bar == 0 else 1.0
    place('lead', marimba(m, vel, 0.65), t0 + rng.normal(0, 0.002), pan=0.08)
    # a second marimba under the main notes from bar 3 on (thirds/sixths below)
    if bar >= 3 and acc and pos % 1 == 0:
        u = below(bar, pos, m)
        if u:
            place('harmony', marimba(u, vel * 0.8, 0.45), t0 + 0.004, pan=-0.25)
    if acc and bar in (3, 4, 7, 8) and pos % 2 == 0:
        place('glock', glock(m + 12, 0.35 if acc != 'end' else 0.5), t0, pan=0.3)

# bar 8: last sparkle and a high plip
for j, m in enumerate([86, 91, 95, 98]):
    place('glock', glock(m, 0.30 - 0.04 * j), tb(8, 0.5) + j * 0.07, pan=float(-0.3 + 0.2 * j))

# the stretch: tremolo dyads climbing and swelling toward the release
STR = [(2.0, (A5, D6)), (2.5, (C6, E6)), (3.0, (D6, G6)), (3.5, (E6, A6))]
for pos, (a, b) in STR:
    t0 = tb(6, pos); k = 0
    while t0 + k * 0.045 < tb(6, pos + 0.5) - 0.01:
        tk = t0 + k * 0.045
        prog = (tk - tb(6, 2)) / (RELEASE - tb(6, 2))
        v = 0.20 + 0.42 * prog ** 1.5
        place('roll', marimba(a if k % 2 == 0 else b, v, 0.5), tk + rng.normal(0, 0.002), pan=-0.12 if k % 2 == 0 else 0.12)
        k += 1

# the release: an accent, then a fast run that falls with the drop (until it leaves the frame)
place('run', marimba(G6, 0.80, 0.8), RELEASE, pan=0.1)
place('run', marimba(B5, 0.55, 0.6), RELEASE + 0.004, pan=-0.1)
RUN = [F6s, E6, D6, C6, B5, A5, G5, F5s, E5, D5, 71, 69, 67]
for j, m in enumerate(RUN):
    tj = RELEASE + 0.06 + j * (EXIT - 0.12 - RELEASE) / len(RUN) * (1 - 0.25 * j / len(RUN))
    place('run', marimba(m, 0.58 - 0.018 * j, 0.6), tj, pan=float(0.4 - 0.065 * j))

# water drops: two in the intro, the big drop appearing, the release, the exit, the end
place('drop', plip(84, 0.45, 1.0), 0.10, pan=-0.3)
place('drop', plip(88, 0.35, 0.9), 0.52, pan=0.35)
place('drop', plip(91, 0.30, 0.9), tb(3, 2) - 0.03, pan=0.2)
place('drop', plip(79, 0.55, 1.2, 0.05), RELEASE + 0.01, pan=0.05)
place('drop', plip(64, 0.55, 1.4, 0.07), EXIT, pan=0.15)
place('drop', plip(93, 0.25, 0.8), tb(8, 1.5), pan=-0.3)

# glock sparkle when the big drop becomes visible
for j, m in enumerate([93, 98, 103]):
    place('glock', glock(m, 0.28 - 0.05 * j), tb(3, 2) + j * 0.08, pan=float(0.1 + 0.25 * j))

# the web's recoil: a vibraphone chord with a 3.14 Hz tremolo that dies away like the vibration
trem = lambda t: 1 - (0.15 + 0.75 * np.exp(-2.2 * t)) * (0.5 - 0.5 * np.cos(2 * np.pi * RECOIL_HZ * t))
for j, m in enumerate([67, 71, 74, 78, 81]):
    place('vibes', vibe_note(m, 4.2, 0.55, trem), RELEASE + 0.01 * j, pan=float(-0.5 + 0.25 * j))
# a quieter vibe swell for the ending chord
for j, m in enumerate([67, 71, 74, 78, 81, 86]):
    place('vibes', vibe_note(m, 3.5, 0.32, lambda t: 1 - 0.3 * (0.5 - 0.5 * np.cos(2 * np.pi * 4.2 * t))),
          tb(8, 0) + 0.012 * j, pan=float(-0.5 + 0.2 * j))

# comp: off-beat marimba dyads (soft mallets) + pizzicato bass + soft pad
for bar, pos, c, span in segs():
    root, v = CH[c]
    if bar == 0:
        place('pad', pad(v, (B1 - tb(0, pos)) + 0.1, attack=0.9), tb(0, pos), gain=0.6)
        continue
    stretch = bar == 6 and pos >= 2
    place('pad', pad(v, span * BEAT + 0.05, attack=0.35 if bar > 1 else 0.6), tb(bar, pos),
          gain=1.3 if stretch else (0.8 if bar <= 2 else 1.0))
    if bar == 8:
        place('bass', pizz(root, 0.85), tb(8, 0))
        place('bass', pizz(root - 12 if root - 12 >= 31 else root, 0.6), tb(8, 0) + 0.003)
        continue
    if stretch:
        place('bass', pizz(root, 0.8), tb(bar, pos))
        continue
    top = v[-2:]
    for b in np.arange(pos, pos + span, 1.0):
        place('comp', marimba(top[0], 0.42, 0.35), tb(bar, b + 0.5), pan=-0.3)
        place('comp', marimba(top[1], 0.38, 0.35), tb(bar, b + 0.5) + 0.006, pan=-0.2)
    bl = root if root >= 40 else root + 12
    place('bass', pizz(bl, 0.85), tb(bar, pos))
    if span >= 4:
        place('bass', pizz(bl + 7, 0.55), tb(bar, pos + 1.5))
        place('bass', pizz(bl + 12, 0.6), tb(bar, pos + 2.5))
        place('bass', pizz(bl + 7, 0.5), tb(bar, pos + 3.5))
    elif span >= 2 and bar != 7:
        place('bass', pizz(bl + 7, 0.55), tb(bar, pos + 1.5))

# drums: light and woody
for bar in range(1, 8):
    for pos in np.arange(0, 4, 0.25):
        t0 = tb(bar, pos)
        if bar == 6 and pos >= 2:
            continue
        if bar == 7 and pos < 2:
            continue
        on8 = pos % 0.5 == 0
        if bar >= 3 or on8:
            place('tick', tick((0.42 if on8 else 0.24) * (0.7 if bar <= 2 else 1.0)), t0 + 0.006, pan=0.35)
    lv = 0.75 if bar <= 2 else 1.0
    if not (bar == 7):
        place('kick', kick(0.8 * lv), tb(bar, 0))
    if bar >= 3 and bar not in (6, 7):
        place('kick', kick(0.55), tb(bar, 2.5))
    for pos in (1, 3):
        if (bar == 6 and pos == 3) or (bar == 7 and pos == 1):
            continue
        place('snap', snap(0.65 * lv), tb(bar, pos) + 0.008, pan=-0.15)
    for pos in ((1.5, 3.5) if bar <= 2 else (1.5, 3.25, 3.75)):
        if (bar == 6 and pos > 2) or (bar == 7 and pos < 2):
            continue
        place('wood', wood(0.5 * lv, 1180 if pos != 3.75 else 1520), tb(bar, pos), pan=0.25)
place('kick', kick(0.55), RELEASE)
place('kick', kick(0.6), tb(7, 2))
place('kick', kick(0.75), tb(8, 0))
place('wood', wood(0.5, 1520), tb(8, 0), pan=0.25)


# ---------------------------------------------------------------- mix
def reverb(x2, T=2.3):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.4)
        ir = (fft_filter(nz, hi=8000) * w + fft_filter(nz, hi=3000) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.02 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


STEMS['pad'] = fft_filter(fft_filter(STEMS['pad'].astype(float), hi=2600), lo=150).astype(np.float32)
STEMS['comp'] = fft_filter(STEMS['comp'].astype(float), lo=150, hi=6000).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, B1), A=(B1, tb(3)), A2=(tb(3), tb(5)), slide=(tb(5), RELEASE), fall=(RELEASE, tb(8)), end=(tb(8), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.30 * reverb(snd)
mix = fft_filter(mix, lo=35)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 18.9, 1.0, np.cos(np.clip((t - 18.9) / 1.1, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.02)
mix /= np.abs(mix).max()
drive = 1.15
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
