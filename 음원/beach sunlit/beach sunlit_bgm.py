"""Original BGM for beach sunlit.mp4 (20 s): the same beach at sunset, the sun 6 degrees above the sea.

A lyrical, warm 6/8 piano ballad in D-flat major that still moves: the piano sings the tune
(single notes in the verse, octaves with a chord tone in the chorus) over rocking 6/8
arpeggios, warm strings, a soft bass and a 6/8 ballad beat. The light glittering on the water
(yoon-seul) is a clean guitar shimmering in 16ths plus little bells with echoes; the sunset glow is
a warm pad with a slight tape drift. Chorus on I - V/3 - vi - iii - IV - iii - ii - V.

Sync (the HUD's "breaking" flag): waves break at 0.50, 4.77, 9.20, 14.10 and 19.17 s. One 6/8 bar
is 2.215 s (dotted quarter = 54.2 BPM), so the 4.77 s wave starts the verse and the 9.20 s wave the
chorus on bar downbeats (crashes), the 14.10 s wave gets a falling cascade of glints, and the last
wave (19.17 s) lands exactly on beat 2 of the final bar with the tune's last note. No noise beds.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261122)

BREAKS = [0.50, 4.77, 9.20, 14.10, 19.17]
BAR = (9.20 - 4.77) / 2           # 2.215 s, one bar of 6/8
E8 = BAR / 6                      # eighth note, 0.369 s
B0 = 4.77 - 2 * BAR               # bar 0 = 0.34 s; verse = bar 2 (4.77); chorus = bar 4 (9.20); end = bar 8 (18.06)


def tb(bar, e=0.0):
    return B0 + bar * BAR + e * E8


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
GAIN = dict(melody=0.42, lh=0.20, gtr=0.16, strings=0.038, cello=0.040, pad=0.030, bass=0.15, bells=0.11,
            glint=0.24, swell=0.07, kick=0.26, snare=0.34, rim=0.30, hat=0.13, tom=0.15, crash=0.06)
SEND = dict(melody=.30, lh=.25, gtr=.35, strings=.45, cello=.40, pad=.45, bass=.04, bells=.55, glint=.55,
            swell=.40, kick=.05, snare=.22, rim=.25, hat=.10, tom=.20, crash=.25)


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


def echo(name, sig, t, pan, gain, delay=E8 * 1.5, fb=0.36, taps=2):
    place(name, sig, t, pan, gain)
    for k in range(1, taps + 1):
        place(name, fft_filter(sig, hi=4500 / k), t + delay * k, -0.6 if k % 2 else 0.6, gain * fb ** k)


# ---------------------------------------------------------------- instruments
def piano(m, dur, vel=0.5):
    f = float(mtof(m)); T = dur + 1.0; t = tt(T)
    t60 = float(np.interp(m, [36, 60, 84, 100], [7.5, 4.5, 2.0, 0.9]))
    out = np.zeros_like(t)
    for n in range(1, 14):
        fn = f * n * np.sqrt(1 + 0.0004 * n * n)
        if fn > 11000:
            break
        a = (1.0 / n ** 1.2) * ((0.28 + 0.52 * vel) ** (n - 1))
        tau = t60 / 6.9 / (1 + 0.45 * (n - 1))
        out += a * np.sin(2 * np.pi * fn * t) * (0.6 * np.exp(-t / (tau * 0.35)) + 0.4 * np.exp(-t / tau))
    out *= np.minimum(1, t / 0.002) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.15))
    return out * vel


def eguitar(m, dur, vel=0.5):
    f = float(mtof(m)); T = dur + 0.5; t = tt(T)
    tau1 = float(np.interp(m, [40, 64, 88], [2.4, 1.4, 0.7]))
    out = np.zeros_like(t)
    for n in range(1, 16):
        if f * n > 8000:
            break
        a = abs(np.sin(n * np.pi * 0.16)) / n ** 1.05
        out += a * np.sin(2 * np.pi * f * n * t + rng.uniform(0, 0.3)) * np.exp(-t / (tau1 / (1 + 0.3 * (n - 1) ** 1.15)))
    env = np.minimum(1, t / 0.002) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.1))
    return out * env * vel


def bowed(m, dur, vel=0.6, attack=0.6, voices=5, bright=3000):
    f0 = float(mtof(m)); T = dur + 1.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for v in range(voices):
        vib = 12 * np.sin(2 * np.pi * rng.uniform(4.8, 5.8) * t + rng.uniform(0, 6)) * np.clip(t / 0.4, 0, 1) + rng.normal(0, 6)
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** (vib / 1200)) / SR
        s = sum(np.sin(k * ph) / k / np.sqrt(1 + (f0 * k / bright) ** 4) for k in range(1, 14) if f0 * k < 9000)
        w = (v + 0.5) / voices
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.4))
    return np.vstack([L * env, R * env]) * vel


def pad(notes, dur, attack=1.0):
    T = dur + 2.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-9, 0.8), (0, 0.5), (9, 0.2)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = sum(np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28)) / k ** 1.3 for k in range(1, 9) if f * k < 6000)
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.7))
    return np.vstack([L * env, R * env])


def ebass(m, dur, vel=0.7):
    f = float(mtof(m)); T = dur + 0.2; t = tt(T)
    out = sum((1.0 / k ** 1.5) * np.sin(2 * np.pi * f * k * t) * ((1.0 if k <= 2 else 0.25 / k) + (1 - (1.0 if k <= 2 else 0.25 / k)) * np.exp(-t * (7 + 5 * k))) for k in range(1, 10) if f * k < 2500)
    env = np.minimum(1, t / 0.008) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.06))
    return np.tanh(1.1 * out) * env * vel


def bell(m, vel=0.3):
    f = float(mtof(m)); t = tt(1.8)
    out = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.8) + 0.16 * np.sin(2 * np.pi * 2.0 * f * t) * np.exp(-t / 0.2)
    out += 0.05 * np.sin(2 * np.pi * 3.0 * f * t) * np.exp(-t / 0.07)
    return out * np.minimum(1, t / 0.0015) * vel


def kick(vel=0.8):
    t = tt(0.45)
    ph = 2 * np.pi * np.cumsum(50 + 80 * np.exp(-t / 0.025)) / SR
    return np.tanh(1.1 * np.sin(ph) * np.exp(-t / 0.22)) * np.minimum(1, t / 0.002) * vel


def snare(vel=0.8):
    t = tt(0.5)
    tone = np.sin(2 * np.pi * np.cumsum(180 * (1 + 0.3 * np.exp(-t / 0.01))) / SR) * np.exp(-t / 0.08)
    nz = fft_filter(noise(len(t)), lo=900, hi=6500); nz /= nz.std()
    return (0.85 * tone + 0.42 * np.tanh(0.7 * nz) * np.exp(-t / 0.13)) * np.minimum(1, t / 0.001) * vel


def rim(vel=0.5):
    t = tt(0.12)
    tone = np.sin(2 * np.pi * 760 * t) * np.exp(-t / 0.02) + 0.5 * np.sin(2 * np.pi * 1580 * t) * np.exp(-t / 0.01)
    return tone * np.minimum(1, t / 0.0006) * vel


def hat(vel=0.3):
    t = tt(0.06)
    nz = fft_filter(noise(len(t)), lo=6500, hi=12500); nz /= nz.std()
    return nz * np.exp(-t / 0.016) * np.minimum(1, t / 0.0008) * vel


def tom(m, vel=0.7):
    f = float(mtof(m)); t = tt(0.7)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.25 * np.exp(-t / 0.04))) / SR
    return (np.sin(ph) * np.exp(-t / 0.28) + 0.3 * np.sin(1.6 * ph) * np.exp(-t / 0.1)) * np.minimum(1, t / 0.002) * vel


def crash(vel=0.6):
    t = tt(2.8)
    nz = fft_filter(noise(len(t)), lo=3200, hi=11000); nz /= nz.std()
    metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in [2900, 4300, 5900]) / 3
    return (np.tanh(0.6 * nz) / 0.6 + 0.25 * metal) * np.exp(-t / 1.1) * np.minimum(1, t / 0.002) * vel


# ---------------------------------------------------------------- harmony (D-flat major)
CH = {   # bass root, 6/8 arpeggio (6 eighths), mid voicing (strings / pad / guitar)
    'Dbmaj9':  (37, [49, 56, 63, 65, 72, 68], [60, 63, 65, 68]),
    'Gbmaj9':  (42, [54, 61, 68, 70, 77, 73], [61, 65, 68, 70]),
    'Absus4':  (44, [56, 63, 66, 68, 73, 68], [61, 63, 66, 68]),
    'Ab':      (44, [56, 63, 66, 68, 72, 68], [60, 63, 66, 68]),
    'Bbm9':    (46, [46, 53, 60, 61, 65, 60], [61, 65, 68, 72]),
    'Ab/C':    (48, [48, 56, 63, 68, 72, 63], [60, 63, 68, 72]),
    'Fm7/Ab':  (44, [44, 51, 60, 63, 68, 63], [60, 63, 65, 68]),
    'Gbmaj7':  (42, [54, 61, 65, 68, 73, 65], [61, 65, 66, 70]),
    'Fm7':     (41, [53, 60, 63, 68, 72, 63], [60, 63, 65, 68]),
    'Ebm7':    (39, [51, 58, 61, 65, 70, 66], [61, 63, 66, 70]),
    'Ab7sus4': (44, [56, 63, 66, 68, 73, 66], [61, 63, 66, 68]),
    'Dbend':   (37, [49, 56, 63, 65, 68, 73], [61, 63, 65, 68, 73]),
}
PROG = [(0, 0, 'Dbmaj9'), (1, 0, 'Gbmaj9'), (1, 3, 'Absus4'),
        (2, 0, 'Bbm9'), (3, 0, 'Gbmaj7'), (3, 3, 'Ab'),
        (4, 0, 'Dbmaj9'), (4, 3, 'Ab/C'), (5, 0, 'Bbm9'), (5, 3, 'Fm7/Ab'),
        (6, 0, 'Gbmaj7'), (6, 3, 'Fm7'), (7, 0, 'Ebm7'), (7, 3, 'Ab7sus4'), (7, 5, 'Ab'), (8, 0, 'Dbend')]
TONES = {k: sorted({n % 12 for n in v[1] + v[2]}) for k, v in CH.items()}


def chord_at(bar, e):
    c = PROG[0][2]
    for b, p, name in PROG:
        if (bar, e) >= (b, p):
            c = name
    return c


def segs():
    for i, (bar, e, c) in enumerate(PROG):
        nb, ne = (PROG[i + 1][0], PROG[i + 1][1]) if i + 1 < len(PROG) else (9, 0)
        yield bar, e, c, (nb - bar) * 6 + (ne - e)


def under_tone(bar, e, m):
    cand = [n for n in range(m - 9, m - 2) if n % 12 in TONES[chord_at(bar, e)]]
    return cand[-1] if cand else None


# ---------------------------------------------------------------- melody (bar, eighth, eighths, midi)
Db4, Eb4, F4, Gb4, Ab4, Bb4, C5, Db5, Eb5, F5, Gb5, Ab5, Bb5 = 61, 63, 65, 66, 68, 70, 72, 73, 75, 77, 78, 80, 82
MEL = [
    # intro motif
    (1, 0, 2, F5), (1, 2, 1, Eb5), (1, 3, 2, Db5), (1, 5, 1, C5),
    # verse (from the 4.77 s wave)
    (2, 0, 2, Db5), (2, 2, 1, C5), (2, 3, 1.5, Db5), (2, 4.5, 0.5, Eb5), (2, 5, 1, F5),
    (3, 0, 1, Bb4), (3, 1, 1, Db5), (3, 2, 1, F5), (3, 3, 1.5, Eb5), (3, 4.5, 0.5, Db5), (3, 5, 1, Eb5),
    # chorus (from the 9.20 s wave)
    (4, 0, 2, F5), (4, 2, 1, Gb5), (4, 3, 2, Ab5), (4, 5, 1, Eb5),
    (5, 0, 1.5, Db5), (5, 1.5, 0.5, C5), (5, 2, 1, Db5), (5, 3, 2, C5), (5, 5, 1, Ab4),
    (6, 0, 1, Bb4), (6, 1, 1, Db5), (6, 2, 1, F5), (6, 3, 2, Ab5), (6, 5, 1, Eb5),
    (7, 0, 1.5, F5), (7, 1.5, 0.5, Eb5), (7, 2, 1, Gb5), (7, 3, 2, Bb5), (7, 5, 1, Ab5),
    (8, 0, 3, F5), (8, 3, 4, Db5),
]
for bar, e, d, m in MEL:
    t0 = tb(bar, e); dur = d * E8 * 0.98
    if bar <= 3:
        vel = 0.50 if bar == 1 else 0.58
        place('melody', piano(m, dur, vel), t0, pan=0.05)
        continue
    vel = 0.66 if e in (0, 3) else 0.58
    place('melody', piano(m, dur, vel), t0, pan=0.05)
    place('melody', piano(m - 12, dur, vel * 0.6), t0, pan=-0.05)
    if d >= 2:
        u = under_tone(bar, e, m)
        if u:
            place('melody', piano(u, dur, vel * 0.42), t0 + 0.01, pan=0.0)

# piano left hand: rocking 6/8 arpeggios, a bass octave on each chord in the chorus
for bar, e0, c, span in segs():
    root, arp, mid = CH[c]
    if bar == 8:
        for j, m in enumerate(arp):
            place('lh', piano(m, 2.6, 0.40), tb(8, 0) + j * 0.06, pan=float(-0.3 + 0.1 * j))
        place('lh', piano(root, 2.6, 0.45), tb(8, 0), pan=-0.2)
        continue
    for k in range(span):
        e = e0 + k
        v = (0.44 if e % 3 == 0 else 0.34) * (0.85 if bar <= 1 else 1.0)
        place('lh', piano(arp[e % 6], E8 * 1.6, v), tb(bar, e), pan=float(-0.35 + 0.05 * (e % 6)))
    if bar >= 4:
        place('lh', piano(root if root >= 36 else root + 12, span * E8, 0.42), tb(bar, e0), pan=-0.25)

# the glitter on the water: clean guitar, 8ths in the verse, shimmering 16ths in the chorus
for bar, e0, c, span in segs():
    if bar < 2 or bar == 8:
        continue
    up = [n + 12 for n in CH[c][2]]
    step = 0.5 if bar >= 4 else 1.0
    pat = [0, 2, 1, 3, 2, 1] if step == 1.0 else [0, 2, 3, 1, 3, 2, 0, 3, 2, 1, 3, 2]
    for k in range(int(round(span / step))):
        e = e0 + k * step
        v = (0.34 if k % 2 == 0 else 0.26) * (1.0 if bar >= 4 else 0.8)
        place('gtr', eguitar(up[pat[k % len(pat)]], E8 * 1.5, v), tb(bar, e), pan=float(0.55 if k % 2 else -0.15))

# strings: a slow, warm bed from the verse; cellos join on the chorus
for bar, e0, c, span in segs():
    if bar < 2:
        continue
    root, arp, mid = CH[c]
    d = (span * E8 + 0.05) if bar < 8 else 2.2
    lv = 0.7 if bar < 4 else 1.0
    for m in mid:
        place('strings', bowed(m, d, 0.5, attack=0.7 if bar in (2, 4) else 0.35), tb(bar, e0), gain=lv)
    if bar >= 4:
        r = root + 12 if root < 43 else root
        place('cello', bowed(r, d, 0.5, attack=0.4, voices=4, bright=2200), tb(bar, e0), pan=-0.3)

# warm pad (the sunset glow), mostly the intro and under the chorus
for bar, e0, c, span in segs():
    lv = 1.0 if bar <= 1 else (0.5 if bar <= 3 else 0.55)
    d = (span * E8 + 0.05) if bar < 8 else 2.0
    place('pad', pad(CH[c][2], d, attack=1.2 if bar == 0 else 0.5), tb(bar, e0) if bar > 0 else 0.0, gain=lv)

# bass
for bar, e0, c, span in segs():
    if bar < 2:
        continue
    r = CH[c][0]
    if bar == 8:
        place('bass', ebass(r, 2.0, 0.8), tb(8, 0)); continue
    if bar < 4:
        place('bass', ebass(r, span * E8 * 0.95, 0.7), tb(bar, e0)); continue
    if span >= 3:
        place('bass', ebass(r, 2 * E8 * 0.95, 0.78), tb(bar, e0))
        place('bass', ebass(r + 7, E8 * 0.9, 0.55), tb(bar, e0 + 2))
        if span > 3:
            place('bass', ebass(r, (span - 3) * E8 * 0.95, 0.7), tb(bar, e0 + 3))
    else:
        place('bass', ebass(r, span * E8 * 0.95, 0.7), tb(bar, e0))

# glints of sunlight on the water: little bells on a 16th grid with echoes (D-flat major)
SCALE = [77, 80, 82, 84, 85, 87, 89, 92, 94]
dens = lambda t: float(np.interp(t, [0, 4.6, 4.8, 9.1, 9.3, 17.9, 18.1, 20], [0.30, 0.30, 0.16, 0.16, 0.10, 0.10, 0.30, 0.30]))
for k in range(int((DUR - 0.5) / (E8 / 2))):
    t = tb(0) + k * E8 / 2
    if 0.3 < t < 19.6 and rng.random() < dens(t):
        echo('bells', bell(int(rng.choice(SCALE)), rng.uniform(0.16, 0.30)), t + rng.uniform(0, 0.02),
             pan=float(rng.uniform(-.8, .8)), gain=1.0)

# waves: rising glints into the first, a falling cascade at 14.10, glints with the last note at 19.17
for j, m in enumerate([68, 73, 77, 80, 85, 89]):
    place('glint', bell(m, 0.26), 0.04 + j * 0.07, pan=float(-0.6 + 0.24 * j))
for j, m in enumerate([94, 92, 89, 87, 85, 82, 80, 77]):
    place('glint', bell(m, 0.32 - 0.02 * j), BREAKS[3] + j * 0.07, pan=float(0.7 - 0.18 * j))
for j, m in enumerate([85, 89, 92, 97]):
    place('glint', bell(m, 0.28 - 0.04 * j), BREAKS[4] + j * 0.08, pan=float(-0.4 + 0.27 * j))


# reversed piano chord swelling into the chorus (tonal, no noise)
def rev_swell(notes, T):
    sig = sum(piano(m, T, 0.5) for m in notes)[:int(T * SR)][::-1]
    return sig * np.linspace(0, 1, len(sig)) ** 1.6


place('swell', rev_swell([61, 65, 68, 72, 77], 1.8), BREAKS[2] - 1.8, pan=0.0)
place('swell', rev_swell([61, 65, 68, 73], 1.2), BREAKS[1] - 1.2, pan=-0.2, gain=0.6)

# drums: a 6/8 ballad beat from the verse, fuller in the chorus, tom fill into the last bar
for bar in range(2, 8):
    chorus = bar >= 4
    place('kick', kick(0.85 if chorus else 0.65), tb(bar, 0))
    if chorus or bar == 3:
        place('kick', kick(0.5), tb(bar, 5))
    if chorus:
        if not (bar == 7):
            place('snare', snare(0.8), tb(bar, 3) + 0.006, pan=-0.05)
    else:
        place('rim', rim(0.55), tb(bar, 3) + 0.006, pan=-0.1)
    step = 0.5 if bar >= 6 else 1.0
    for k in range(int(6 / step)):
        e = k * step
        if bar == 7 and e >= 3:
            continue
        v = 0.40 if e % 3 == 0 else (0.28 if e % 1 == 0 else 0.18)
        place('hat', hat(v * (0.8 if bar < 4 else 1.0)), tb(bar, e) + 0.008, pan=0.3)
place('snare', snare(0.8), tb(7, 3) + 0.006, pan=-0.05)
for j, (e, m) in enumerate([(3.5, 57), (4, 55), (4.5, 52), (5, 50), (5.5, 45)]):
    place('tom', tom(m, 0.6 + 0.05 * j), tb(7, e), pan=float(0.3 - 0.15 * j))
for t0, v in ((tb(2), 0.5), (tb(4), 0.9), (tb(6), 0.55), (tb(8), 0.8)):
    place('crash', crash(v), t0, pan=0.25)
place('kick', kick(0.9), tb(8))


# ---------------------------------------------------------------- mix
def reverb(x2, T=2.8):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.45)
        ir = (fft_filter(nz, hi=7000) * w + fft_filter(nz, hi=2800) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.03 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


def wow(x2, depth_ms=0.9, rate=0.5):
    n = x2.shape[1]; idx = np.arange(n, dtype=float)
    d = (depth_ms / 1000 * SR) * (1 + np.sin(2 * np.pi * rate * idx / SR)) + 2
    return np.vstack([np.interp(idx - d, idx, x2[c]) for c in range(2)])


def chorus_fx(x2, depth_ms=2.0, rate=0.8):
    n = x2.shape[1]; idx = np.arange(n, dtype=float)
    out = np.empty_like(x2)
    for c, ph in ((0, 0.0), (1, np.pi / 2)):
        d = (7 + depth_ms * (1 + np.sin(2 * np.pi * rate * idx / SR + ph)) / 2) / 1000 * SR
        out[c] = 0.7 * x2[c] + 0.5 * np.interp(idx - d, idx, x2[c])
    return out


for k in ('pad', 'strings', 'cello'):
    STEMS[k] = wow(STEMS[k].astype(float)).astype(np.float32)
STEMS['gtr'] = chorus_fx(fft_filter(STEMS['gtr'].astype(float), lo=250, hi=7000)).astype(np.float32)
STEMS['pad'] = fft_filter(fft_filter(STEMS['pad'].astype(float), hi=2200), lo=150).astype(np.float32)
STEMS['strings'] = fft_filter(STEMS['strings'].astype(float), lo=180, hi=4500).astype(np.float32)
STEMS['lh'] = fft_filter(STEMS['lh'].astype(float), lo=60).astype(np.float32)
STEMS['swell'] = fft_filter(STEMS['swell'].astype(float), lo=200).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, tb(2)), verse=(tb(2), tb(4)), chorus1=(tb(4), tb(6)), chorus2=(tb(6), tb(8)), end=(tb(8), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.32 * reverb(snd)
mix = fft_filter(mix, lo=32)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.1, 1.0, np.cos(np.clip((t - 19.1) / 0.9, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.03)
mix /= np.abs(mix).max()
drive = 1.2
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
