"""Original BGM for snow.mp4 (20 s): a snowflake growing in a cold chamber, from a tiny hexagonal plate to a
fully branched star.

Cold but powerful and determined, in the spirit of an ice-queen anthem (original melody): a driving
piano ostinato in D minor that breaks out into a soaring F-major chorus. Mostly acoustic sounds: piano
(melody in octaves, left-hand bass octaves), low strings, horns and a cold "ah" choir as the bed,
timpani and a deep orchestral drum. The bass is emphasised: piano octaves, contrabasses, timpani and
the big drum carry the weight. Celesta only as sparse ice glints.

Sync (from the HUD): the temperature plunges from -2 to -15 C between 2.5 and 4.1 s (timpani roll and a
low swell), the six arms shoot out at 3.60 s (a full low hit on a bar downbeat), side branches start at
5.46 s (the ostinato and the verse begin), the chorus breaks out at 9.19 s, and the flake is fully
grown at 16.63 s (the big F-major arrival, then a glinting sweep as the light turns across it).
At 128.9 BPM all three HUD stage changes fall on bar downbeats (within 3 ms).
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261126)

PLUNGE, ARMS, BRANCHES, DONE = 2.5, 3.602, 5.464, 16.633
BAR = 1.8616                      # 128.9 BPM, fitted to the three stage changes
BEAT = BAR / 4
B0 = ARMS                         # bar 0 = six arms; bar 1 = side branches; bar 3 = chorus; bar 7 = fully grown


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
GAIN = dict(melody=0.46, ostinato=0.27, lh=0.22, basses=0.075, strings=0.045, horns=0.080, choir=0.030,
            timp=0.20, drum=0.32, cymbal=0.07, ice=0.11)
SEND = dict(melody=.25, ostinato=.22, lh=.12, basses=.15, strings=.45, horns=.40, choir=.50, timp=.25,
            drum=.15, cymbal=.30, ice=.55)


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
def piano(m, dur, vel=0.5):
    f = float(mtof(m)); T = dur + 1.0; t = tt(T)
    t60 = float(np.interp(m, [24, 36, 60, 84, 100], [9.0, 7.5, 4.2, 1.9, 0.9]))
    out = np.zeros_like(t)
    for n in range(1, 16):
        fn = f * n * np.sqrt(1 + 0.0004 * n * n)
        if fn > 11000:
            break
        a = (1.0 / n ** 1.1) * ((0.30 + 0.55 * vel) ** (n - 1))
        tau = t60 / 6.9 / (1 + 0.42 * (n - 1))
        out += a * np.sin(2 * np.pi * fn * t + rng.uniform(0, 0.2)) * (0.62 * np.exp(-t / (tau * 0.35)) + 0.38 * np.exp(-t / tau))
    out *= np.minimum(1, t / 0.002) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.14))
    return out * vel


def bowed(m, dur, vel=0.6, attack=0.4, voices=5, bright=2600):
    f0 = float(mtof(m)); T = dur + 1.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for v in range(voices):
        vib = 10 * np.sin(2 * np.pi * rng.uniform(4.8, 5.8) * t + rng.uniform(0, 6)) * np.clip(t / 0.4, 0, 1) + rng.normal(0, 6)
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** (vib / 1200)) / SR
        s = sum(np.sin(k * ph) / k / np.sqrt(1 + (f0 * k / bright) ** 4) for k in range(1, 16) if f0 * k < 7000)
        w = (v + 0.5) / voices
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.35))
    return np.vstack([L * env, R * env]) * vel


def horns(notes, dur, vel=0.6, attack=0.3):
    T = dur + 0.8; t = tt(T)
    fc = 450 + 1000 * np.clip(t / (attack * 2), 0, 1)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-5, .7), (5, .3)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = sum(np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6)) / k / np.sqrt(1 + (f * k / fc) ** 4) for k in range(1, 18) if f * k < 6000)
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.3))
    return np.vstack([np.tanh(0.8 * L), np.tanh(0.8 * R)]) * env * vel


VOWEL_AH = [(700, 110, 1.0), (1150, 130, 0.45), (2600, 200, 0.08)]


def choir(m, dur, attack=0.5):
    f0 = float(mtof(m)); T = dur + 1.4; t = tt(T)
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
    return (body + 0.2 * nz * np.exp(-t / 0.02)) * np.minimum(1, t / 0.002) * vel


def big_drum(vel=0.9):
    """A deep orchestral bass drum (taiko-like): low skin pitch dropping, long boom, a soft slap."""
    t = tt(1.2)
    ph = 2 * np.pi * np.cumsum(46 + 40 * np.exp(-t / 0.05)) / SR
    boom = np.sin(ph) * np.exp(-t / 0.42) + 0.3 * np.sin(1.58 * ph) * np.exp(-t / 0.12)
    slap = fft_filter(noise(len(t)), lo=150, hi=1500) * np.exp(-t / 0.012)
    return np.tanh(1.2 * (boom + 0.25 * slap / (np.abs(slap).max() + 1e-9))) * np.minimum(1, t / 0.003) * vel


def cymbal(vel=0.6, swell=0.0):
    T = 2.8 + swell; t = tt(T)
    nz = fft_filter(noise(len(t)), lo=3000, hi=10000); nz /= nz.std()
    metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in [2900, 4100, 5300]) / 3
    x = np.tanh(0.6 * nz) / 0.6 + 0.3 * metal
    env = np.where(t < swell, (t / max(swell, 1e-3)) ** 2.5, np.exp(-(t - swell) / 1.0)) if swell else np.exp(-t / 1.0) * np.minimum(1, t / 0.002)
    return x * env * vel


def ice(m, vel=0.3):
    """A cold glint: celesta-like, with a soft (not clicky) attack."""
    f = float(mtof(m)); t = tt(1.8)
    out = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.8) + 0.15 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.2)
    return out * np.sin(np.minimum(1, t / 0.006) * np.pi / 2) * vel


# ---------------------------------------------------------------- harmony (D minor -> F major)
CH = {   # bass root (octave 2), ostinato tones, string / horn / choir voicing
    'Dm':  (38, [57, 62, 65, 69], [57, 62, 65]),
    'Bb':  (34, [58, 62, 65, 70], [58, 62, 65]),
    'Gm':  (43, [55, 58, 62, 67], [55, 62, 67]),
    'A':   (45, [57, 61, 64, 69], [57, 61, 64]),
    'Asus': (45, [57, 62, 64, 69], [57, 62, 64]),
    'F':   (41, [57, 60, 65, 69], [57, 60, 65]),
    'C':   (36, [55, 60, 64, 67], [55, 60, 64]),
    'F/A': (45, [57, 60, 65, 69], [57, 60, 65]),
    'Gm7': (43, [55, 58, 62, 65], [55, 62, 65]),
    'Fend': (41, [57, 60, 65, 69, 72], [57, 60, 65, 69]),
}
PROG = [(-2, 0, 'Dm'), (-1, 0, 'Dm'), (-1, 3, 'Asus'),
        (0, 0, 'Dm'), (1, 0, 'Dm'), (1, 2, 'Bb'), (2, 0, 'Gm'), (2, 2, 'A'),
        (3, 0, 'F'), (3, 2, 'C'), (4, 0, 'Dm'), (4, 2, 'Bb'), (5, 0, 'F/A'), (5, 2, 'Bb'), (6, 0, 'Gm7'), (6, 2, 'C'),
        (7, 0, 'Fend')]


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


# ---------------------------------------------------------------- melody: piano in octaves (bar, beat, beats, midi)
A4, Bb4, C5, Cs5, D5, E5, F5, G5, A5, C6 = 69, 70, 72, 73, 74, 76, 77, 79, 81, 84
MEL = [
    # verse (from the side branches)
    (1, 0, 1, D5), (1, 1, .5, A4), (1, 1.5, .5, D5), (1, 2, 1.5, F5), (1, 3.5, .5, D5),
    (2, 0, .5, Bb4), (2, .5, .5, C5), (2, 1, 1, D5), (2, 2, 1, Cs5), (2, 3, 1, E5),
    # chorus (F major)
    (3, 0, .5, C5), (3, .5, 1.5, F5), (3, 2, .5, E5), (3, 2.5, 1.5, G5),
    (4, 0, 1.5, A5), (4, 1.5, .5, G5), (4, 2, 1, F5), (4, 3, 1, D5),
    (5, 0, .5, C5), (5, .5, 1, F5), (5, 1.5, .5, G5), (5, 2, 1.5, A5), (5, 3.5, .5, G5),
    (6, 0, 1, F5), (6, 1, .5, G5), (6, 1.5, .5, A5), (6, 2, 2, C6),
    # fully grown
    (7, 0, 3, A5), (7, 3, 1, G5), (8, 0, 3, F5),
]
for bar, p, d, m in MEL:
    t0 = tb(bar, p); dur = d * BEAT * 0.98
    chorus = bar >= 3
    vel = (0.68 if p % 1 == 0 else 0.60) + (0.05 if chorus else 0.0)
    place('melody', piano(m, dur, vel), t0, pan=0.08)
    place('melody', piano(m - 12, dur, vel * (0.75 if chorus else 0.6)), t0 + 0.004, pan=-0.05)
    if chorus and d >= 1:
        tones = CH[chord_at(bar, p)][2]
        cand = [n for n in range(m - 9, m - 2) if n % 12 in {x % 12 for x in tones}]
        if cand:
            place('melody', piano(cand[-1], dur, vel * 0.45), t0 + 0.01, pan=0.0)

# intro: a quiet, high, cold ostinato (D minor), then the driving 8th ostinato from the side branches
INTRO_PAT = [74, 69, 77, 69, 76, 69, 74, 69]
for k in range(16):
    t0 = tb(-2, k * 0.5)
    if t0 < 0:
        continue
    v = 0.30 + 0.12 * (k / 16)
    place('ostinato', piano(INTRO_PAT[k % 8] + 12, BEAT * 0.9, v), t0, pan=0.25)
for bar, pos, c, span in segs():
    if bar < 1 or bar >= 7:
        continue
    tones = CH[c][1]
    pat = [0, 2, 1, 3, 2, 1, 3, 2]
    for k in range(int(span * 2)):
        q = pos + k * 0.5
        v = (0.46 if q % 1 == 0 else 0.38) * (1.1 if bar >= 3 else 1.0)
        place('ostinato', piano(tones[pat[k % 8]], BEAT * 0.8, v), tb(bar, q), pan=-0.2)

# piano left hand: bass octaves (the backbone)
for bar, pos, c, span in segs():
    r = CH[c][0]
    if bar <= -1:
        if bar == -2:
            place('lh', piano(r - 12, 2 * BAR, 0.45), tb(-2, 0) if tb(-2, 0) > 0 else 0.0)
        continue
    if bar == 7:
        for m in (r - 12, r):
            place('lh', piano(m, 3.2, 0.85), tb(7, 0))
        continue
    if bar == 0:
        for m in (r - 12, r):
            place('lh', piano(m, BAR, 0.9), tb(0, 0))
        continue
    step = 1.0 if bar >= 3 else 2.0
    for q in np.arange(pos, pos + span, step):
        v = 0.78 if q % 2 == 0 else 0.62
        for m, g in ((r - 12, 0.65), (r, 1.0)):
            place('lh', piano(m, step * BEAT * 0.95, v), tb(bar, q), gain=g)

# contrabasses (arco), cellos/violas bed, horns and choir
for bar, pos, c, span in segs():
    r, _, voi = CH[c]
    d = span * BEAT + 0.05 if bar < 7 else 3.2
    if bar >= 0:
        place('basses', bowed(r - 12 if r - 12 >= 28 else r, d, 0.6, attack=0.15 if bar in (0, 7) else 0.3, voices=4, bright=1200), tb(bar, pos))
        place('basses', bowed(r, d, 0.45, attack=0.25, voices=4, bright=1600), tb(bar, pos), gain=0.7)
    if bar >= 1:
        lv = 0.6 if bar < 3 else 1.0
        for m in voi:
            place('strings', bowed(m, d, 0.5, attack=0.4 if bar != 3 else 0.2), tb(bar, pos), gain=lv)
    if bar >= 3:
        place('horns', horns([m - 12 for m in voi], d, 0.6 if bar < 7 else 0.75, attack=0.25), tb(bar, pos), pan=-0.25)
        for m in voi[1:]:
            place('choir', choir(m + 12, d, attack=0.35), tb(bar, pos), gain=1.0 if bar < 7 else 1.3)
# the cold low swell during the temperature plunge
for m in (38, 50):
    place('basses', bowed(m, ARMS - PLUNGE, 0.55, attack=ARMS - PLUNGE, voices=4, bright=1200), PLUNGE, gain=0.9)
for m in (62, 65, 69):
    place('choir', choir(m + 12, ARMS - PLUNGE + 0.6, attack=ARMS - PLUNGE), PLUNGE, gain=0.6)

# timpani: a roll through the temperature plunge into the six-arms hit; hits on the big moments
n_roll = 24
for k in range(n_roll):
    tk = PLUNGE + k * (ARMS - PLUNGE) / n_roll
    place('timp', timpani(38, 0.08 + 0.42 * (k / n_roll) ** 1.6), tk, pan=-0.1 if k % 2 else 0.1)
for t0, m, v in ((ARMS, 38, 0.95), (tb(1), 38, 0.6), (tb(3), 41, 0.85), (tb(5), 41, 0.7), (DONE, 41, 1.0)):
    place('timp', timpani(m, v), t0)
# a second roll into the fully-grown moment
for k in range(16):
    tk = tb(6, 2) + k * (2 * BEAT) / 16
    place('timp', timpani(36, 0.10 + 0.35 * (k / 16) ** 1.5), tk, pan=-0.1 if k % 2 else 0.1)

# the big drum: on the hits, then driving beats 1 and 3 (plus pick-ups) in the chorus
place('drum', big_drum(1.0), ARMS)
for bar in range(1, 7):
    if bar < 3:
        hits = [(0, .8), (2.5, .5)]
    else:
        hits = [(0, 1.0), (1.5, .55), (2, .85), (3.5, .5)]
    for q, v in hits:
        place('drum', big_drum(v), tb(bar, q))
place('drum', big_drum(1.0), DONE)

# cymbals: a swell into the chorus and into the fully-grown arrival
place('cymbal', cymbal(0.6, swell=1.2), tb(3) - 1.2, pan=0.2)
place('cymbal', cymbal(0.8, swell=1.4), DONE - 1.4, pan=0.2)
place('cymbal', cymbal(0.45), ARMS, pan=-0.2)

# ice glints: sparse in the intro, a sweep as the light turns across the finished flake
for t0, m in [(0.25, 86), (0.9, 93), (1.6, 89), (2.2, 98), (3.0, 91)]:
    place('ice', ice(m, 0.32), t0, pan=float(rng.uniform(-.6, .6)))
for bar, m in ((3, 89), (4, 93), (5, 96), (6, 91)):
    place('ice', ice(m, 0.26), tb(bar, 0) + 0.01, pan=0.45)
    place('ice', ice(m - 7, 0.18), tb(bar, 0.5) + 0.01, pan=-0.45)
for j, m in enumerate([81, 84, 86, 89, 93, 96, 98, 101]):
    place('ice', ice(m, 0.30 - 0.02 * j), DONE + 0.45 + j * 0.11, pan=float(-0.6 + 0.17 * j))
for j, m in enumerate([93, 89, 86, 84, 81]):
    place('ice', ice(m, 0.2), 18.6 + j * 0.13, pan=float(0.5 - 0.25 * j))


# ---------------------------------------------------------------- mix
def reverb(x2, T=3.0):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.5)
        ir = (fft_filter(nz, hi=7500) * w + fft_filter(nz, hi=3000) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.03 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


STEMS['strings'] = fft_filter(STEMS['strings'].astype(float), lo=180, hi=4000).astype(np.float32)
STEMS['basses'] = fft_filter(STEMS['basses'].astype(float), hi=1500).astype(np.float32)
STEMS['choir'] = fft_filter(STEMS['choir'].astype(float), lo=250, hi=4500).astype(np.float32)
STEMS['horns'] = fft_filter(STEMS['horns'].astype(float), lo=120, hi=3500).astype(np.float32)
STEMS['timp'] = fft_filter(STEMS['timp'].astype(float), hi=1800).astype(np.float32)
STEMS['drum'] = fft_filter(STEMS['drum'].astype(float), lo=36).astype(np.float32)
STEMS['lh'] = fft_filter(STEMS['lh'].astype(float), lo=32).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, ARMS), arms=(ARMS, BRANCHES), verse=(BRANCHES, tb(3)), chorus=(tb(3), DONE), grown=(DONE, 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.30 * reverb(snd)
mix = fft_filter(mix, lo=28)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.0, 1.0, np.cos(np.clip((t - 19.0) / 1.0, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.02)
mix /= np.abs(mix).max()
drive = 1.35
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
