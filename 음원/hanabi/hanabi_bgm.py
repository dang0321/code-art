"""Original Japanese summer-festival BGM for hanabi.mp4 (20 s).

Mood reference: bittersweet mid-tempo J-pop summer-night ballad, plus matsuri
instruments (taiko, shinobue, chanchiki bell, koto, furin). All synthesized here;
melody and arrangement are original.  Firework bursts in the video are scored
with exact-time sparkle/boom accents.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 3
rng = np.random.default_rng(20261118)

BPM = 96.0
BEAT = 60.0 / BPM
T0 = 1.90                      # bar-1 downbeat == first burst in the video


def tb(b):                     # absolute beat -> seconds
    return T0 + b * BEAT


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


# ---------------------------------------------------------------- stems
STEMS = {}
GAIN = dict(piano=0.34, koto=0.50, fue=0.32, pad=0.030, bass=0.32, odaiko=0.30,
            shime=0.32, kane=0.14, furin=0.16, spark=0.11, boom=0.20, whistle=0.15)
SEND = dict(piano=.28, koto=.30, fue=.38, pad=.45, bass=.04, odaiko=.16, shime=.12,
            kane=.15, furin=.55, spark=.45, boom=.12, whistle=.40)


def place(name, sig, t, pan=0.0, gain=1.0):
    buf = STEMS.setdefault(name, np.zeros((2, N), np.float32))
    i = int(round(t * SR))
    if sig.ndim == 1:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * np.sqrt(2)
    else:
        st = sig.copy()
    k = min(st.shape[1], max(int(0.02 * SR), int(0.05 * st.shape[1])))
    st[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2      # click-free tail
    if i < 0:
        st, i = st[:, -i:], 0
    n = min(st.shape[1], N - i)
    if n > 0:
        buf[:, i:i + n] += (st[:, :n] * gain).astype(np.float32)


# ---------------------------------------------------------------- instruments
def piano(m, dur, vel=0.5):
    f = float(mtof(m)); T = dur + 0.8; t = tt(T)
    t60 = float(np.interp(m, [36, 60, 84, 100], [7.0, 4.0, 1.8, 0.8]))
    out = np.zeros_like(t)
    for n in range(1, 14):
        fn = f * n * np.sqrt(1 + 0.0004 * n * n)
        if fn > 15000:
            break
        a = (1.0 / n ** 1.15) * ((0.35 + 0.6 * vel) ** (n - 1))
        tau = t60 / 6.9 / (1 + 0.45 * (n - 1))
        out += a * np.sin(2 * np.pi * fn * t) * (0.7 * np.exp(-t / (tau * 0.35)) + 0.3 * np.exp(-t / tau))
    out *= np.minimum(1, t / 0.003) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.10))
    k = int(0.015 * SR)
    nz = fft_filter(rng.standard_normal(k) * np.exp(-np.arange(k) / SR / 0.003), lo=1000, hi=6000)
    out[:k] += nz * 0.15 * vel
    return out * vel


def koto(m, vel=0.5, T=None):
    f = float(mtof(m)); T = T or float(np.interp(m, [50, 74, 98], [3.2, 2.2, 1.2])); t = tt(T)
    bend = 1 + 0.008 * np.exp(-t / 0.025)           # tsume twang settling
    ph = 2 * np.pi * f * np.cumsum(bend) / SR
    out = np.zeros_like(t)
    for n in range(1, 18):
        if f * n > 15000:
            break
        a = abs(np.sin(n * np.pi * 0.18)) / n ** 1.2
        out += a * np.sin(n * ph) * np.exp(-t * (1.2 + 0.8 * n) * (f / 440) ** 0.35)
    k = int(0.006 * SR)
    out[:k] += fft_filter(rng.standard_normal(k) * np.exp(-np.arange(k) / SR / 0.0012), lo=2000) * 0.3
    return out * np.minimum(1, t / 0.001) * vel


def fue(m, dur, vel=0.5, scoop=0.7, breath=0.10, swell=0.0, decay=None):
    f = float(mtof(m)); T = dur + 0.35; t = tt(T)
    cents = -100 * scoop * np.exp(-t / 0.045)
    vib = np.clip((t - 0.22) / 0.35, 0, 1) * np.sin(2 * np.pi * 5.4 * t) * 14
    fr = f * 2 ** ((cents + vib) / 1200)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    tone = np.sin(ph) + 0.30 * np.sin(2 * ph + 0.3) + 0.12 * np.sin(3 * ph + 0.7) + 0.05 * np.sin(4 * ph)
    env = np.minimum(1, t / 0.04) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.05))
    env *= 1 + swell * np.clip(t / max(dur, 1e-3), 0, 1)
    if decay:
        env *= np.exp(-t / decay)
    nz = fft_filter(rng.standard_normal(len(t)), lo=f * 0.9, hi=f * 3.5)
    nz /= nz.std() + 1e-9
    return (tone + breath * nz * (0.6 * np.exp(-t / 0.06) + 0.4)) * env * vel


def whistle(t_len, m0, m1):
    T = t_len; t = tt(T + 0.04); x = np.clip(t / T, 0, 1)
    fr = mtof(m0 + (m1 - m0) * x ** 1.6) * (1 + 0.004 * np.sin(2 * np.pi * 7 * t))
    ph = 2 * np.pi * np.cumsum(fr) / SR
    tone = np.sin(ph) + 0.15 * np.sin(2 * ph)
    nz = fft_filter(rng.standard_normal(len(t)), lo=1800, hi=8000)
    nz /= nz.std()
    env = (0.15 + 0.85 * x ** 1.3) * np.where(t < T, 1.0, np.exp(-(t - T) / 0.012))
    return (tone + 0.35 * nz) * env


def pad(notes, dur, attack=0.35):
    T = dur + 2.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-8, 0.85), (0, 0.5), (8, 0.15)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = np.zeros_like(t)
            for k in range(1, 9):
                if f * k > 9000:
                    break
                s += np.sin(2 * np.pi * f * k * t + rng.uniform(0, 2 * np.pi)) / k
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2
    env *= np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.35))
    return np.vstack([L * env, R * env])


def bass(m, dur, vel=0.6):
    f = float(mtof(m)); T = dur + 0.25; t = tt(T)
    ph = 2 * np.pi * f * t
    tone = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    env = np.minimum(1, t / 0.006) * (0.65 + 0.35 * np.exp(-t / 0.25))
    env *= np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.06))
    return tone * env * vel


def odaiko(vel=1.0):
    t = tt(1.2)
    ph = 2 * np.pi * np.cumsum(55 + 70 * np.exp(-t / 0.04)) / SR
    body = np.sin(ph) * np.exp(-t / 0.35) + 0.3 * np.sin(1.58 * ph) * np.exp(-t / 0.12)
    nz = fft_filter(rng.standard_normal(len(t)), hi=900); nz /= nz.std()
    return (body + 0.45 * nz * np.exp(-t / 0.03)) * np.minimum(1, t / 0.001) * vel


def shime(vel=0.5):
    t = tt(0.35)
    ph = 2 * np.pi * np.cumsum(330 * (1 + 0.15 * np.exp(-t / 0.01))) / SR
    nz = fft_filter(rng.standard_normal(len(t)), lo=1500, hi=7000); nz /= nz.std()
    return (np.sin(ph) * np.exp(-t / 0.07) + 0.6 * nz * np.exp(-t / 0.012)) * vel


def kane(vel=0.5, open_=True):
    t = tt(0.6 if open_ else 0.15)
    f0 = 1650 * (1 + rng.uniform(-0.004, 0.004))
    out = np.zeros_like(t)
    for r, a in zip([1, 1.47, 2.08, 2.71, 3.36, 4.18, 5.02], [1, .75, .6, .45, .3, .22, .15]):
        d = (0.22 if open_ else 0.035) / (1 + 0.25 * (r - 1))
        out += a * np.sin(2 * np.pi * f0 * r * t + rng.uniform(0, 6.28)) * np.exp(-t / d)
    nz = fft_filter(rng.standard_normal(len(t)), lo=3000); nz /= nz.std()
    return (out + 0.5 * nz * np.exp(-t / 0.002)) * np.minimum(1, t / 0.0005) * vel


def furin(f0, vel=0.3):
    t = tt(6.0); out = np.zeros_like(t)
    for r, d, a in zip([1, 2.76, 5.40, 8.93], [1.8, 1.0, 0.5, 0.25], [1, .5, .3, .15]):
        for dt in (1.0, 1.0012):
            out += a * np.sin(2 * np.pi * f0 * r * dt * t) * np.exp(-t / d)
    return out * np.minimum(1, t / 0.0008) * vel * 0.5


def ping(f, d, vel):
    t = tt(d * 5)
    return np.sin(2 * np.pi * f * t) * np.exp(-t / d) * np.minimum(1, t / 0.0008) * vel


def click(vel):
    k = 96
    return fft_filter(rng.standard_normal(k), lo=1500, hi=9000) * np.exp(-np.arange(k) / 20) * vel


def boom(size=1.0):
    t = tt(1.6)
    ph = 2 * np.pi * np.cumsum(46 + 40 * np.exp(-t / 0.04)) / SR
    nz = fft_filter(rng.standard_normal(len(t)), hi=350); nz /= nz.std()
    return (np.sin(ph) * np.exp(-t / 0.40) + 0.5 * nz * np.exp(-t / 0.12)) * np.minimum(1, t / 0.004) * size


def spark(t0, kind, size=1.0):
    if kind == 'peony':
        ts = t0 + np.sort(rng.exponential(0.25, int(10 + 16 * size)))
        for ti in ts[ts < t0 + 1.4]:
            fall = np.exp(-(ti - t0) / 0.6)
            place('spark', ping(rng.uniform(2800, 8000), rng.uniform(0.03, 0.10),
                                rng.uniform(.3, 1) * size * fall), ti, pan=rng.uniform(-.8, .8))
    elif kind == 'crackle':
        ts = t0 + 0.12 + np.sort(rng.exponential(0.35, int(70 * size)))
        for ti in ts[ts < t0 + 1.6]:
            place('spark', click(rng.uniform(.4, 1) * size * 1.6), ti, pan=rng.uniform(-.9, .9))
    elif kind == 'kamuro':
        L = 2.8
        for _ in range(int(45 * size)):
            x = rng.random() ** 1.6
            place('spark', ping(rng.uniform(2200, 5200) * (1 - 0.15 * x), rng.uniform(0.12, 0.3),
                                rng.uniform(.2, .6) * size * (1 - 0.8 * x)), t0 + L * x, pan=rng.uniform(-.85, .85))
        t = tt(L)
        nz = fft_filter(rng.standard_normal((2, len(t))), lo=3000, hi=9000)
        nz /= nz.std()
        env = np.minimum(1, t / 0.25) * np.exp(-t / 1.1) * 0.12 * size
        place('spark', nz * env, t0)
    elif kind == 'senrin':
        for _ in range(int(18 * size)):
            ti = t0 + 0.1 + rng.uniform(0, 0.9)
            tl = tt(0.2)
            place('spark', np.sin(2 * np.pi * rng.uniform(180, 400) * tl) * np.exp(-tl / 0.04) * 0.5 * size,
                  ti, pan=rng.uniform(-.8, .8))
            for j in range(4):
                place('spark', click(rng.uniform(.5, 1) * size * 1.4), ti + j * rng.uniform(0.005, 0.03),
                      pan=rng.uniform(-.9, .9))


# ---------------------------------------------------------------- score
CH = {
    'Ebmaj7': ([58, 62, 63, 67], 39),
    'F':      ([57, 60, 65, 67], 41),
    'Dm7':    ([57, 60, 62, 65], 38),
    'Gm7':    ([58, 62, 65, 67], 43),
    'Cm7':    ([58, 60, 63, 67], 36),
    'Fsus4':  ([58, 60, 65, 67], 41),
    'F/A':    ([57, 60, 65, 69], 45),
    'Bbadd9': ([58, 60, 62, 65, 70], 34),
}
PROG = [(-3, 5, 'Ebmaj7'), (2, 2, 'F'), (4, 2, 'Dm7'), (6, 2, 'Gm7'), (8, 2, 'Cm7'),
        (10, 1, 'Fsus4'), (11, 1, 'F'), (12, 2, 'Ebmaj7'), (14, 1, 'F/A'), (15, 1, 'F'),
        (16, 2, 'Ebmaj7'), (18, 2, 'F'), (20, 2, 'Dm7'), (22, 2, 'Gm7'),
        (24, 1, 'Cm7'), (25, 1, 'F'), (26, 3.3, 'Bbadd9')]


def section_level(b):
    return 0.55 if b < 0 else 0.7 if b < 12 else 0.85 if b < 16 else 1.0


# pad
for s, d, c in PROG:
    place('pad', pad(CH[c][0], d * BEAT + 0.08, attack=1.4 if s < 0 else 0.3), tb(s), gain=section_level(s))

# bass
for s, d, c in PROG:
    root = CH[c][1]
    if s < 0:
        place('bass', bass(root, 2 * BEAT * 0.95, .5), tb(0))
    elif s < 12:
        place('bass', bass(root, d * BEAT * 0.95, .5), tb(s))
    elif s < 16:
        for k in range(int(d * 2)):
            place('bass', bass(root, BEAT * 0.45, .45 + 0.04 * (s - 12 + k / 2)), tb(s + k / 2))
    elif s < 24:
        place('bass', bass(root, BEAT * 1.25, .62), tb(s))
        place('bass', bass(root, BEAT * 0.45, .5), tb(s + 1.5))
    else:
        place('bass', bass(root, d * BEAT * 0.9, .62), tb(s))


def pan_of(m):
    return float(np.clip((m - 64) / 40, -0.6, 0.6))


# piano: intro pickup
for b, m in [(-2, 70), (-1.5, 72), (-1, 74), (-0.5, 77)]:
    place('piano', piano(m, BEAT * 1.5, .32), tb(b), pan=pan_of(m))
# piano: verse arpeggios
for s, d, c in PROG:
    v = CH[c][0]
    if 0 <= s < 12 or s < 0:
        s0 = max(s, 0)
        steps = int(((s + d) - s0) * 2)
        pat = [0, 1, 2, 3] if steps >= 4 else [0, 2]
        for k in range(steps):
            m = v[pat[k % len(pat)]]
            place('piano', piano(m, BEAT * 1.5, .40 if k == 0 else .32), tb(s0 + k / 2), pan=pan_of(m))
# piano: build (quarter block chords, crescendo)
for b in [12, 13, 14, 15, 15.5]:
    c = 'Ebmaj7' if b < 14 else 'F/A' if b < 15 else 'F'
    for m in CH[c][0]:
        place('piano', piano(m, BEAT * 0.9, .30 + 0.07 * (b - 12)), tb(b), pan=pan_of(m))
# piano: chorus (3-3-2 push)
for s, d, c in PROG:
    if 16 <= s < 24:
        for off, dd, vel in [(0, 1.4, .42), (1.5, .5, .32)]:
            for m in CH[c][0] + [CH[c][1] + 24]:
                place('piano', piano(m, BEAT * dd, vel), tb(s + off), pan=pan_of(m))
# piano: cadence + rolled final chord
for b, c in [(24, 'Cm7'), (25, 'F')]:
    for m in CH[c][0] + [CH[c][1] + 24]:
        place('piano', piano(m, BEAT * 0.95, .5), tb(b), pan=pan_of(m))
for i, m in enumerate([46, 53, 58, 60, 62, 65, 70, 74]):
    place('piano', piano(m, BEAT * 3.0, .45), tb(26) + i * 0.05, pan=pan_of(m))

# koto: verse melody
MEL_VERSE = [(0.5, 74, .5), (1.0, 75, .5), (1.5, 77, 1.0), (2.5, 72, .5), (3.0, 77, .75), (3.75, 79, .25),
             (4.0, 81, 1.0), (5.0, 79, .5), (5.5, 77, .5), (6.0, 74, 1.5), (7.5, 70, .5),
             (8.0, 72, .5), (8.5, 74, .5), (9.0, 75, 1.0), (10.0, 74, .5), (10.5, 72, .5), (11.0, 72, 1.0)]
for b, m, d in MEL_VERSE:
    place('koto', koto(m, .55), tb(b), pan=-0.3)
# build: rising line, koto + fue crescendo
RISE = [70, 72, 74, 77, 79, 81, 82, 84]
for k, m in enumerate(RISE):
    b = 12 + k / 2
    place('koto', koto(m, .45 + 0.03 * k), tb(b), pan=-0.3)
    place('fue', fue(m, BEAT * 0.48, .22 + 0.045 * k, scoop=0.3), tb(b), pan=0.1)
# chorus melody: fue lead, koto an octave below
MEL_CH = [(16, 86, 1.0), (17, 84, .5), (17.5, 82, .5), (18, 84, 1.0), (19, 82, .5), (19.5, 79, .5),
          (20, 81, 1.0), (21, 82, .5), (21.5, 81, .5), (22, 79, .5), (22.5, 86, 1.5),
          (24, 84, .5), (24.5, 82, .5), (25, 81, 1.0)]
for b, m, d in MEL_CH:
    acc = 1.25 if b == 22.5 else 1.0
    place('fue', fue(m, BEAT * d * 0.95, .55 * acc, swell=0.15 if d >= 1 else 0), tb(b), pan=0.1)
    place('koto', koto(m - 12, .40 * acc), tb(b), pan=-0.3)
place('fue', fue(82, BEAT * 2.9, .55, decay=1.3), tb(26), pan=0.1)
# ending glitter (falling like gold willow)
for b, m, v in zip([26.5, 26.9, 27.35, 27.85, 28.4], [94, 91, 89, 86, 82], [.35, .3, .27, .23, .2]):
    place('koto', koto(m, v), tb(b), pan=float(rng.uniform(-.5, .5)))

# percussion
for s in (0, 4, 8):
    place('odaiko', odaiko(1.0 if s == 0 else .8), tb(s), pan=-0.1)
    place('odaiko', odaiko(.5), tb(s + 2), pan=-0.1)
    for o in (1, 3):
        place('shime', shime(.35), tb(s + o), pan=0.3)
place('shime', shime(.3), tb(11.5), pan=0.3)
for b in (9, 11):
    place('kane', kane(.3), tb(b), pan=0.45)
for k in range(4):
    place('shime', shime(.35 + 0.04 * k), tb(12 + k / 2), pan=0.3)
for k in range(8):
    place('shime', shime(.45 + 0.05 * k), tb(14 + k / 4), pan=0.3)
for b, v in [(12, .9), (14, .8), (15, .85), (15.5, .9), (15.75, .95)]:
    place('odaiko', odaiko(v), tb(b), pan=-0.1)
for s in (16, 20):
    for o, v in [(0, 1.0), (1.5, .6), (2, .85), (3, .7)]:
        place('odaiko', odaiko(v), tb(s + o), pan=-0.1)
    for k in range(8):
        place('shime', shime(.65 if k in (2, 6) else .45), tb(s + k / 2), pan=0.3)
    for b in range(4):
        place('kane', kane(.45), tb(s + b), pan=0.45)
        place('kane', kane(.30, False), tb(s + b + 0.5), pan=0.45)
        place('kane', kane(.25, False), tb(s + b + 0.75), pan=0.45)
for b, v in [(23.5, .8), (23.75, .9), (24, 1.0), (25, .8), (26, 1.1)]:
    place('odaiko', odaiko(v), tb(b), pan=-0.1)
for b, v in [(25.5, .6), (25.75, .75)]:
    place('shime', shime(v), tb(b), pan=0.3)
for b, v in [(24, .4), (25, .35), (26, .5)]:
    place('kane', kane(v), tb(b), pan=0.45)

# summer-night furin and launch whistle in the intro
for t, f0, v in [(0.10, 2650, .30), (0.85, 3100, .22), (8.95, 2800, .25), (18.55, 2900, .30), (19.25, 2500, .25)]:
    place('furin', furin(f0, v), t, pan=float(rng.uniform(-.6, .6)))
place('whistle', whistle(1.86 - 0.60, 72, 93), 0.60, pan=0.05)

# firework bursts, synced to the video (seconds)
BURSTS = [(1.90, 'peony', 1.0, 1.0), (4.80, 'peony', .8, .8), (6.50, 'peony', 1.0, 1.0),
          (8.73, 'peony', .7, .6), (10.13, 'peony', 1.0, 1.0), (10.73, 'crackle', 1.0, .7),
          (11.20, 'peony', .40, .30), (11.33, 'peony', .40, .30), (11.47, 'peony', .50, .35),
          (11.60, 'peony', .35, .25), (11.73, 'peony', .45, .30), (11.87, 'peony', .35, .25),
          (12.00, 'peony', .50, .35), (12.13, 'peony', .35, .25), (12.27, 'peony', .40, .30),
          (13.90, 'kamuro', 1.0, 1.0), (14.50, 'peony', .7, .6), (14.73, 'peony', .6, .5),
          (15.90, 'kamuro', 1.2, 1.1), (16.10, 'kamuro', .6, .5), (17.70, 'senrin', 1.0, .7),
          (18.33, 'crackle', .5, 0.0)]
for t, kind, size, bs in BURSTS:
    spark(t, kind, size)
    if bs > 0:
        place('boom', boom(bs), t, pan=float(rng.uniform(-.3, .3)))


# ---------------------------------------------------------------- mix
def reverb(x2, T=2.6):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = rng.standard_normal(n)
        w = np.exp(-t / 0.35)
        ir = (fft_filter(nz, hi=7000) * w + fft_filter(nz, hi=2200) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.022 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


STEMS['pad'] = fft_filter(fft_filter(STEMS['pad'].astype(float), hi=1600), lo=120).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, T0), verse=(T0, tb(12)), build=(tb(12), tb(16)), chorus=(tb(16), tb(24)), end=(tb(24), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.30 * reverb(snd)
mix = fft_filter(mix, lo=30)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.1, 1.0, np.cos(np.clip((t - 19.1) / 0.9, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.01)
mix /= np.abs(mix).max()
drive = 1.2
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
