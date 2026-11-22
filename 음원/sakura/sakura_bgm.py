"""Original BGM for sakura.mp4 (20 s): night cherry blossoms under a full moon.

Lively but wistful: 6/8, E minor / G major. Running piano arpeggios (petals
tumbling), breathy shakuhachi-like flute lead, string swells, harp glissandi at
each gust, koto plinks for petals landing on the path, light frame drum + shaker.
Bar length is set so the three gusts in the video (3.2 s, 8.7 s, 16.0 s) fall
on bar lines; the last chord lands at 17.83 s.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261120)

BAR = (16.0 - 3.2) / 7          # 1.8286 s per 6/8 bar
E8 = BAR / 6                    # eighth note
K0 = 3.2                        # bar k=0 starts at the first gust


def tk(k, e=0.0):               # bar index k (can be -1), eighth position e
    return K0 + k * BAR + e * E8


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
GAIN = dict(piano=0.34, flute=0.30, strings=0.024, violins=0.050, harp=0.20, koto=0.27,
            glock=0.17, drum=0.22, shaker=0.50, swell=0.030, bass=0.20)
SEND = dict(piano=.30, flute=.40, strings=.45, violins=.42, harp=.40, koto=.40,
            glock=.60, drum=.20, shaker=.15, swell=.45, bass=.05)


def place(name, sig, t, pan=0.0, gain=1.0):
    buf = STEMS.setdefault(name, np.zeros((2, N), np.float32))
    i = int(round(t * SR))
    if sig.ndim == 1:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * np.sqrt(2)
    else:
        st = sig.copy()
    k = min(st.shape[1], max(int(0.02 * SR), int(0.05 * st.shape[1])))
    st[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    if i < 0:
        st, i = st[:, -i:], 0
    n = min(st.shape[1], N - i)
    if n > 0:
        buf[:, i:i + n] += (st[:, :n] * gain).astype(np.float32)


# ---------------------------------------------------------------- instruments
def piano(m, dur, vel=0.5):
    f = float(mtof(m)); T = dur + 0.9; t = tt(T)
    t60 = float(np.interp(m, [36, 60, 84, 100], [7.0, 4.0, 1.8, 0.8]))
    out = np.zeros_like(t)
    for n in range(1, 14):
        fn = f * n * np.sqrt(1 + 0.0004 * n * n)
        if fn > 14000:
            break
        a = (1.0 / n ** 1.2) * ((0.30 + 0.55 * vel) ** (n - 1))
        tau = t60 / 6.9 / (1 + 0.45 * (n - 1))
        out += a * np.sin(2 * np.pi * fn * t) * (0.65 * np.exp(-t / (tau * 0.35)) + 0.35 * np.exp(-t / tau))
    out *= np.minimum(1, t / 0.003) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.12))
    k = int(0.012 * SR)
    out[:k] += fft_filter(noise(k) * np.exp(-np.arange(k) / SR / 0.003), lo=1000, hi=5000) * 0.10 * vel
    return out * vel


def koto(m, vel=0.5):
    f = float(mtof(m)); T = float(np.interp(m, [50, 74, 98], [3.0, 2.0, 1.1])); t = tt(T)
    ph = 2 * np.pi * f * np.cumsum(1 + 0.007 * np.exp(-t / 0.025)) / SR
    out = np.zeros_like(t)
    for n in range(1, 16):
        if f * n > 14000:
            break
        out += abs(np.sin(n * np.pi * 0.2)) / n ** 1.25 * np.sin(n * ph) * np.exp(-t * (1.3 + 0.8 * n) * (f / 440) ** 0.35)
    return out * np.minimum(1, t / 0.001) * vel


def harp(m, vel=0.4):
    f = float(mtof(m)); T = float(np.interp(m, [40, 60, 90], [4.0, 3.0, 1.4])); t = tt(T)
    out = np.zeros_like(t)
    for n in range(1, 11):
        if f * n > 12000:
            break
        out += abs(np.sin(n * np.pi * 0.3)) / n ** 1.5 * np.sin(2 * np.pi * f * n * t) * np.exp(-t * (0.9 + 0.6 * n) * (f / 440) ** 0.3)
    return out * np.minimum(1, t / 0.002) * vel


def glock(m, vel=0.4):
    f = float(mtof(m)); t = tt(3.0); out = np.zeros_like(t)
    for r, d, a in zip([1, 2.76, 5.40], [1.5, 0.35, 0.12], [1, .3, .12]):
        if f * r < 16000:
            out += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d)
    return out * np.minimum(1, t / 0.0008) * vel


def shaku(m, dur, vel=0.5, meri=0.0, swell=0.0, decay=None):
    """Breathy, wistful bamboo-flute lead (shakuhachi-like): slow attack, air, meri bend."""
    f = float(mtof(m)); T = dur + 0.45; t = tt(T)
    cents = -100 * meri * np.exp(-t / 0.11) - 25 * np.exp(-t / 0.04)
    vib = np.clip((t - 0.3) / 0.45, 0, 1) * np.sin(2 * np.pi * 4.6 * t) * 20
    ph = 2 * np.pi * np.cumsum(f * 2 ** ((cents + vib) / 1200)) / SR
    tone = np.sin(ph) + 0.18 * np.sin(2 * ph + 0.5) + 0.06 * np.sin(3 * ph + 1.0)
    env = np.minimum(1, t / 0.08) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.1))
    env *= 1 + swell * np.clip(t / max(dur, 1e-3), 0, 1)
    if decay:
        env *= np.exp(-t / decay)
    nz = fft_filter(noise(len(t)), lo=f * 0.8, hi=f * 5); nz /= nz.std() + 1e-9
    air = 0.07 + 0.30 * np.exp(-t / 0.06)          # breath mostly at the onset, little steady hiss
    return (tone + air * nz) * env * vel


def pad(notes, dur, attack=0.6):
    T = dur + 2.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-8, 0.85), (0, 0.5), (8, 0.15)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = np.zeros_like(t)
            for k in range(1, 9):
                if f * k > 9000:
                    break
                s += np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28)) / k
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.45))
    return np.vstack([L * env, R * env])


def violins(m, dur, vel=0.6, attack=0.25):
    T = dur + 0.9; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for v in range(5):
        det = rng.normal(0, 6)
        vib = 12 * np.sin(2 * np.pi * rng.uniform(5.0, 6.0) * t + rng.uniform(0, 6.28)) * np.clip(t / 0.4, 0, 1)
        ph = 2 * np.pi * np.cumsum(float(mtof(m)) * 2 ** ((det + vib) / 1200)) / SR
        s = np.zeros_like(t)
        for k in range(1, 12):
            if float(mtof(m)) * k > 9000:
                break
            s += np.sin(k * ph) / k
        w = (v + 0.5) / 5
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.3))
    return np.vstack([L * env, R * env]) * vel


def bassnote(m, dur, vel=0.5):
    f = float(mtof(m)); T = dur + 0.5; t = tt(T)
    ph = 2 * np.pi * f * t
    tone = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    env = np.minimum(1, t / 0.03) * np.exp(-t / 2.5) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.15))
    return tone * env * vel


def drum(vel=0.6):
    """Soft frame drum / small taiko."""
    t = tt(0.9)
    ph = 2 * np.pi * np.cumsum(85 * (1 + 0.5 * np.exp(-t / 0.02))) / SR
    nz = fft_filter(noise(len(t)), hi=1200); nz /= nz.std()
    return (np.sin(ph) * np.exp(-t / 0.28) + 0.3 * nz * np.exp(-t / 0.02)) * np.minimum(1, t / 0.002) * vel


def shaker(vel=0.4):
    t = tt(0.07)
    nz = fft_filter(noise(len(t)), lo=5000, hi=12000); nz /= nz.std()
    nz = np.tanh(0.6 * nz) / 0.6                       # tame noise spikes
    return nz * np.minimum(1, t / 0.006) * np.exp(-t / 0.02) * vel


def swell(notes, T, vel=0.5):
    """Tonal string crescendo into a gust (replaces the old noise swell)."""
    x = pad(notes, T, attack=0.05)[:, :int(T * SR)]
    t = tt(x.shape[1] / SR)
    return x * ((t / T) ** 2.2) * vel


# ---------------------------------------------------------------- harmony (bar k, half 0/1, chord)
CH = {
    'Cmaj9':  [48, 55, 59, 62, 64],
    'Em9':    [40, 52, 59, 62, 66, 67],
    'Cmaj7':  [48, 55, 59, 64],
    'Dadd9':  [50, 57, 62, 64, 66],
    'Am7':    [45, 52, 57, 60, 67],
    'B7':     [47, 54, 57, 63, 66],
    'D6':     [50, 57, 62, 66, 71],
    'Bm7':    [47, 54, 57, 62, 66],
    'Em7':    [40, 52, 59, 62, 67],
    'Am9':    [45, 52, 59, 60, 64],
    'D':      [50, 57, 62, 66, 69],
    'Bsus4':  [47, 54, 59, 64, 66],
    'B7b9':   [47, 54, 57, 60, 63],
    'Em_add9': [40, 52, 55, 59, 64, 66, 67, 71],
}
PROG = [(-1, 0, 2, 'Cmaj9'), (0, 0, 2, 'Em9'), (1, 0, 1, 'Cmaj7'), (1, 1, 1, 'Dadd9'),
        (2, 0, 1, 'Am7'), (2, 1, 1, 'B7'),
        (3, 0, 1, 'Cmaj7'), (3, 1, 1, 'D6'), (4, 0, 1, 'Bm7'), (4, 1, 1, 'Em7'),
        (5, 0, 1, 'Am9'), (5, 1, 1, 'D'), (6, 0, 1, 'Bsus4'), (6, 1, 1, 'B7b9'),
        (7, 0, 1, 'Cmaj9'), (7, 1, 1, 'Dadd9'), (8, 0, 2, 'Em_add9')]


def sect_lvl(k):
    return 0.55 if k < 0 else 0.7 if k < 3 else 1.0 if k < 7 else 0.95


# piano arpeggios: 8ths in the intro/A section, 16ths (running) from the first big gust
for k, h, ln, c in PROG:
    v = CH[c]
    if c == 'Em_add9':
        for i, m in enumerate(v):                      # rolled final chord
            place('piano', piano(m, 2.4, .42), tk(8) + i * 0.045, pan=float(np.clip((m - 60) / 30, -.5, .5)))
        continue
    body = v[1:] + [n + 12 for n in v[1:3]]
    seq = [v[0]] + body + body[-2:0:-1]
    step = 0.5 if k >= 3 else 1.0
    n_steps = int(ln * 3 / step)
    for j in range(n_steps):
        m = seq[j % len(seq)]
        vel = (0.46 if j == 0 else 0.34 + 0.06 * ((j * step) % 3 == 0)) * (0.75 + 0.25 * sect_lvl(k))
        place('piano', piano(m, E8 * step * 2.2, vel), tk(k, h * 3 + j * step), pan=float(np.clip((m - 60) / 30, -.5, .5)))
# intro pickup (rising Em arpeggio into the intro bar)
for i, m in enumerate([59, 64, 67, 71]):
    place('piano', piano(m, E8 * 2, .32 + .03 * i), tk(-1) - (4 - i) * E8, pan=float((m - 62) / 30))

# bass (piano left hand / soft low strings) on the dotted-quarter pulse
for k, h, ln, c in PROG:
    root = CH[c][0]
    if k < 0:
        continue
    place('bass', bassnote(root - 12 if root >= 47 else root, ln * 3 * E8 * 0.95 if k < 8 else 2.3,
                           .55 * sect_lvl(k)), tk(k, h * 3))

# strings bed from bar 1, fuller in the chorus; violins double the lead an octave below in the chorus
for k, h, ln, c in PROG:
    if k >= 1:
        v = sorted(set(n + 12 for n in CH[c][1:5]))
        place('strings', pad(v, ln * 3 * E8 + 0.05 if k < 8 else 2.3, attack=1.0 if k == 1 else 0.35),
              tk(k, h * 3), gain=0.55 if k < 3 else 1.0)

# lead melody: (bar k, eighth pos, length in eighths, midi)
MEL = [(0, 0, 2, 76), (0, 2, 1, 74), (0, 3, 2, 71), (0, 5, 1, 74),
       (1, 0, 2, 71), (1, 2, 1, 67), (1, 3, 2, 69), (1, 5, 1, 66),
       (2, 0, 1.5, 67), (2, 1.5, .5, 69), (2, 2, 1, 72), (2, 3, 2, 71), (2, 5, 1, 75),
       (3, 0, 1.5, 76), (3, 1.5, .5, 79), (3, 2, 1, 83), (3, 3, 1.5, 81), (3, 4.5, .5, 78), (3, 5, 1, 76),
       (4, 0, 2, 78), (4, 2, 1, 74), (4, 3, 1.5, 79), (4, 4.5, .5, 78), (4, 5, 1, 76),
       (5, 0, 1, 72), (5, 1, 1, 76), (5, 2, 1, 83), (5, 3, 2, 81), (5, 5, 1, 78),
       (6, 0, 2, 76), (6, 2, 1, 78), (6, 3, 1.5, 75), (6, 4.5, .5, 78), (6, 5, 1, 84),
       (7, 0, 2, 83), (7, 2, 1, 79), (7, 3, 1.5, 81), (7, 4.5, .5, 78), (7, 5, 1, 76),
       (8, 0, 1, 78), (8, 1, 6.2, 76)]
for k, e, ln, m in MEL:
    last = (k, e) == (8, 1)
    vel = (0.50 if k < 3 else 0.60) * (1.12 if (k, e) in [(3, 2), (5, 2), (6, 5)] else 1.0)
    place('flute', shaku(m, ln * E8 * (0.96 if ln < 2 else 0.98), vel, meri=0.5 if ln >= 2 and k < 3 else 0.15,
                         swell=0.15 if ln >= 2 else 0, decay=1.5 if last else None), tk(k, e), pan=0.05)
    if 3 <= k <= 7:
        place('violins', violins(m - 12, ln * E8 * 0.98, .55, attack=0.08 if ln < 1.5 else 0.2), tk(k, e))
place('violins', violins(64, 2.3, .45, attack=0.4), tk(8, 1))

# gusts: harp glissando up + string swell + noise swell before the bar
GUSTS = [(3.2, 4.83, 0.7), (8.7, 12.13, 1.0), (16.0, 17.63, 0.85)]
GL_NOTES = [52, 55, 57, 59, 62, 64, 67, 69, 71, 74, 76, 79, 81, 83, 86, 88, 91]   # E minor pentatonic
for g0, g1, a in GUSTS:
    for i, m in enumerate(GL_NOTES):
        place('harp', harp(m, (0.28 + 0.2 * i / len(GL_NOTES)) * a), g0 - 0.02 + i * 0.032,
              pan=float(-0.6 + 1.2 * i / len(GL_NOTES)))
    pre = [c for kk, hh, ll, c in PROG if tk(kk, hh * 3) < g0 - 0.01][-1]       # chord sounding before the gust
    T = 1.0 + 0.5 * a
    place('swell', swell(sorted(set(n + 12 for n in CH[pre][1:5])), T, a), g0 - T)
    place('drum', drum(0.9 * a), g0, pan=-0.05)
# a falling harp sweep as the big gust dies down
for i, m in enumerate(GL_NOTES[::-1][:12]):
    place('harp', harp(m, 0.30 - 0.015 * i), 12.15 + i * 0.05, pan=float(0.6 - 1.2 * i / 12))

# moon: glock chimes in the intro and at the end
for t, m, v in [(0.10, 88, .32), (0.75, 95, .22), (17.90, 95, .28), (18.40, 91, .22), (18.95, 88, .18)]:
    place('glock', glock(m, v), t, pan=float(rng.uniform(-.5, .5)))

# petals landing on the path: sparse koto plinks, more of them as the path fills up
PENTA_HI = [76, 79, 81, 83, 86, 88, 91]
for _ in range(34):
    t = 2.0 + 17.0 * rng.random() ** 0.7
    place('koto', koto(int(rng.choice(PENTA_HI)), rng.uniform(.12, .28)), t, pan=float(rng.uniform(-.8, .8)))

# rhythm: shaker 8ths from bar 1, soft frame drum on the dotted quarters in the chorus and coda
for k in range(1, 8):
    for e in range(6):
        place('shaker', shaker([.5, .22, .3, .42, .22, .3][e] * (0.7 if k < 3 else 1.0)), tk(k, e), pan=0.35)
    if k >= 3:
        for e, v in [(0, .7), (3, .45), (5, .25)]:
            place('drum', drum(v), tk(k, e), pan=-0.05)
place('drum', drum(.75), tk(8), pan=-0.05)


# ---------------------------------------------------------------- mix
def reverb(x2, T=2.8):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.4)
        ir = (fft_filter(nz, hi=8000) * w + fft_filter(nz, hi=3000) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.025 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


STEMS['strings'] = fft_filter(fft_filter(STEMS['strings'].astype(float), hi=3000), lo=180).astype(np.float32)
STEMS['violins'] = fft_filter(fft_filter(STEMS['violins'].astype(float), hi=4500), lo=250).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, 3.2), A=(3.2, 8.7), B=(8.7, 16.0), C=(16.0, tk(8)), end=(tk(8), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.34 * reverb(snd)
mix = fft_filter(mix, lo=35)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.0, 1.0, np.cos(np.clip((t - 19.0) / 1.0, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.05)
mix /= np.abs(mix).max()
drive = 1.2
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
