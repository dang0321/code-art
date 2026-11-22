"""Original BGM for anemone and clownfish.mp4 (20 s): a glowing sea anemone and a clownfish
playing hide-and-seek in it.

A bouncy undersea calypso in F major at 120 BPM, in the spirit of a cheerful "under the sea"
number (original melody). The tune itself is in the bass: a plucked upright bass (modelled to
sound like the real wooden instrument - plucked partials, a little pitch "boing" on the attack, a
wooden body resonance and finger thump) doubled an octave higher by a soft wooden marimba. Around
it: steel pan off-beat chords and rolls, nylon-guitar chops, bongos, a soft shaker and claves.
Bubble "bloops" are tuned water drops. No synth leads, no noise beds.

Sync (the fish, tracked by its orange pixels): it swims across the front from 0 s, dives behind
the anemone at ~4.3 s (a diving bass run and rising bubbles), stays hidden 6.3-10 s (a swaying,
lighter B section with steel-pan rolls), peeks out at 10.0 s (an octave "boing" and a bongo pop
on the bar downbeat) and is back at the same spot at 12.0 s, where the theme returns. Its loop
takes 12 s = 6 bars, so the theme lines up with the same picture both times. It hides again at
~16 s (a second dive), and the tune ends with a "ta-da" on F at 19.0 s.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261125)

BEAT = 0.5                        # 120 BPM
BAR = 4 * BEAT
DIVE1, PEEK, BACK, DIVE2, END = 4.3, 10.0, 12.0, 16.0, 19.0
SW = 0.04                         # a touch of lilt on the off-8ths


def tb(bar, pos=0.0):
    frac = pos - np.floor(pos)
    if abs(frac - 0.5) < 1e-6:
        pos += SW
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


def noise(n):
    return rng.standard_normal(n)


STEMS = {}
GAIN = dict(bass=0.62, marimba=0.26, pan=0.24, panroll=0.14, gtr=0.17, bongo=0.24, shaker=0.13, clave=0.22,
            kick=0.24, bubble=0.24, peek=0.28)
SEND = dict(bass=.08, marimba=.20, pan=.30, panroll=.40, gtr=.18, bongo=.15, shaker=.10, clave=.18,
            kick=.04, bubble=.40, peek=.25)


def place(name, sig, t, pan=0.0, gain=1.0):
    buf = STEMS.setdefault(name, np.zeros((2, N), np.float32))
    i = int(round(t * SR))
    if sig.ndim == 1:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * np.sqrt(2)
    else:
        st = sig.copy()
    k = min(st.shape[1], max(int(0.006 * SR), int(0.04 * st.shape[1])))
    st[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    if i < 0:
        st, i = st[:, -i:], 0
    n = min(st.shape[1], N - i)
    if n > 0:
        buf[:, i:i + n] += (st[:, :n] * gain).astype(np.float32)


# ---------------------------------------------------------------- instruments
BODY = [(98.0, 0.020, 0.9), (182.0, 0.016, 0.6), (290.0, 0.010, 0.35)]   # wooden body modes (Hz, tau, gain)


def upright(m, dur, vel=0.7):
    """Plucked upright bass: string partials (plucked near the end of the fingerboard), higher ones
    dying faster, a small pitch drop on the attack, a wooden body resonance and a finger thump."""
    f = float(mtof(m)); T = dur + 0.35; t = tt(T)
    bend = 2 ** ((14 * np.exp(-t / 0.035)) / 1200)                 # +14 cents settling in 35 ms
    ph = 2 * np.pi * np.cumsum(f * bend) / SR
    tau0 = float(np.interp(m, [36, 48, 60], [1.1, 0.8, 0.55]))
    out = np.zeros_like(t)
    for n in range(1, 16):
        if f * n > 4000:
            break
        a = abs(np.sin(n * np.pi * 0.17)) / n ** 0.9 * (0.55 + 0.45 * vel) ** (n - 1)
        out += a * np.sin(n * ph * (1 + 0.0004 * n)) * np.exp(-t / (tau0 / (1 + 0.85 * (n - 1))))
    # wooden body: short resonances excited by the pluck
    for fb, tb_, g in BODY:
        out += g * 0.35 * np.sin(2 * np.pi * fb * t) * np.exp(-t / tb_)
    thump = fft_filter(noise(int(0.02 * SR)), hi=900) * np.exp(-tt(0.02) / 0.004)
    out[:len(thump)] += 0.12 * vel * thump / (np.abs(thump).max() + 1e-9)
    env = np.sin(np.minimum(1, t / 0.003) * np.pi / 2) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.06))
    return np.tanh(1.15 * out) * env * vel


def marimba(m, vel=0.6, hard=0.4):
    f = float(mtof(m))
    tau = float(np.interp(m, [43, 60, 72, 84, 96], [1.2, 0.75, 0.46, 0.28, 0.17]))
    T = tau * 5 + 0.05; t = tt(T)
    out = np.sin(2 * np.pi * f * t) * (0.7 * np.exp(-t / tau) + 0.3 * np.exp(-t / (tau * 0.25)))
    for r, a, k in [(3.98, 0.18 + 0.25 * vel * hard, 0.16), (9.85, 0.03 + 0.08 * vel * hard, 0.05)]:
        if f * r < 12000:
            out += a * np.sin(2 * np.pi * f * r * t + rng.uniform(0, 6)) * np.exp(-t / (tau * k))
    att = 0.0015 + 0.002 * (1 - hard)
    return out * np.sin(np.minimum(1, t / att) * np.pi / 2) * vel


def steelpan(m, vel=0.5, T=1.4):
    """Steel pan: near-harmonic partials (octave and twelfth tuned), the octave blooming a moment
    after the strike, a slight shimmer from two neighbouring note areas."""
    f = float(mtof(m)); t = tt(T)
    s = float(np.interp(m, [55, 72, 88], [1.2, 0.85, 0.5]))
    out = np.sin(2 * np.pi * f * t) * np.exp(-t / (0.9 * s))
    out += 0.5 * (1 - np.exp(-t / 0.018)) * np.sin(2 * np.pi * 2.0 * f * t) * np.exp(-t / (0.45 * s))
    out += 0.18 * np.sin(2 * np.pi * 2.0 * 1.003 * f * t) * np.exp(-t / (0.35 * s))
    out += 0.16 * np.sin(2 * np.pi * 3.0 * f * t) * np.exp(-t / (0.22 * s))
    out += 0.05 * np.sin(2 * np.pi * 4.02 * f * t) * np.exp(-t / (0.12 * s))
    return out * np.sin(np.minimum(1, t / 0.003) * np.pi / 2) * vel


def nylon(m, dur, vel=0.4):
    f = float(mtof(m)); T = dur + 0.08; t = tt(T)
    out = np.zeros_like(t)
    for n in range(1, 12):
        if f * n > 5000:
            break
        out += abs(np.sin(n * np.pi * 0.27)) / n ** 1.2 * np.sin(2 * np.pi * f * n * t) * np.exp(-t / (0.9 / (1 + 0.5 * (n - 1))))
    env = np.sin(np.minimum(1, t / 0.003) * np.pi / 2) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.025))
    return out * env * vel


def bongo(hi=True, vel=0.6):
    f = 430.0 if hi else 305.0; t = tt(0.25)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.18 * np.exp(-t / 0.012))) / SR
    body = np.sin(ph) * np.exp(-t / 0.09) + 0.3 * np.sin(1.6 * ph) * np.exp(-t / 0.04)
    slap = fft_filter(noise(len(t)), lo=1500, hi=6000) * np.exp(-t / 0.003)
    return (body + 0.25 * slap / (np.abs(slap).max() + 1e-9)) * np.minimum(1, t / 0.0008) * vel


def shaker(vel=0.4):
    t = tt(0.07)
    nz = fft_filter(noise(len(t)), lo=4500, hi=10000); nz /= nz.std()
    return nz * np.sin(np.minimum(1, t / 0.012) * np.pi / 2) * np.exp(-t / 0.022) * vel


def clave(vel=0.5):
    t = tt(0.08)
    return np.sin(2 * np.pi * 2450 * t) * np.exp(-t / 0.018) * np.minimum(1, t / 0.0005) * vel


def kick(vel=0.8):
    t = tt(0.35)
    ph = 2 * np.pi * np.cumsum(58 + 60 * np.exp(-t / 0.025)) / SR
    return np.sin(ph) * np.exp(-t / 0.15) * np.minimum(1, t / 0.002) * vel


def bloop(m, vel=0.4, rise=1.0, decay=0.04):
    """A bubble: a sine whose pitch springs upward as it rings."""
    f0 = float(mtof(m)); t = tt(0.3)
    fr = f0 * 2 ** (rise * (1 - np.exp(-t / 0.014)))
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / decay) * np.minimum(1, t / 0.001) * vel


# ---------------------------------------------------------------- harmony (F major)
CH = {   # steel-pan / guitar voicing
    'F':      [65, 69, 72],
    'Bb':     [65, 70, 74],
    'C7':     [64, 67, 70],
    'D7':     [66, 69, 72],
    'Gm7':    [65, 70, 74],
    'Bbmaj7': [65, 69, 74],
    'Am7':    [64, 67, 72],
    'Fend':   [65, 69, 72, 77],
}
PROG = [(0, 0, 'F'), (1, 0, 'Bb'), (1, 2, 'C7'), (2, 0, 'F'), (2, 2, 'D7'),
        (3, 0, 'Bbmaj7'), (4, 0, 'Am7'), (4, 2, 'D7'), (5, 0, 'Gm7'), (5, 2, 'C7'),
        (6, 0, 'F'), (7, 0, 'Bb'), (7, 2, 'C7'), (8, 0, 'Am7'), (8, 2, 'D7'), (9, 0, 'Gm7'), (9, 1, 'C7'), (9, 2, 'Fend')]


def segs():
    pr = PROG
    for i, (bar, pos, c) in enumerate(pr):
        nb, npos = (pr[i + 1][0], pr[i + 1][1]) if i + 1 < len(pr) else (10, 0)
        yield bar, pos, c, (nb - bar) * 4 + (npos - pos)


# ---------------------------------------------------------------- the bass tune (bar, beat, beats, midi)
F2, G2, A2, Bb2, C3, D3, E3, F3, Fs3, G3, A3, Bb3, C4, D4 = 41, 43, 45, 46, 48, 50, 52, 53, 54, 55, 57, 58, 60, 62
HOOK_A = [(0, F2, .5), (.75, F3, .25), (1.5, C3, .5), (2, A2, .5), (2.5, C3, .25), (2.75, D3, .25), (3, E3, .5), (3.5, F3, .5)]
HOOK_B = [(0, Bb2, .5), (.75, Bb3, .25), (1.5, F3, .5), (2, C3, .5), (2.5, E3, .25), (2.75, G3, .25), (3, Bb3, .5), (3.5, A3, .25), (3.75, G3, .25)]
BASS = []
for bar, motif in ((0, HOOK_A), (1, HOOK_B), (6, HOOK_A), (7, HOOK_B)):
    BASS += [(bar, p, d, m) for p, m, d in motif]
BASS += [
    # bar 2: F, then D7 and the dive behind the anemone
    (2, 0, .5, F2), (2, .75, F3, .25), (2, 1.5, .5, C3), (2, 2, .5, D3), (2, 2.5, .25, Fs3), (2, 2.75, .25, A3),
    (2, 3, .25, C4), (2, 3.25, .25, A3), (2, 3.5, .25, Fs3), (2, 3.75, .25, D3),
    # hidden: a swaying B section
    (3, 0, 1.5, Bb2), (3, 1.5, .5, F3), (3, 2, 1, D3), (3, 3, .5, A2), (3, 3.5, .5, C3),
    (4, 0, 1.5, A2), (4, 1.5, .5, E3), (4, 2, 1, D3), (4, 3, .5, Fs3), (4, 3.5, .5, A3),
    # the peek (10.0 s): an octave boing, then climbing back to the theme
    (5, 0, .5, G2), (5, .5, .4, G3), (5, 1.5, .5, D3), (5, 2, .5, C3), (5, 2.5, .5, E3), (5, 3, .5, G3), (5, 3.5, .5, Bb3),
    # bar 8: hides again (second dive)
    (8, 0, .5, A2), (8, .75, .25, A3), (8, 1.5, .5, E3), (8, 2, .5, D3), (8, 2.5, .25, Fs3), (8, 2.75, .25, A3),
    (8, 3, .25, D4), (8, 3.25, .25, C4), (8, 3.5, .25, A3), (8, 3.75, .25, Fs3),
    # bar 9: home - ta-da on F at 19.0 s
    (9, 0, .5, G2), (9, .75, .25, G3), (9, 1, .5, C3), (9, 1.5, .25, E3), (9, 1.75, .25, G3),
    (9, 2, 1.5, F2),
]
for bar, p, d, m in BASS:
    t0 = tb(bar, p) + rng.normal(0, 0.003)
    acc = p % 1 == 0
    v = (0.85 if acc else 0.72) * (1.0 if bar not in (3, 4) else 0.85)
    place('bass', upright(m, d * BEAT * 0.92, v), t0, pan=0.0)
    place('marimba', marimba(m + 12, v * 0.75, 0.4), t0 + 0.003, pan=0.15)
place('bass', upright(F3, 1.0, 0.6), tb(9, 2) + 0.004)          # ta-da: F2 + F3 together

# steel pan: off-beat chords in the A sections, gentle rolls while the fish hides
for bar, pos, c, span in segs():
    v = CH[c]
    if c == 'Fend':
        continue
    if bar in (3, 4):
        # rolled chord (tremolo on two notes), the steel-pan way to sustain
        a, b = v[-2] + 12, v[-1] + 12
        k = 0
        while k * 0.07 < span * BEAT - 0.05:
            place('panroll', steelpan(a if k % 2 == 0 else b, 0.30 + 0.05 * np.sin(k * 0.8), 0.6), tb(bar, pos) + k * 0.07,
                  pan=-0.25 if k % 2 == 0 else 0.25)
            k += 1
        continue
    for q in np.arange(pos, pos + span, 1.0):
        for j, m in enumerate(v):
            place('pan', steelpan(m + 12, 0.36 if j == len(v) - 1 else 0.28, 0.6), tb(bar, q + 0.5) + j * 0.004,
                  pan=float(-0.35 + 0.2 * j))

# steel-pan answers in the gaps of the bass tune (A sections), short and in a middle register
ANS = [(0, 2.0, [72, 74, 77]), (1, 1.0, [74, 77]), (6, 2.0, [72, 74, 77]), (7, 1.0, [74, 77, 81]),
       (5, 1.0, [74, 79]), (9, 1.0, [74, 79])]
for bar, p, notes in ANS:
    for j, m in enumerate(notes):
        place('pan', steelpan(m + 12, 0.42, 0.9), tb(bar, p + 0.25 * j), pan=0.3)
# final ta-da chord
for j, m in enumerate(CH['Fend']):
    place('pan', steelpan(m + 12, 0.45, 1.6), END + j * 0.02, pan=float(-0.3 + 0.2 * j))

# nylon-guitar chops on the off-beats ("and" of 2 and 4), out while the fish hides
for bar in list(range(0, 3)) + list(range(5, 10)):
    for q in (1.5, 3.5):
        if bar == 9 and q >= 2:
            continue
        c = [cc for b, p, cc in PROG if (b, p) <= (bar, q)][-1]
        for j, m in enumerate(CH[c]):
            place('gtr', nylon(m - 12 + (12 if j == 0 else 0), 0.10, 0.45), tb(bar, q) + j * 0.008, pan=-0.3)

# percussion
for bar in range(10):
    hidden = bar in (3, 4)
    if bar < 9 or True:
        for q in np.arange(0, 4, 0.25):
            if bar == 9 and q >= 2:
                break
            v = 0.5 if q % 1 == 0 else (0.35 if q % 0.5 == 0 else 0.25)
            place('shaker', shaker(v * (0.7 if hidden else 1.0)), tb(bar, q), pan=0.4)
    # bongos: a martillo-like pattern
    pat = [(0, False, .5), (.5, True, .4), (1, True, .55), (1.5, True, .35), (2, False, .5), (2.5, True, .4),
           (3, True, .6), (3.25, True, .35), (3.5, False, .45)]
    for q, hi, v in pat:
        if bar == 9 and q >= 2:
            continue
        place('bongo', bongo(hi, v * (0.7 if hidden else 1.0)), tb(bar, q), pan=0.25 if hi else 0.15)
    if not hidden:
        for q in (0, 1.5, 3):           # 3+3+2 claves
            if not (bar == 9 and q >= 2):
                place('clave', clave(0.45), tb(bar, q), pan=-0.4)
        for q in (0, 2):
            if not (bar == 9 and q >= 2):
                place('kick', kick(0.8), tb(bar, q))
    else:
        place('kick', kick(0.6), tb(bar, 0))
place('kick', kick(0.9), END)

# bubbles when the fish dives (rising), and the peek "pop"
for t0 in (DIVE1, DIVE2):
    for j, m in enumerate([67, 72, 76, 79, 84]):
        place('bubble', bloop(m, 0.42, 1.0), t0 + 0.15 + j * 0.16 + rng.uniform(0, 0.04), pan=float(rng.uniform(-.5, .5)))
place('peek', bongo(True, 0.8), PEEK, pan=0.2)
place('peek', bloop(72, 0.5, 1.2, 0.05), PEEK + 0.02, pan=-0.2)


# ---------------------------------------------------------------- mix
def reverb(x2, T=1.8):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.3)
        ir = (fft_filter(nz, hi=7000) * w + fft_filter(nz, hi=3000) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.02 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


STEMS['gtr'] = fft_filter(STEMS['gtr'].astype(float), lo=150).astype(np.float32)
STEMS['pan'] = fft_filter(STEMS['pan'].astype(float), hi=7000).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(A=(0, 4), dive=(4, 6), hidden=(6, 10), peek=(10, 12), A2=(12, 16), end=(16, 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.22 * reverb(snd)
mix = fft_filter(mix, lo=30)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.3, 1.0, np.cos(np.clip((t - 19.3) / 0.7, 0, 1) * np.pi / 2) ** 2)
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
