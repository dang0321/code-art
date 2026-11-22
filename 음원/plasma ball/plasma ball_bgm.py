"""Original BGM for "plasma ball.mp4" (20 s), v2: modern EDM, beat-forward, strong bass.

F minor, 120 BPM (10 bars = 20 s). The intro keeps v1's off-beat "dum-dum-dum-dum" bass, which the
user liked. The retro parts are gone (vibrato saw lead, ping-pong delays, 16th arpeggio, Bbm-C).
The drop (10 s, right after the plasma surge at 9.67 s) is a syncopated supersaw chord-stab hook
whose top voice carries the tune, over Fm-Db-Ab-Eb, rolling 16th bass + sub, a hard four-on-the-floor
kick, clap+snare on 2 & 4, 16th hats, deep sidechain pumping, and a snare-roll build with a
quarter-second breath before the hit. Filament surges get short tonal arc zaps (quieter than v1).
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 3
rng = np.random.default_rng(20261201)

BPM = 120.0
BEAT = 60.0 / BPM            # 0.5 s, bar = 2 s
DROP = 6                      # bar 6 starts at 10.0 s
SURGES = [2.53, 5.10, 9.67, 11.30, 13.57, 15.73, 18.70]


def tb(bar, pos=0.0):
    return ((bar - 1) * 4 + pos) * BEAT


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
GAIN = dict(kick=0.46, bass=0.33, sub=0.30, chords=0.020, hook=0.095, hookhi=0.43, clap=0.50, snare=0.30,
            hat=0.27, crash=0.13, riser=0.05, zap=0.28, impact=0.42)
SEND = dict(kick=.02, bass=.03, sub=.0, chords=.25, hook=.22, hookhi=.25, clap=.25, snare=.25, hat=.08,
            crash=.20, riser=.30, zap=.30, impact=.15)


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
def saw(f, t, fc, kmax=60, ph0=0.0):
    out = np.zeros_like(t)
    for k in range(1, kmax + 1):
        if f * k > 15000:
            break
        out += np.sin(2 * np.pi * f * k * t + ph0 * k) / k / np.sqrt(1 + (f * k / fc) ** 4)
    return out


def bass(m, dur, vel=0.8, bright=1.0):
    """Thumping synth bass: detuned saw + square-ish layer, punchy filter envelope, saturation."""
    f = float(mtof(m)); T = dur + 0.05; t = tt(T)
    fc = 300 + 1800 * bright * np.exp(-t / 0.07)
    x = saw(f * 2 ** (-8 / 1200), t, fc) + saw(f * 2 ** (8 / 1200), t, fc, ph0=1.1)
    x += 0.6 * sum(np.sin(2 * np.pi * f * k * t) / k / np.sqrt(1 + (f * k / fc) ** 4) for k in range(1, 30, 2) if f * k < 8000)
    x = np.tanh(2.0 * x) / np.tanh(2.0)
    env = np.minimum(1, t / 0.003) * (0.75 + 0.25 * np.exp(-t / 0.08)) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.012))
    return x * env * vel


def subnote(m, dur, vel=0.8):
    f = float(mtof(m)); T = dur + 0.05; t = tt(T)
    env = np.minimum(1, t / 0.004) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.015))
    return np.sin(2 * np.pi * f * t) * env * vel


def supersaw(notes, dur, vel=0.6, fc=4500):
    T = dur + 0.3; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for v in range(7):
            det = (v - 3) * 9 + rng.normal(0, 2)
            s = saw(float(mtof(m)) * 2 ** (det / 1200), t, fc, kmax=30, ph0=rng.uniform(0, 6))
            w = (v + 0.5) / 7
            L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.minimum(1, t / 0.01) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.12))
    return np.vstack([L * env, R * env]) * vel


def lead_line(notes, glide=0.035):
    """Bold saw lead: two detuned saws + a sub-octave square, very short glides, delayed vibrato."""
    L = int(DUR * SR) + SR
    pitch = np.full(L, np.nan); amp = np.zeros(L); vibe = np.zeros(L); onset = np.zeros(L)
    for i, (on, off, m, v) in enumerate(notes):
        i0 = int(on * SR)
        nxt = int(notes[i + 1][0] * SR) if i + 1 < len(notes) else L
        end = min(nxt, int((off + 0.4) * SR), L)
        tl = np.arange(end - i0) / SR
        prev = notes[i - 1] if i > 0 else None
        legato = prev is not None and on - prev[1] < 0.02 and prev[2] != m
        p = np.full(end - i0, float(m))
        if legato:
            g = min(len(p), int(glide * SR))
            p[:g] = m + (prev[2] - m) * np.exp(-np.arange(g) / (g / 4))
        pitch[i0:end] = p
        env = np.minimum(1, tl / 0.008) * (0.85 + 0.15 * np.exp(-tl / 0.15))
        env *= np.where(tl < off - on, 1.0, np.exp(-(tl - (off - on)) / 0.06))
        amp[i0:end] = env * v
        vibe[i0:end] = np.clip((tl - 0.35) / 0.3, 0, 1)
        onset[i0:end] = tl
    idx = np.where(~np.isnan(pitch), np.arange(L), 0); np.maximum.accumulate(idx, out=idx)
    pitch = pitch[idx]; pitch[np.isnan(pitch)] = 69
    t = np.arange(L) / SR
    fr = mtof(pitch) * 2 ** (vibe * 14 * np.sin(2 * np.pi * 5.5 * t) / 1200)
    ph1 = 2 * np.pi * np.cumsum(fr * 2 ** (-7 / 1200)) / SR
    ph2 = 2 * np.pi * np.cumsum(fr * 2 ** (7 / 1200)) / SR
    fc = 1800 + 2600 * np.exp(-onset / 0.12)
    x = np.zeros(L)
    for k in range(1, 26):
        a = 1.0 / k / np.sqrt(1 + (k * fr / fc) ** 4)
        x += a * (np.sin(k * ph1) + np.sin(k * ph2 + 0.7 * k))
    x += 0.25 * np.sin(0.5 * ph1)                 # clean sub-octave for weight (no aliasing)
    x = np.tanh(1.2 * x)
    return (x * amp)[:int(DUR * SR)]


def pluck(m, vel=0.4):
    f = float(mtof(m)); t = tt(0.35)
    fc = 700 + 3500 * np.exp(-t / 0.04)
    return saw(f, t, fc, kmax=30) * np.minimum(1, t / 0.002) * np.exp(-t / 0.1) * vel


def kick(vel=1.0):
    t = tt(0.4)
    ph = 2 * np.pi * np.cumsum(46 + 140 * np.exp(-t / 0.022)) / SR
    k = int(0.003 * SR)
    x = np.tanh(1.5 * np.sin(ph) * np.exp(-t / 0.16))
    x[:k] += fft_filter(noise(k), lo=2500) * 0.25
    return x * vel


def clap(vel=0.8):
    t = tt(0.35)
    nz = fft_filter(noise(len(t)), lo=900, hi=6000); nz /= nz.std()
    env = sum(np.where(t >= d, np.exp(-(t - d) / 0.006), 0) for d in (0, 0.008, 0.017))
    env = env + np.where(t >= 0.025, np.exp(-(t - 0.025) / 0.1), 0) * 0.8
    return np.tanh(0.7 * nz) * env * vel


def hat(vel=0.4, open_=False):
    t = tt(0.25 if open_ else 0.05)
    nz = fft_filter(noise(len(t)), lo=7000, hi=14000); nz /= nz.std()
    return np.tanh(0.6 * nz) / 0.6 * np.exp(-t / (0.07 if open_ else 0.012)) * np.minimum(1, t / 0.0008) * vel


def crash(vel=0.8):
    t = tt(2.5)
    nz = fft_filter(noise(len(t)), lo=4000, hi=13000); nz /= nz.std()
    return np.tanh(0.6 * nz) / 0.6 * np.exp(-t / 0.8) * np.minimum(1, t / 0.002) * vel


def riser(T, m0=53, m1=77, vel=0.6):
    """Tonal riser: a saw fifth sweeping up an octave and a half, opening as it climbs."""
    t = tt(T); x = np.zeros_like(t); xx = t / T
    m = m0 + (m1 - m0) * xx ** 1.5
    for off in (0, 7, 12):
        ph = 2 * np.pi * np.cumsum(mtof(m + off)) / SR
        fc = 800 + 6000 * xx ** 2
        for k in range(1, 12):
            x += np.sin(k * ph) / k / np.sqrt(1 + (k * mtof(m + off) / fc) ** 4)
    return x * xx ** 2.2 * vel


def zap(vel=0.5):
    """Plasma arc: a quick downward FM chirp with a tonal crackle; short and mid-range."""
    t = tt(0.28)
    f = 220 + 1400 * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph + 3.0 * np.exp(-t / 0.05) * np.sin(2.7 * ph))
    return np.tanh(1.5 * x) * np.exp(-t / 0.07) * np.minimum(1, t / 0.001) * vel


def snare_body(vel=0.8):
    """Tight snare layered under the clap: punchy tone + short, bright crack."""
    t = tt(0.25)
    tone = np.sin(2 * np.pi * np.cumsum(200 * (1 + 0.4 * np.exp(-t / 0.01))) / SR) * np.exp(-t / 0.05)
    nz = fft_filter(noise(len(t)), lo=1500, hi=9000); nz /= nz.std()
    return (0.9 * tone + 0.5 * np.tanh(0.7 * nz) * np.exp(-t / 0.07)) * np.minimum(1, t / 0.0008) * vel


def impact(vel=1.0):
    t = tt(2.0)
    body = np.sin(2 * np.pi * np.cumsum(40 + 90 * np.exp(-t / 0.05)) / SR) * np.exp(-t / 0.6)
    return np.tanh(1.5 * body) * np.minimum(1, t / 0.002) * vel


# ---------------------------------------------------------------- harmony (one chord per bar)
CH = {'Fm': ([53, 56, 60, 65], 29), 'Db': ([53, 56, 61, 65], 37), 'Ab': ([51, 56, 60, 63], 32),
      'Eb': ([51, 55, 58, 63], 39)}
BARS = {1: 'Fm', 2: 'Fm', 3: 'Db', 4: 'Ab', 5: 'Eb', 6: 'Fm', 7: 'Db', 8: 'Ab', 9: 'Eb', 10: 'Fm'}
GAP = 0.25                                   # a breath of silence right before the drop
GAP_T = tb(DROP) - GAP

# bass: the off-beat "dum-dum" intro the user liked, rolling 16ths after the drop
for bar in range(1, 10):
    v, r = CH[BARS[bar]]
    bright = 0.35 if bar == 1 else (0.7 if bar < DROP else 1.0)
    if bar < DROP:
        for beat in range(4):
            t0 = tb(bar, beat + 0.5)
            if t0 >= GAP_T - 0.01:
                continue
            place('bass', bass(r + 12, BEAT * 0.42, 0.85, bright), t0)
            place('sub', subnote(r, BEAT * 0.42, 0.9), t0)
    else:
        for beat in range(4):
            for s16 in (1, 2, 3):
                place('bass', bass(r + (24 if s16 == 2 else 12), BEAT * 0.2, 0.95 if s16 == 1 else 0.78, bright),
                      tb(bar, beat + s16 / 4))
            place('sub', subnote(r, BEAT * 0.9, 0.95), tb(bar, beat))
place('bass', bass(41, 1.6, 0.9, 1.0), tb(10)); place('sub', subnote(29, 1.6, 0.9), tb(10))

# off-beat house chord stabs in bars 2-3; a held, deeply pumping chord bed under the drop
for bar in (2, 3):
    notes = [n + 12 for n in CH[BARS[bar]][0]]
    for beat in range(4):
        place('chords', supersaw(notes, BEAT * 0.3, 0.55, fc=3200), tb(bar, beat + 0.5))
for bar in range(DROP, 10):
    place('chords', supersaw([n + 12 for n in CH[BARS[bar]][0]], BEAT * 4 * 0.98, 0.45, fc=4000), tb(bar))

# EDM hook: syncopated supersaw chord stabs (3-3-2 feel); the top voice carries the tune
HOOK_POS = [0, 3, 6, 8, 11, 14]
HOOK = {6: [72, 72, 75, 77, 75, 72], 7: [73, 73, 77, 80, 77, 73], 8: [72, 72, 75, 80, 75, 72],
        9: [70, 70, 75, 79, 77, 75]}


def stab_voicing(bar, top):
    tones = [n + 12 for n in CH[BARS[bar]][0]] + [n + 24 for n in CH[BARS[bar]][0]]
    under = sorted({n for n in tones if top - 12 < n < top})
    return under[-3:] + [top]


def hook_bar(bar, tops, fc, vel, layer_hi=True):
    for p16, top in zip(HOOK_POS, tops):
        t0 = tb(bar, p16 / 4)
        if bar < DROP and t0 >= GAP_T - 0.01:
            continue
        ln = BEAT * (0.36 if p16 in (6, 14) else 0.30)
        place('hook', supersaw(stab_voicing(bar, top), ln, vel, fc=fc), t0)
        if layer_hi:
            place('hookhi', pluck(top + 12, vel * 0.9), t0, pan=0.0)


# teaser in bars 4-5: the same hook, filtered down and opening up into the build
hook_bar(4, HOOK[8], 900, 0.35, layer_hi=False)
hook_bar(5, HOOK[9], 1600, 0.42, layer_hi=False)
for bar in range(DROP, 10):
    hook_bar(bar, HOOK[bar], 7000, 0.62)
place('hook', supersaw([65, 68, 72, 77], 1.3, 0.7, fc=6000), tb(10))
place('hookhi', pluck(89, 0.6), tb(10))

# drums: hard 4/4, clap + snare on 2 & 4, 16th closed hats + off-beat open hats
for bar in range(2, 11):
    for beat in range(4):
        if bar == 10 and beat > 0:
            break
        place('kick', kick(1.1 if bar >= DROP else 1.0), tb(bar, beat))
    if 3 <= bar <= 9:
        for beat in (1, 3):
            if bar == 5 and beat == 3:
                continue
            place('clap', clap(0.85), tb(bar, beat), pan=0.05)
            place('snare', snare_body(0.8), tb(bar, beat))
    if 2 <= bar <= 9:
        for beat in range(4):
            if bar >= DROP or tb(bar, beat + 0.5) < GAP_T - 0.01:
                place('hat', hat(0.5, open_=True), tb(bar, beat + 0.5), pan=0.3)
            if bar >= 3:
                for s_, v_ in [(0.25, .22), (0.75, .22)] + ([(0, .3)] if bar >= DROP else []):
                    if bar >= DROP or tb(bar, beat + s_) < GAP_T - 0.01:
                        place('hat', hat(v_), tb(bar, beat + s_), pan=0.35)
# build: snare roll speeding up (8ths -> 16ths), tonal riser, then a quarter-second breath before the drop
roll = [tb(5, k * 0.5) for k in range(4)] + [tb(5, 2 + k * 0.25) for k in range(7)]
for k, t0 in enumerate(roll):
    if t0 < GAP_T - 0.01:
        place('snare', snare_body(0.25 + 0.06 * k), t0)
        place('clap', clap(0.2 + 0.05 * k), t0, pan=0.05)
place('riser', riser(2.0 - GAP, 53, 77, 0.9), tb(DROP) - 2.0)
for t0, v in [(tb(DROP), 1.0), (tb(10), 0.9)]:
    place('impact', impact(v), t0); place('crash', crash(0.8), t0, pan=-0.2)

# plasma surges: short tonal arc zaps (a little quieter than v1)
for t0 in SURGES:
    place('zap', zap(0.6), t0, pan=float(rng.uniform(-.5, .5)))
    place('zap', zap(0.35), t0 + 0.06, pan=float(rng.uniform(-.6, .6)))


# ---------------------------------------------------------------- mix
def reverb(x2, T=1.6):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.25)
        ir = (fft_filter(nz, hi=9000) * w + fft_filter(nz, hi=3500) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.012 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


# sidechain pump: everything melodic ducks under each kick (from bar 2) and breathes back
tN = np.arange(N) / SR
pump = np.ones(N)
for bar in range(2, 11):
    for beat in range(4):
        if bar == 10 and beat > 0:
            break
        x = tN - tb(bar, beat)
        pump -= np.where((x >= 0) & (x < BEAT), 0.75 * np.exp(-x / 0.12), 0)
pump = np.clip(pump, 0.2, 1.0)
for k, depth in [('chords', 1.0), ('bass', 0.6), ('hook', 0.55), ('hookhi', 0.4)]:
    STEMS[k] = (STEMS[k].astype(float) * (1 - depth * (1 - pump))).astype(np.float32)
STEMS['chords'] = fft_filter(STEMS['chords'].astype(float), lo=180).astype(np.float32)
STEMS['hook'] = fft_filter(STEMS['hook'].astype(float), lo=200).astype(np.float32)
# the pre-drop breath: everything except the drop hits goes quiet for a quarter second
gate = np.interp(tN, [GAP_T - 0.03, GAP_T, tb(DROP) - 0.004, tb(DROP)], [1.0, 0.03, 0.03, 1.0])
for k in STEMS:
    if k not in ('impact', 'crash'):
        STEMS[k] = (STEMS[k].astype(float) * gate).astype(np.float32)
STEMS['bass'] = fft_filter(STEMS['bass'].astype(float), lo=40, hi=4000).astype(np.float32)
STEMS['sub'] = fft_filter(STEMS['sub'].astype(float), hi=150).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, tb(2)), A=(tb(2), tb(5)), build=(tb(5), tb(DROP)), drop=(tb(DROP), tb(10)), end=(tb(10), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.20 * reverb(snd)
mix = fft_filter(mix, lo=28)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.0, 1.0, np.cos(np.clip((t - 19.0) / 1.0, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.01)
mix /= np.abs(mix).max()
drive = 1.6
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
