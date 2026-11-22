"""hanabi2_bgm.py - the soundtrack for hanabi2.mp4: a piano-and-strings J-pop ballad in E-flat at
96 bpm, eight bars long. Every instrument is synthesised from how the real one makes its sound - a
piano's stiff, slightly mistuned strings; a string section of players who are never quite together;
timpani, taiko and an acoustic kit - then played into a concert-hall reverb.

The chorus lands on the Niagara (10.3 s) and the last chord on the kaleidoscope finale (18 s).

    python hanabi2_bgm.py                         -> hanabi2_bgm.wav
    python hanabi2_bgm.py --mux silent.mp4 out.mp4 -> also lays the music under a silent video
"""
import math, os, shutil, subprocess, sys, wave
import numpy as np

SR = 44100
DUR = 20.0
NS = int(SR * (DUR + 4))                  # room for the tails
TAU = math.tau
rng = np.random.default_rng(11)
BPM = 96
BEAT = 60 / BPM
BAR = 4 * BEAT
T0 = 0.3                                  # first downbeat, so that bar 5 falls on the Niagara


def at(bar, beat=0.0):
    return T0 + (bar - 1) * BAR + beat * BEAT


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def band(lo, hi, k=2):                    # a soft band-pass response
    return lambda f: 1 / (1 + (lo / np.maximum(f, 1)) ** (2 * k)) / (1 + (f / hi) ** (2 * k))


def coloured(n, resp):                    # white noise shaped by a frequency response
    m = 1 << max(1, (n - 1).bit_length())
    X = np.fft.rfft(rng.normal(size=m))
    s = np.fft.irfft(X * resp(np.fft.rfftfreq(m, 1 / SR)), m)[:n]
    return s / (np.sqrt((s ** 2).mean()) + 1e-12)


class Bus:
    def __init__(self, send):
        self.x, self.send = np.zeros((2, NS)), send

    def add(self, s, t, pan=0.0, g=1.0):
        i = int(round(t * SR))
        if i < 0:
            s, i = s[-i:], 0
        n = min(len(s), NS - i)
        if n > 0:
            a = (np.clip(pan, -1, 1) + 1) * math.pi / 4
            self.x[0, i:i + n] += s[:n] * math.cos(a) * g
            self.x[1, i:i + n] += s[:n] * math.sin(a) * g


# --- piano: inharmonic partials of a stiff string, three strings per key slightly out of tune with
# each other (the slow beating), a fast first decay and a long aftersound, a felt hammer's thump
def piano(m, vel, off):
    f0 = hz(m)
    tr = 0.1 if m > 50 else 0.25                         # dampers: quick in the treble, slow on the bass
    tau0 = 9.0 * 2 ** (-(m - 36) / 20)
    L = min(off + 5 * tr, 5 * tau0, 9.0)
    t = np.arange(int(L * SR)) / SR
    n = np.arange(1, 40)
    B = np.clip(0.00035 * 2 ** ((m - 60) / 18), 6e-5, 0.004)
    fn = f0 * n * np.sqrt(1 + B * n * n)
    n, fn = n[fn < 12000], fn[fn < 12000]
    a = n ** -(2.0 - 1.1 * vel) * (0.15 + np.abs(np.sin(math.pi * n / 7.2)))   # struck an eighth along
    a = a / (1 + (fn / (1600 + 5200 * vel)) ** 2)                               # soft blows stay dark
    if m < 48:
        a[0] *= 0.6
    tau = tau0 / (1 + (fn / 2200) ** 1.3 + 0.04 * (n - 1))
    beat = fn * 0.00045 * rng.uniform(0.4, 1.4, len(n))
    ph = rng.uniform(0, TAU, len(n))
    s = np.zeros(len(t))
    for i0 in range(0, len(t), 32768):                   # in chunks, to keep the arrays small
        T = t[i0:i0 + 32768, None]
        env = 0.62 * np.exp(-T / (0.22 * tau)) + 0.38 * np.exp(-T / tau)
        s[i0:i0 + 32768] = (a * env * (1 + 0.28 * np.cos(TAU * beat * T + ph)) * np.sin(TAU * fn * T + ph)).sum(1)
    s *= (1 - np.exp(-t / 0.0025)) * np.where(t > off, np.exp(-(t - off) / tr), 1.0)
    s /= math.sqrt((a ** 2).sum())
    k = int(0.04 * SR)
    s[:k] += coloured(k, band(60, 900 + 1500 * vel)) * np.exp(-np.arange(k) / SR / 0.007) * 0.04 * vel
    return s * vel ** 1.5


# --- strings: each section is several players, each a little out of tune with the rest, each with
# their own vibrato (fading in after the bow settles) and bow pressure; the wooden body colours it all
BODIES = {"vn": [(290, 0.9, 0.35), (480, 1.0, 0.3), (1000, 0.5, 0.5), (2800, 1.2, 0.6)],
          "va": [(230, 0.9, 0.35), (380, 1.0, 0.3), (850, 0.5, 0.5), (2200, 1.0, 0.6)],
          "vc": [(100, 0.8, 0.35), (200, 1.0, 0.3), (580, 0.6, 0.5), (1600, 0.8, 0.6)],
          "cb": [(60, 0.8, 0.4), (110, 1.0, 0.35), (300, 0.5, 0.5), (900, 0.5, 0.6)]}


def body(f, kind):
    g = 0.25 + sum(gn * np.exp(-0.5 * (np.log2(np.maximum(f, 1) / fc) / w) ** 2) for fc, gn, w in BODIES[kind])
    return g / (1 + (f / 6000) ** 4)


def bowed(m, t0, dur, dyn, kind="vn", players=6, glide=None, att=0.12, rel=0.35):
    f0 = hz(m)
    nS = int((dur + rel) * SR)
    t = np.arange(nS) / SR
    lvl = dyn(t0 + np.minimum(t, dur))
    env = lvl * smooth(t / att) * np.where(t > dur, np.exp(-(t - dur) / (rel / 3)), 1.0)
    n = np.arange(1, int(min(48, 11000 / f0)) + 1)
    bright = float(lvl.max())
    out = np.zeros(nS)
    for p in range(players):
        det = 2 ** (rng.normal(0, 5) / 1200)
        vib = rng.uniform(0.0022, 0.004) * smooth((t - 0.22) / 0.5) * np.sin(TAU * rng.uniform(5.0, 6.3) * t + rng.uniform(0, TAU))
        f = f0 * det * (1 + vib)
        if glide is not None:                              # a slide up or down into the note
            g = smooth(t / 0.09)
            f = f * (hz(glide) / f0) ** (1 - g)
        phase = TAU * np.cumsum(f) / SR
        fn = n * f0
        amp = body(fn, kind) / n / (1 + (fn / (2200 + 4500 * bright)) ** 2)
        wob = 1 + 0.08 * np.sin(TAU * rng.uniform(0.3, 0.9) * t + rng.uniform(0, TAU))
        late = int(abs(rng.normal(0, 0.012)) * SR)
        phs = rng.uniform(0, TAU, len(n))
        s = np.zeros(nS)
        for i0 in range(0, nS, 32768):
            s[i0:i0 + 32768] = np.sin(phase[i0:i0 + 32768, None] * n + phs) @ amp
        out[late:] += (s * wob)[:nS - late]
    out /= math.sqrt(players) * math.sqrt((amp ** 2).sum()) * 2.2
    rosin = coloured(nS, lambda fr: body(fr, kind) * band(300, 7000)(fr)) * 0.03
    return (out + rosin) * env


# --- percussion: membranes and bars ring in their own inharmonic modes; cymbals are coloured noise
def modes(f0, ratios, amps, decays, L, bend=0.0, bend_t=0.05):
    t = np.arange(int(L * SR)) / SR
    pitch = 1 + bend * np.exp(-t / bend_t)
    ph = TAU * np.cumsum(pitch) / SR
    return sum(a * np.exp(-t / d) * np.sin(ph * f0 * r) for r, a, d in zip(ratios, amps, decays)), t


def timpani(m, vel, L=3.5):
    s, t = modes(hz(m), [1.0, 1.5, 1.99, 2.44, 2.9], [1, 0.6, 0.35, 0.2, 0.12], [1.7, 1.1, 0.8, 0.5, 0.35], L, 0.04, 0.03)
    k = int(0.05 * SR)
    s[:k] += coloured(k, band(40, 900)) * np.exp(-t[:k] / 0.012) * 0.5
    return s * vel * 0.6


def taiko(vel, f0=64, L=2.0):
    s, t = modes(f0, [1, 1.59, 2.14, 2.3, 2.65, 2.92], [1, 0.5, 0.3, 0.25, 0.15, 0.1], [0.75, 0.42, 0.3, 0.26, 0.2, 0.15], L, 0.35, 0.05)
    k = int(0.03 * SR)
    s[:k] += coloured(k, band(700, 4000)) * np.exp(-t[:k] / 0.005) * 0.5     # the bachi striking the skin
    return s * vel * 0.8


def kick(vel):
    t = np.arange(int(0.6 * SR)) / SR
    ph = TAU * np.cumsum(48 + 70 * np.exp(-t / 0.035)) / SR
    s = np.sin(ph) * np.exp(-t / 0.26)
    k = int(0.02 * SR)
    s[:k] += coloured(k, band(1500, 6000)) * np.exp(-t[:k] / 0.003) * 0.25
    return s * vel


def snare(vel):
    t = np.arange(int(0.45 * SR)) / SR
    s = (np.sin(TAU * 185 * t) * np.exp(-t / 0.11) + 0.5 * np.sin(TAU * 290 * t) * np.exp(-t / 0.07)
         + 0.3 * np.sin(TAU * 340 * t) * np.exp(-t / 0.05)) * 0.6
    s += coloured(len(t), band(1800, 9000)) * np.exp(-t / (0.12 + 0.06 * vel)) * 0.35      # the wires
    return s * vel


def xstick(vel):
    t = np.arange(int(0.12 * SR)) / SR
    return (coloured(len(t), band(600, 3000)) * np.exp(-t / 0.02) * 0.4 + np.sin(TAU * 470 * t) * np.exp(-t / 0.03)) * vel * 0.5


def hat(vel, open_=False):
    t = np.arange(int((0.6 if open_ else 0.12) * SR)) / SR
    return coloured(len(t), band(6500, 15000)) * np.exp(-t / (0.32 if open_ else 0.04)) * vel * 0.22


def shaker(vel):
    t = np.arange(int(0.08 * SR)) / SR
    return coloured(len(t), band(3500, 11000)) * smooth(t / 0.012) * np.exp(-t / 0.025) * vel * 0.11


def crash(vel, L=3.5):
    t = np.arange(int(L * SR)) / SR
    s = coloured(len(t), band(2800, 16000, 1)) * np.exp(-t / 1.6) + 0.6 * coloured(len(t), band(400, 3000)) * np.exp(-t / 0.7)
    f = rng.uniform(2500, 11000, 90)                       # a shimmer of metal modes
    s += 0.3 * sum(np.sin(TAU * fi * t + rng.uniform(0, TAU)) * np.exp(-t / rng.uniform(0.6, 2.2)) for fi in f) / 9.5
    return s * smooth(t / 0.003) * vel * 0.3


def cym_roll(L, vel):                      # a suspended cymbal rolled with soft mallets, swelling
    t = np.arange(int(L * SR)) / SR
    sw = (t / L) ** 2.2
    s = coloured(len(t), band(1200, 11000, 1)) * sw * (1 + 0.15 * np.sin(TAU * 13 * t))
    return s * vel * 0.22


def glock(m, vel):
    s, t = modes(hz(m), [1, 2.76, 5.4, 8.93], [1, 0.35, 0.15, 0.08], [1.8, 0.6, 0.25, 0.12], 2.5)
    k = int(0.01 * SR)
    s[:k] += coloured(k, band(3000, 10000)) * np.exp(-t[:k] / 0.002) * 0.3
    return s * vel * 0.35


# --- harmony: IV-V-iii-vi in the chorus, the progression that lifts a J-pop ballad
CHORDS = [  # bar, beat, beats, bass, voicing
    (1, 0, 4, 44, [51, 55, 60, 63]),       # Abmaj7
    (2, 0, 2, 43, [53, 58, 62]),           # Gm7
    (2, 2, 2, 48, [51, 55, 58]),           # Cm7
    (3, 0, 2, 41, [51, 56, 60]),           # Fm7
    (3, 2, 2, 46, [51, 53, 56]),           # Bb7sus4
    (4, 0, 2, 44, [51, 55, 60]),           # Abmaj7
    (4, 2, 1, 46, [51, 53, 58]),           # Bbsus4
    (4, 3, 1, 46, [50, 53, 58]),           # Bb
    (5, 0, 2, 44, [55, 60, 63, 67]),       # Abmaj7
    (5, 2, 2, 46, [53, 58, 62, 68]),       # Bb7
    (6, 0, 2, 43, [53, 58, 62, 65]),       # Gm7
    (6, 2, 2, 48, [55, 58, 63, 67]),       # Cm7
    (7, 0, 2, 41, [56, 60, 63, 68]),       # Fm7
    (7, 2, 1, 46, [56, 58, 63, 65]),       # Bb7sus4
    (7, 3, 1, 46, [56, 58, 62, 65]),       # Bb7
    (8, 0, 6, 39, [55, 58, 63, 65, 70]),   # Ebadd9, held to the end
]

# --- melody: an original tune - a quiet verse that climbs, then a chorus that leaps to the top
VERSE = [(1, 1, .5, 70), (1, 1.5, .5, 72), (1, 2, 1, 75), (1, 3, .5, 74), (1, 3.5, .5, 75),
         (2, 0, 1.5, 79), (2, 1.5, .5, 77), (2, 2, 1, 75), (2, 3, 1, 72),
         (3, .5, .5, 68), (3, 1, .5, 72), (3, 1.5, .5, 75), (3, 2, 1, 77), (3, 3, .5, 75), (3, 3.5, .5, 77),
         (4, 0, 1.5, 79), (4, 1.5, .5, 80), (4, 2, 1, 82), (4, 3, .5, 79), (4, 3.5, .5, 82)]
CHORUS = [(5, 0, 1, 84), (5, 1, .5, 82), (5, 1.5, .5, 79), (5, 2, 1.5, 82), (5, 3.5, .5, 75),
          (6, 0, .5, 77), (6, .5, 1, 79), (6, 1.5, .5, 82), (6, 2, 1, 84), (6, 3, .5, 82), (6, 3.5, .5, 79),
          (7, 0, 1, 80), (7, 1, .5, 79), (7, 1.5, .5, 77), (7, 2, .5, 79), (7, 2.5, 1.5, 82),
          (8, 0, 5.2, 87)]


def chord_at(t):
    for b, bt, n, bass, v in CHORDS:
        if at(b, bt) <= t + 1e-6 < at(b, bt) + n * BEAT:
            return at(b, bt), at(b, bt) + n * BEAT, bass, v
    return None


def dyn(t):                                # the strings' overall level through the song
    return np.interp(t, [0, at(2), at(3), at(4), at(4, 3.8), at(5), at(7), at(8), at(8, 2), at(8, 4.5), 22],
                     [0.2, 0.22, 0.3, 0.38, 0.8, 0.85, 0.9, 1.0, 1.0, 0.55, 0.4])


def human(t, s=0.006):
    return t + rng.normal(0, s)


# --- arrangement
PIANO, STR, PERC, GLK = Bus(0.3), Bus(0.45), Bus(0.18), Bus(0.4)
END = at(8, 6)


def ppan(m):
    return float(np.clip((m - 62) / 40, -0.6, 0.6))


def play_piano(m, t, vel, until):
    PIANO.add(piano(m, float(np.clip(vel + rng.normal(0, 0.03), 0.05, 1)), max(until - t + 0.02, 0.1)), human(t), ppan(m))


for b, bt, n, bass, v in CHORDS:
    t0, t1 = at(b, bt), at(b, bt) + n * BEAT
    if b <= 4:                             # verse: bass note, then a soft rippling broken chord
        play_piano(bass, t0, 0.42 + 0.04 * b, t1)
        pat = [0, 1, 2, len(v) - 1, 1, 2, 0, 2]
        for i in range(int(n * 2)):
            play_piano(v[pat[i % 8] % len(v)], t0 + i * BEAT / 2, 0.28 + 0.05 * (b - 1) + 0.06 * (i % 2 == 0), t1)
    elif b <= 7:                           # chorus: driving octave bass, chords on the beat
        for off, vv in ((0, 0.8), (1.5, 0.55)):
            if off < n:
                play_piano(bass - 12, t0 + off * BEAT, vv, t1)
                play_piano(bass, t0 + off * BEAT, vv * 0.85, t1)
        for i in range(int(n)):
            for m in v:
                play_piano(m + 12, t0 + i * BEAT, 0.36 if i == 0 else 0.27, t1)
    else:                                  # the last chord, rolled from the bottom to the top
        roll = [27, 39, 46, 51, 55, 58, 63, 65, 67, 70, 75, 79]
        for i, m in enumerate(roll):
            play_piano(m, t0 + i * 0.022, 0.78 - 0.02 * i, END)

for b, bt, d, m in VERSE + CHORUS:         # the tune on the piano: alone in the verse, in octaves after
    t0 = at(b, bt)
    c = chord_at(t0)
    until = c[1] if c else t0 + d * BEAT
    if b <= 4:
        play_piano(m, t0, 0.5 + 0.05 * (b - 1), max(until, t0 + d * BEAT))
    else:
        play_piano(m, t0, 0.82, max(until, t0 + d * BEAT))
        play_piano(m - 12, t0, 0.7, max(until, t0 + d * BEAT))

# strings: a low pad from bar 2, violins take the tune in bar 4, everyone in the chorus
for b, bt, n, bass, v in CHORDS:
    t0, t1 = at(b, bt), at(b, bt) + n * BEAT
    if b == 1:
        continue
    d = t1 - t0 + 0.06
    STR.add(bowed(bass if bass >= 36 else bass + 12, t0, d, dyn, "vc", 4), t0, 0.35, 0.9)
    STR.add(bowed(v[1] + (12 if b >= 5 else 0), t0, d, dyn, "va", 4), t0, 0.15, 0.55)
    if b >= 3:
        STR.add(bowed(v[-1] + 12, t0, d, dyn, "vn", 4), t0, -0.3, 0.35)
    if b >= 5:
        STR.add(bowed(bass - 12 if bass - 12 >= 28 else bass, t0, d, dyn, "cb", 3), t0, 0.45, 0.8)
        STR.add(bowed(v[0], t0, d, dyn, "va", 4), t0, 0.2, 0.4)
prev = None
for b, bt, d, m in VERSE[15:] + CHORUS:     # the violins sing it, sliding between notes
    t0 = at(b, bt)
    dd = d * BEAT + 0.05
    gl = prev if prev is not None and abs(prev - m) <= 7 else None
    STR.add(bowed(m, t0, dd, dyn, "vn", 6, glide=gl, att=0.06), t0, -0.45, 0.85)
    if b >= 5:
        STR.add(bowed(m - 12, t0, dd, dyn, "vn", 5, glide=None if gl is None else gl - 12, att=0.06), t0, -0.15, 0.6)
    prev = m

# percussion: a hush of shaker and rim in bar 3, the build in bar 4, the full kit and taiko after
for b in (3, 4, 5, 6, 7):                   # shaker in sixteenths, bars 3-7
    for i in range(16):
        PERC.add(shaker(0.5 + 0.3 * (i % 4 == 0) + 0.15 * (i % 2 == 0)), human(at(b, i / 4), 0.004), 0.5)
for b in (3, 4):
    for k in (0, 2.5) if b == 3 else (0, 1.5, 2):
        PERC.add(kick(0.45), human(at(b, k)), 0.0)
    for k in (1, 3) if b == 3 else (1,):
        PERC.add(xstick(0.5), human(at(b, k)), 0.1)
for i in range(8):                         # the snare roll and cymbal swell into the chorus
    PERC.add(snare(0.25 + 0.6 * (i / 7) ** 1.5), human(at(4, 2 + i / 4), 0.004), 0.05)
PERC.add(cym_roll(BAR, 1.0), at(4, 0), 0.3)
for i in range(int(2 * BEAT * 18)):        # a timpani roll on B-flat, swelling
    PERC.add(timpani(46, 0.15 + 0.55 * (i / (2 * BEAT * 18)) ** 2, 1.2), at(4, 2) + i / 18 + rng.normal(0, 0.004), -0.2)
PERC.add(taiko(0.55), at(4, 3), 0.0)
PERC.add(taiko(0.75), at(4, 3.5), 0.0)
for b in (5, 6, 7):
    for k in (0, 1.5, 2, 3.5):
        PERC.add(kick(0.85 if k in (0, 2) else 0.6), human(at(b, k), 0.004), 0.0)
    for k in (1, 3):
        PERC.add(snare(0.8), human(at(b, k), 0.004), 0.05)
    for i in range(8):
        PERC.add(hat(0.5 if i % 2 == 0 else 0.3, open_=(i == 7)), human(at(b, i / 2), 0.004), -0.25)
    PERC.add(taiko(0.95 if b == 5 else 0.75), at(b, 0), 0.0)
    PERC.add(taiko(0.5), at(b, 2), 0.0)
for i in range(4):                          # the fill out of bar 7
    PERC.add(snare(0.45 + 0.15 * i), human(at(7, 3 + i / 4), 0.004), -0.1 + 0.07 * i)
for i in range(int(BEAT * 18)):
    PERC.add(timpani(46, 0.2 + 0.5 * (i / (BEAT * 18)) ** 2, 1.2), at(7, 3) + i / 18, -0.2)
PERC.add(crash(1.0), at(5), -0.4)
PERC.add(crash(0.7), at(7), 0.4)
PERC.add(timpani(44, 0.9), at(5), -0.2)
for p, dt, v in ((-0.4, 0.0, 1.0), (0.45, 0.03, 0.85)):     # the finale: both crashes, timpani, taiko
    PERC.add(crash(v, 4.0), at(8) + dt, p)
PERC.add(kick(1.0), at(8), 0.0)
PERC.add(taiko(1.0, 60, 2.6), at(8), 0.0)
PERC.add(timpani(39, 1.0, 4.0), at(8), -0.2)

# glockenspiel: a few bells on the long chorus notes, and a sparkle up the last chord
for b, bt, d, m in CHORUS[:-1]:
    if d >= 1:
        GLK.add(glock(m + 12, 0.45), human(at(b, bt)), 0.5)
for i, m in enumerate([87, 91, 94, 99, 103, 99, 94, 99]):
    GLK.add(glock(m, 0.5 - 0.03 * i), at(8, 0.25 * i), 0.5 - 0.1 * (i % 3))


# --- hall: a concert-hall reverb - bass lingers, treble dies sooner, early reflections off the walls
def hall_ir(L=3.2):
    n = int(L * SR)
    t = np.arange(n) / SR
    f = np.fft.rfftfreq(n, 1 / SR)
    ir = np.zeros((2, n))
    for ch in (0, 1):
        W = np.fft.rfft(rng.normal(size=n))
        sig = sum(np.fft.irfft(W * b(f), n) * np.exp(-6.91 * t / rt)
                  for b, rt in ((band(1, 400), 2.6), (band(400, 3500), 2.1), (band(3500, 20000), 1.2)))
        sig *= smooth((t - 0.02) / 0.04)
        for _ in range(14):
            k = int(rng.uniform(0.008, 0.07) * SR)
            sig[k] += rng.uniform(0.3, 1.0) * (1 - k / (0.08 * SR)) * 4 * (np.abs(sig).max())
        ir[ch] = sig / np.sqrt((sig ** 2).sum())
    return ir


def convolve(x, h):
    m = 1 << (len(x) + len(h) - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(h, m), m)[:len(x)]


# --- mix: the buses, the hall, a gentle glue compressor and a soft limiter
def compress(x, thr=-18.0, ratio=2.0, att=0.015, rel=0.3, blk=256):
    nb = x.shape[1] // blk
    lev = 20 * np.log10(np.sqrt((x[:, :nb * blk].reshape(2, nb, blk) ** 2).mean((0, 2))) + 1e-9)
    want = np.minimum(0, (thr - lev) * (1 - 1 / ratio))
    g, out = 0.0, np.zeros(nb)
    a_a, a_r = math.exp(-blk / SR / att), math.exp(-blk / SR / rel)
    for i in range(nb):
        k = a_a if want[i] < g else a_r
        g = k * g + (1 - k) * want[i]
        out[i] = g
    gain = 10 ** (np.interp(np.arange(x.shape[1]), np.arange(nb) * blk + blk / 2, out) / 20)
    return x * gain


def mix():
    buses = (PIANO, STR, PERC, GLK)
    gains = (0.55, 0.85, 0.68, 0.5)                # the strings carry the chorus
    dry = sum(B.x * g for B, g in zip(buses, gains))
    send = sum(B.x * g * B.send for B, g in zip(buses, gains))
    ir = hall_ir()
    wet = np.stack([convolve(send[0] * 0.75 + send[1] * 0.25, ir[0]), convolve(send[1] * 0.75 + send[0] * 0.25, ir[1])])
    x = dry + 0.55 * wet
    lo = band(32, 20000, 1)                    # no rumble below the double basses
    m = 1 << (x.shape[1] - 1).bit_length()
    x = np.stack([np.fft.irfft(np.fft.rfft(c, m) * lo(np.fft.rfftfreq(m, 1 / SR)), m)[:x.shape[1]] for c in x])
    x = compress(x)
    x = x[:, :int(DUR * SR)]
    t = np.arange(x.shape[1]) / SR
    x *= smooth(t / 0.01) * (1 - smooth((t - (DUR - 1.1)) / 1.1))       # fade the last chord away
    x *= 10 ** (-15.5 / 20) / np.sqrt((x ** 2).mean())                  # about -14 LUFS for Reels
    x = 0.97 * np.tanh(x / 0.97)
    return x


def write_wav(path, x):
    y = (np.clip(x, -1, 1) * 32767).astype("<i2").T.copy()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


if __name__ == "__main__":
    out = mix()
    write_wav("hanabi2_bgm.wav", out)
    print("peak", float(np.abs(out).max()), "rms dB", float(20 * np.log10(np.sqrt((out ** 2).mean()))))
    if "--mux" in sys.argv:
        k = sys.argv.index("--mux")
        ff = shutil.which("ffmpeg") or "ffmpeg"
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", sys.argv[k + 1], "-i", "hanabi2_bgm.wav",
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-shortest", "-movflags", "+faststart", sys.argv[k + 2]], check=True)
        print("done ->", sys.argv[k + 2])
