"""Procedural ASCII anemone with a clownfish swimming around its crown."""
import argparse
import inspect
import math
from pathlib import Path
import re
import shutil
import subprocess
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1080, 1920
ART_HEIGHT = 1344
CELL_W, CELL_H = 9, 12
COLS, ROWS = WIDTH // CELL_W, ART_HEIGHT // CELL_H
FPS, DURATION = 30, 20
TAU = math.tau
CENTER = np.float32((539, 698))
CROWN_WIDTH, CROWN_HEIGHT = 1.20, 1.28
DISC_Y = 119
GLYPHS = r" .:;+*#@=|-/\oO"
MARK = {character: index for index, character in enumerate(GLYPHS)}
INKS = np.array([MARK[k] for k in ".:;+=#@"], dtype=np.uint8)

GRID_Y, GRID_X = np.indices((ROWS, COLS), dtype=np.float32)
PX = (GRID_X + .5) * CELL_W
PY = (GRID_Y + .5) * CELL_H
SKY = np.float32((.62, .83, 1.0))
ICE = np.float32((.78, .91, 1.0))
PEARL = np.float32((.97, .99, 1.0))
LENS = 1500


def smoothstep(start, stop, x):
    phase = np.clip((x - start) / (stop - start), 0, 1)
    return phase * phase * (3 - 2 * phase)


def bell(x, width):
    return np.exp(-((x / width) ** 2))


def project(x, y, z):
    depth = z - .123 * y
    perspective = 1 / (1 - depth / LENS)
    return CENTER[0] + x * perspective, CENTER[1] + (y + .235 * z) * perspective, depth


def stroke(dx, dy):
    return np.select(
        (np.abs(dx) > 1.7 * np.abs(dy), np.abs(dy) > 2.0 * np.abs(dx), dx * dy > 0),
        (MARK["-"], MARK["|"], MARK["\\"]),
        default=MARK["/"],
    )


class CellScene:
    def __init__(self, depth, ink, marks):
        self.depth = depth.copy()
        self.ink = ink.copy()
        self.marks = marks.copy()
        self.light = np.zeros_like(ink)
        self.volume_layers = []

    def splat(self, x, y, z, ink, marks, glow):
        columns = np.intp(x / CELL_W)
        rows = np.intp(y / CELL_H)
        valid = (x >= 0) & (y >= 0) & (columns < COLS) & (rows < ROWS)
        cells = (rows * COLS + columns)[valid]
        depth = z[valid]
        if not len(cells):
            return
        # Keep the nearest surface when a fish, stalk, or rock crosses a glyph cell.
        nearest = np.full(ROWS * COLS, -np.inf, dtype=depth.dtype)
        np.maximum.at(nearest, cells, depth)
        candidates = np.flatnonzero(depth == nearest[cells])
        _, first = np.unique(cells[candidates], return_index=True)
        chosen = candidates[first]
        row, col = np.divmod(cells[chosen], COLS)
        front = depth[chosen] > self.depth[row, col]
        row, col, chosen = row[front], col[front], chosen[front]
        self.depth[row, col] = depth[chosen]
        self.ink[row, col] = ink[valid][chosen]
        self.marks[row, col] = marks[valid][chosen]
        self.light[row, col] = self.ink[row, col] * glow[valid][chosen, None]


    def composite_volume(self, x, y, z, ink, marks, glow, alpha, tissue):
        columns = np.intp(x / CELL_W)
        rows = np.intp(y / CELL_H)
        inside = (x >= 0) & (y >= 0) & (columns < COLS) & (rows < ROWS)
        hits = np.flatnonzero(inside)
        cell = rows * COLS + columns
        in_front = z[hits] > self.depth[rows[hits], columns[hits]]
        hits = hits[in_front]
        if not len(hits):
            return

        # A tube can cross a cell many times. Keep one surface per tube, per cell.
        key = tissue[hits] * (ROWS * COLS) + cell[hits]
        order = np.lexsort((-z[hits], key))
        distinct = np.r_[True, np.diff(key[order]) != 0]
        hits = hits[order[distinct]]

        # Blend the layers in depth order so a near arm can transmit a dim far arm.
        order = np.lexsort((-z[hits], cell[hits]))
        hits = hits[order]
        sequence = np.arange(len(hits))
        starts = np.r_[True, np.diff(cell[hits]) != 0]
        rank = sequence - np.maximum.accumulate(np.where(starts, sequence, 0))
        for index in (3, 2, 1, 0):
            layer = hits[rank == index]
            row, col = rows[layer], columns[layer]
            transmission = 1 - alpha[layer]
            ink_layer = ink[layer] * (.32 + .76 * alpha[layer, None])
            self.volume_layers.append((row, col, transmission, ink_layer, marks[layer]))
            self.light[row, col] *= transmission[:, None]
            self.light[row, col] += ink_layer * glow[layer, None]

    def glyph_pixels(self, atlas):
        glyphs = atlas[self.marks][..., None] * self.ink[:, :, None, None, :]
        for row, col, transmission, ink, marks in self.volume_layers:
            behind = glyphs[row, col] * transmission[:, None, None, None]
            emission = atlas[marks][..., None] * ink[:, None, None, :]
            glyphs[row, col] = behind + emission
        return glyphs


class Water:
    def __init__(self, rng):
        self.x = rng.uniform(-580, 580, 260)
        self.y = rng.uniform(-600, 465, 260)
        self.z = rng.uniform(-500, 260, 260)
        self.phase = rng.uniform(0, TAU, 260)
        self.speed = rng.uniform(4, 14, 260)
        self.glint = rng.random(260)

    def paint(self, field, time):
        x = self.x + 36 * np.sin(.23 * time + self.phase) + 2.7 * time
        y = (self.y - self.speed * time + 600) % 1065 - 600
        px, py, depth = project(x, y, self.z)
        shimmer = (.44 + .56 * np.sin(1.8 * time + self.phase) ** 4)
        plume = bell(x + 19, 343) * bell(y - 37, 365)
        power = (.031 + .17 * self.glint ** 3) * shimmer
        power += .24 * plume * shimmer * self.glint ** 6
        ink = power[:, None] * np.float32((.35, .82, .94))
        mark = np.where(power > .21, MARK["+"], np.where(power > .12, MARK[":"], MARK["."]))
        field.splat(px, py, depth, ink, mark, power * .33)


class Disc:
    def __init__(self, texture, details):
        u = (PX - CENTER[0]) / (211 * CROWN_WIDTH)
        v = (PY - 851) / 188
        width = 1.04 - .33 * np.maximum(v, 0)
        shape = (u / width) ** 2 + (1.29 * v) ** 2
        shape += .072 * np.sin(9.2 * u + 1.3 * v)
        inside = shape < 1
        curved = np.sqrt(np.maximum(0, 1 - shape))
        folds = .5 + .5 * np.sin(34 * u + 8 * u * v + 4 * v + np.sin(11 * u)) ** 8
        self.x, self.y = PX[inside], PY[inside]
        self.u, self.v = u[inside], v[inside]
        self.depth = (-117 + 152 * curved)[inside]
        self.core = curved[inside]
        self.folds = folds[inside]
        self.texture = texture[inside]
        self.detail = details[inside]
        self.opacity = .10 + .29 * self.core ** .65
        self.tissue = np.zeros(self.x.size, dtype=np.int32)

    def samples(self, time):
        wave = .81 + .19 * np.sin(1.52 * time + 8.2 * self.v + 4.3 * self.u)
        center = bell(self.v + .53, .69) * bell(self.u + .08, .93)
        streaks = .35 + .65 * self.folds ** 2
        light = (.24 + .38 * center) * (.29 + .71 * self.core) * wave
        light *= (.50 + .50 * self.texture) * (.59 + .41 * streaks)
        tint = np.float32((.35, .65, 1.0)) + center[:, None] * np.float32((.20, .17, 0))
        inks = tint * light[:, None]
        marks = np.select(
            ((light > .27) & (self.detail > .88), (self.folds > .82) & (light > .18),
             self.detail > .44),
            (MARK["+"], MARK["|"], MARK[":"]),
            default=MARK["."])
        glow = .55 + .41 * self.core
        return self.x, self.y, self.depth, inks, marks, glow, self.opacity, self.tissue


class Anemone:
    def __init__(self, rng, texture, details):
        count = 73
        self.disc = Disc(texture, details)
        self.tubes = np.arange(1, count + 1, dtype=np.int32)[:, None, None]
        rank = (np.arange(count, dtype=np.float32) + .6) / count
        angle = np.arange(count) * 2.39996323 + rng.uniform(-.23, .23, count)
        self.ax = np.cos(angle)[:, None]
        self.az = np.sin(angle)[:, None]
        self.rank = rank[:, None]
        self.root = (29 + 148 * np.sqrt(rank))[:, None]
        self.height = (416 - 158 * rank + rng.uniform(-62, 57, count))[:, None]
        self.fan = (43 + 170 * rank + rng.uniform(-18, 44, count))[:, None]
        self.drift = rng.uniform(-37, 36, count)[:, None]
        self.phase = rng.uniform(0, TAU, count)[:, None]
        self.lag = rng.uniform(.80, 1.18, count)[:, None]
        self.thickness = rng.uniform(5.8, 11.7, count)[:, None]
        self.variation = rng.uniform(.0, 1.0, count)[:, None]
        self.burn = rng.uniform(.49, 1.03, count)[:, None]
        self.grain = rng.random((count, 125), dtype=np.float32)
        self.s = np.linspace(0, 1, 125, dtype=np.float32)[None, :]
        self.cross = np.float32((-1.22, -.83, -.47, -.11, .22, .57, .92, 1.24))

    def sway(self, time):
        s = self.s
        # A shared current bends the crown first. Each tip follows at its own pace.
        current = time * (1.18 * self.lag) - 3.4 * s
        lean = 39 * np.sin(current + .49 * self.phase)
        lean += 24 * np.sin(.65 * time - 4.4 * s + self.phase)
        twitch = 17 * np.sin(2.13 * time - 7.5 * s + 1.73 * self.phase)
        curl = (53 + 61 * self.rank) * smoothstep(.57, 1.10, s)
        curl *= .65 + .35 * np.sin(1.32 * time - 3.8 * s + self.phase)
        reach = self.root + self.fan * s ** 1.31 - curl
        side = self.drift * s ** 1.25 + (lean + twitch) * s ** 1.83
        bend = 45 * np.sin(self.phase - 1.1) * bell(s - .74, .24)
        side += bend * smoothstep(.19, .67, s) * (.62 + .38 * np.sin(time * 1.09 + self.phase))
        x = self.ax * reach - self.az * side + 21 * np.sin(.80 * time - 2.1 * s) * s ** 2
        z = .72 * self.az * reach + self.ax * side * .62
        y = DISC_Y - self.height * (1.22 * s - .24 * s * s)
        hang = 54 * self.rank ** 2 * (.7 + .3 * np.sin(self.phase + time * 1.18))
        y += .38 * curl + hang * smoothstep(.66, 1.0, s)
        y += 19 * np.sin(1.62 * time - 4.0 * s + self.phase) * s ** 2
        y -= .091 * (self.root - 93) * self.az
        return x * CROWN_WIDTH, DISC_Y + (y - DISC_Y) * CROWN_HEIGHT, z

    def paint(self, field, time):
        x, y, z = self.sway(time)
        px, py, depth = project(x, y, z)
        dx, dy = np.gradient(px, axis=1), np.gradient(py, axis=1)
        magnitude = np.maximum(.01, np.hypot(dx, dy))
        nx, ny = -dy / magnitude, dx / magnitude
        s = self.s
        bulb = bell(s - .937, .062)
        radius = self.thickness * (1.00 - .46 * s + .64 * bulb)
        radius *= 1.04 - .30 * smoothstep(.965, 1.002, s)
        radius *= 1 / (1 - depth / LENS)
        jitter = self.cross[None, None, :]
        offset = radius[..., None] * jitter
        xx = px[..., None] + nx[..., None] * offset
        yy = py[..., None] + ny[..., None] * offset
        front = np.sqrt(np.maximum(0, 1.12 - jitter * jitter))
        zz = depth[..., None] + front * radius[..., None] * .80
        # Light is strongest in the soft bulb and trails off toward the fixed root.
        tips = smoothstep(.61, .96, s)
        ripple = .70 + .30 * np.sin(time * .97 - s * 6.7 + self.phase)
        # Internal light drifts along the tissue on staggered phases.
        pulse_center = .53 + .17 * np.sin(.94 * time + self.phase)
        inner_light = bell(s - pulse_center, .205) * (.49 + .51 * self.variation)
        core = (.42 + .70 * s + .28 * tips + .22 * inner_light) * self.burn
        core *= (.68 + .32 * ripple) * (.79 + .21 * smoothstep(-240, 224, z))
        core += .23 * bulb * (.50 + .50 * np.sin(time * 1.51 + self.phase) ** 2)
        core += .21 * bell(s - .955, .041) * self.variation ** 3
        surface = core[..., None] * (.55 + .45 * front) * (1 - .19 * np.abs(jitter))
        surface *= (.70 + .30 * self.grain[..., None])
        surface *= np.where(jitter < 0, 1.08, .91)
        base_blue = np.float32((.43, .72, 1.0))
        blue = base_blue + (SKY - base_blue) * s[..., None]
        color = blue + tips[..., None] * self.variation[..., None] * (ICE - blue)
        white = smoothstep(.57, 1.13, surface)
        ink = surface[..., None] * (color[..., None, :] + white[..., None] * .84 * (PEARL - color[..., None, :]))
        directional = np.broadcast_to(stroke(dx, dy)[..., None], surface.shape)
        marks = np.where((s[..., None] > .87) & (surface > .57), MARK["o"], directional)
        marks = np.where((s[..., None] > .92) & (surface > .82) & (self.grain[..., None] > .71),
                         MARK["O"], marks)
        marks = np.where((surface < .31) | ((np.abs(jitter) > .82) & (self.grain[..., None] < .62)),
                         np.where(surface > .16, MARK[":"], MARK["."]), marks)
        glint = (surface > .81) & (self.grain[..., None] > .966)
        glint &= (s[..., None] > .47) & (inner_light[..., None] > .35)
        marks = np.where(glint, MARK["*"], marks)
        glow = np.broadcast_to(.77 + (.32 * tips + .19 * inner_light)[..., None], surface.shape)
        opacity = (.17 + .34 * front + .08 * bulb[..., None]) * np.ones_like(surface)
        tubes = np.broadcast_to(self.tubes, surface.shape)
        parts = (xx.ravel(), yy.ravel(), zz.ravel(), ink.reshape(-1, 3),
                 marks.ravel(), glow.ravel(), opacity.ravel(), tubes.ravel())
        base = self.disc.samples(time)
        field.composite_volume(*(np.concatenate((face, foot)) for face, foot in zip(parts, base)))


def seabed(texture, details):
    depth = np.full((ROWS, COLS), -650, dtype=np.float32)
    ink = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    marks = np.zeros((ROWS, COLS), dtype=np.uint8)
    field = CellScene(depth, ink, marks)
    stones = (
        (303, 1001, 276, 136, -173), (626, 1014, 311, 145, -195),
        (475, 1090, 324, 103, 34), (758, 1052, 203, 116, -14),
        (174, 1081, 175, 91, 11), (860, 1118, 186, 72, -22),
        (405, 1151, 228, 64, 186), (652, 1159, 256, 67, 116),
    )
    for x, y, width, height, z in stones:
        u, v = (PX - x) / width, (PY - y) / height
        rough = .045 * np.sin(16 * u + 11 * v) + .059 * np.sin(8.2 * u - 6.9 * v)
        inside = u * u + v * v + rough < 1
        front = np.sqrt(np.maximum(0, 1 - u * u - v * v))
        gleam = .46 + .49 * smoothstep(.49, .95, -v) + .16 * front
        ridges = .56 + .44 * np.sin(27 * u + 14 * v + 2 * np.sin(8 * v)) ** 4
        cyan_cast = .075 * bell(PX - 510, 300) * bell(PY - 1000, 136) * gleam
        light = (.041 + .124 * gleam * ridges + cyan_cast) * (.29 + .71 * texture)
        light *= .54 + .46 * bell(PX - 536, 386)
        rock_ink = light[..., None] * np.float32((.38, .57, .72))
        rock_marks = np.where(details > .91, MARK["-"], np.where(details > .64, MARK[":"], MARK["."]))
        field.splat(PX[inside], PY[inside], (z + 67 * front)[inside],
                    rock_ink[inside], rock_marks[inside], np.zeros(int(inside.sum())))

    return field


def swim_pose(time):
    orbit = -.68 + .506 * time + .105 * np.sin(1.19 * time)
    x = 315 * np.sin(orbit) + 19 * np.sin(2 * orbit - .31)
    z = 272 * np.cos(orbit) + 13
    y = -168 + 58 * np.cos(orbit) + 38 * np.sin(1.07 * time - .37)
    back = smoothstep(.17, .96, -np.cos(orbit))
    y -= 61 * back
    y += 53 * bell(np.cos(orbit) + .71, .26) * (.52 + .48 * np.sin(1.9 * time))
    y += 32 * bell(time - 7.65, .67)
    dx = 315 * np.cos(orbit) + 38 * np.cos(2 * orbit - .31)
    dz = -272 * np.sin(orbit)
    length = np.hypot(dx, dz)
    return x, y, z, dx / length, dz / length


def tail_beat(length, time):
    amplitude = smoothstep(-20, -140, length)
    return amplitude * 21 * np.sin(24.2 * time + .052 * length)


class Clownfish:
    def __init__(self, rng):
        length = np.linspace(-93, 92, 121, dtype=np.float32)
        around = np.linspace(0, TAU, 104, endpoint=False, dtype=np.float32)
        uu = length[:, None]
        ring = around[None, :]
        profile = [-93, -81, -57, -29, 0, 24, 48, 66, 80, 88, 92]
        height = np.interp(length, profile, [6, 16, 32, 44, 48, 46, 39, 30, 20, 12, 1])
        thickness = np.interp(length, profile, [4, 9, 17, 26, 30, 29, 26, 20, 16, 12, 1])
        midpoint = np.interp(length, profile, [1, 1, 2, 1, 0, 1, 3, 5, 6, 8, 9])
        self.u = np.broadcast_to(uu, (len(length), len(around)))
        self.v = np.float32(midpoint[:, None] + height[:, None] * np.cos(ring))
        self.w = np.float32(thickness[:, None] * np.sin(ring))
        self.grain = rng.random(self.u.shape, dtype=np.float32)
        self.skin, self.marks = self.pigment(uu, ring)

        a, b = np.meshgrid(np.linspace(0, 1, 38, dtype=np.float32),
                           np.linspace(-1, 1, 39, dtype=np.float32), indexing="ij")
        tail_u = -88 - a * (49 - 18 * (1 - np.abs(b)) ** 1.6)
        tail_v = b * (6 + 32 * smoothstep(0, 1, a))
        tail_w = 1.7 * np.cos(5 * b) * a
        outer = np.maximum(smoothstep(.92, 1.0, a), smoothstep(.88, 1.0, np.abs(b)))
        tail_light = (.58 + .29 * np.cos(23 * b + 4 * a) ** 10) * (1 - .78 * outer)
        self.fan = (tail_u, tail_v, tail_w, tail_light)

        self.ray = np.linspace(0, 1, 32, dtype=np.float32)[None, :]
        self.base = np.linspace(0, 1, 54, dtype=np.float32)[:, None]

    def pigment(self, u, ring):
        # Three white bands and their dark edges follow the curvature of the body.
        v = self.v
        side = np.sin(ring)
        stripe_a = np.abs(u - (49 + .21 * v + .0022 * v * v)) / (7.5 + .065 * np.maximum(v, 0))
        stripe_b = np.abs(u - (-15 + 13 * bell(v + 5, 34) - .09 * v)) / 9.4
        stripe_c = np.abs(u - (-76 + .12 * v)) / 5.7
        edges = np.minimum(np.minimum(stripe_a, stripe_b), stripe_c)
        white = 1 - smoothstep(.83, 1.16, edges)
        black = 1 - smoothstep(1.13, 1.67, edges)
        y_shade = .62 + .38 * smoothstep(-43, 8, v)
        light = .82 + .17 * np.abs(side) - .10 * smoothstep(72, 94, u)
        shade = y_shade * light * (.88 + .12 * self.grain)
        streak = bell(v - 10, 18) * bell(u - 21, 72) * .25
        orange = np.float32((1.0, .41, .075)) + streak[..., None] * np.float32((.05, .64, .13))
        dark = np.float32((.15, .10, .053))
        ivory = np.float32((.88, .96, 1.0))
        color = orange + black[..., None] * (dark - orange)
        color += white[..., None] * (ivory - color)
        color *= shade[..., None]
        color += (white * (.092 + .051 * np.cos(ring)))[..., None] * np.float32((.67, .94, 1.0))

        eye = ((u - 72) / 10.5) ** 2 + ((v + 8) / 9.6) ** 2
        eye *= 1 + 2.2 * (1 - np.abs(side))
        ring_light = 1 - smoothstep(.94, 1.36, eye)
        eye_core = 1 - smoothstep(.52, .81, eye)
        color += ring_light[..., None] * (np.float32((.76, .35, .09)) - color) * .72
        color += eye_core[..., None] * (np.float32((.021, .038, .055)) - color)
        reflection = bell(u - 73, 3.1) * bell(v + 12, 3.0) * eye_core
        color += reflection[..., None] * np.float32((.65, .76, .79))
        mouth = bell(v - 12, 2.3) * bell(u - 90, 5.8)
        color *= 1 - .60 * mouth[..., None]
        marks = np.where(white > .73, np.where(self.grain > .64, MARK["#"], MARK["+"]),
                         INKS[np.intp(np.clip(1 + 4.0 * shade + .54 * self.grain, 0, len(INKS) - 1))])
        marks = np.where((black > .63) & (white < .35), MARK["."], marks)
        marks = np.where(eye_core > .65, MARK["o"], marks)
        marks = np.where(reflection > .36, MARK["."], marks)
        return color.astype(np.float32), marks

    def plot(self, field, pose, time, u, v, w, color, marks, glow):
        x, y, z, dx, dz = pose
        lateral = w + tail_beat(u, time)
        u, v, lateral = 1.20 * u, 1.14 * v, 1.10 * lateral
        bank = .12 * np.sin(time * 1.96) + .055 * np.sin(time * 3.05)
        twist = .095 * np.sin(time * 1.54)
        sy = v * np.cos(bank) - lateral * np.sin(bank) + twist * u
        sw = v * np.sin(bank) + lateral * np.cos(bank)
        px, py, depth = project(x + dx * u - dz * sw, y + sy, z + dz * u + dx * sw)
        dim = .86 + .14 * smoothstep(-235, 190, depth)
        field.splat(px.ravel(), py.ravel(), depth.ravel(), (color * dim[..., None]).reshape(-1, 3),
                    marks.ravel(), np.full(px.size, glow, dtype=np.float32))

    def fins(self, field, time, pose):
        u, v, w, power = self.fan
        self.paint_fin(field, pose, time, u, v, w, power)
        t, r = self.base, self.ray
        shape = np.maximum(0, np.sin(np.pi * t)) ** .74
        base = -69 + 115 * t
        dorsal_u = base - 19 * r * shape
        dorsal_v = -(25 + 24 * shape + r * (10 + 13 * t) * shape)
        dorsal_w = (r * (2 + 6 * shape)) * np.sin(time * 10 - 6 * t)
        ray = .61 + .39 * np.sin(t * 84 + 3 * r) ** 6
        dorsal_light = (.34 + .45 * ray) * (1 - .61 * smoothstep(.86, 1, r))
        self.paint_fin(field, pose, time, dorsal_u, dorsal_v, dorsal_w, dorsal_light)

        fin_u = -59 + 72 * t - 17 * r * shape
        fin_v = 24 + 18 * shape + r * 16 * shape
        fin_w = r * 5 * np.sin(time * 14.7 - 6 * t)
        self.paint_fin(field, pose, time, fin_u, fin_v, fin_w, dorsal_light * .75)

        for side in (-1, 1):
            flap = .45 + .55 * np.sin(time * 18.1 + side) ** 2
            wing_u = 20 - r * (24 + 24 * t) + 7 * t
            wing_v = 9 + r * (2 + 33 * t)
            wing_w = side * (24 + r * (8 + 16 * flap) * (.39 + .61 * shape))
            wing_light = (.34 + .20 * ray) * (1 - .45 * smoothstep(.75, 1, r))
            self.paint_fin(field, pose, time, wing_u, wing_v, wing_w, wing_light)

    def paint_fin(self, field, pose, time, u, v, w, light):
        u, v, w, light = np.broadcast_arrays(u, v, w, light)
        tint = light[..., None] * np.float32((1.0, .41, .078))
        marks = np.where(light > .61, MARK["+"], np.where(light > .39, MARK[":"], MARK["."]))
        self.plot(field, pose, time, u, v, w, tint, marks, .015)

    def paint(self, field, time):
        pose = swim_pose(time)
        self.fins(field, time, pose)
        self.plot(field, pose, time, self.u, self.v, self.w, self.skin, self.marks, .035)


class Film:
    def __init__(self):
        rng = np.random.default_rng(291834)
        grain = rng.random((ROWS, COLS), dtype=np.float32)
        detail = rng.random((ROWS, COLS), dtype=np.float32)
        self.background = seabed(grain, detail)
        self.water = Water(rng)
        self.anemone = Anemone(rng, grain, detail)
        self.fish = Clownfish(rng)
        self.atlas = make_glyph_atlas()
        self.font = mono(23)
        self.panel = CodePanel()

        distant = bell(PX - 551, 506) * bell(PY - 648, 545)
        ground = bell(PX - 530, 402) * bell(PY - 1056, 235)
        shaft_x = PX - (60 + .54 * PY)
        shaft = bell(shaft_x, 109) * bell(PY - 275, 620)
        self.room = np.zeros((ROWS, COLS, 3), dtype=np.float32)
        self.room[:] = (.55, 2.1, 5)
        self.room += distant[..., None] * np.float32((1.3, 6.5, 11))
        self.room += ground[..., None] * np.float32((1.0, 4.1, 5.2))
        self.room += shaft[..., None] * np.float32((.4, 2.1, 4.3))

    def render(self, time):
        bed = self.background
        field = CellScene(bed.depth, bed.ink, bed.marks)
        field.light[:] = bed.light
        self.water.paint(field, time)
        self.fish.paint(field, time)
        self.anemone.paint(field, time)

        near_glow = cv2.GaussianBlur(field.light, (0, 0), 1.07)
        halo = cv2.GaussianBlur(field.light, (0, 0), 3.08)
        mist = cv2.GaussianBlur(field.light, (0, 0), 6.65)
        backing = self.room + 131 * near_glow + 144 * halo + 80 * mist
        backing += cv2.GaussianBlur(field.ink, (0, 0), .67) * 22
        backing = cv2.resize(np.uint8(np.clip(backing, 0, 255)), (WIDTH, ART_HEIGHT), interpolation=cv2.INTER_LINEAR)
        ink = field.glyph_pixels(self.atlas)
        ink = np.uint8(np.minimum(ink * 255, 255)).transpose(0, 2, 1, 3, 4).reshape(ART_HEIGHT, WIDTH, 3)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), "#080e16")
        canvas.paste(Image.fromarray(cv2.add(backing, ink)), (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((33, 29), "anemone.py", font=self.font, fill="#aacedc")
        draw.text((WIDTH - 33, 29), f"t {time:05.2f}s", font=self.font, fill="#9bbdca", anchor="ra")
        self.panel.draw(canvas, time)
        return canvas


def mono(size):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


SYNTAX = re.compile(r'(#.+$)|(\b(?:def|for|in|if|return|import|as|and|or|else|from)\b)|'
                    r'(\b\d+(?:\.\d+)?\b)|("[^"\n]*")|'
                    r'(\b[a-zA-Z_]\w*(?=\())|([=+*\-/<>])')
TOKEN_COLORS = ("#8992a5", "#c792e8", "#efaa75", "#a6c484", "#91b9de", "#91a7be")
LINE_HEIGHT = 29
PANEL_W, PANEL_H = 1020, 526
HEADER_H = 47


class CodePanel:
    def __init__(self):
        self.font = mono(24)
        self.advance = self.font.getlength("0")
        source = Path(__file__).read_text(encoding="utf-8").splitlines()
        self.sections = []
        for function in (Anemone.sway, Anemone.paint, Disc.samples, Film.render):
            lines, line_number = inspect.getsourcelines(function)
            start = max(0, line_number - 4)
            end = start + len(lines) + 15
            rows = []
            for number, line in enumerate(source[start:end], start + 1):
                indent = " " * (len(line) - len(line.lstrip()))
                wrapped = textwrap.wrap(
                    line.lstrip(), width=64, initial_indent=indent, subsequent_indent=indent + "    ",
                    break_long_words=False, break_on_hyphens=False,
                ) or [""]
                rows.append((str(number), wrapped[0]))
                rows.extend(("", continuation) for continuation in wrapped[1:])
            body = Image.new("RGB", (PANEL_W, len(rows) * LINE_HEIGHT), "#0b1520")
            draw = ImageDraw.Draw(body)
            for index, (number, line) in enumerate(rows):
                y = index * LINE_HEIGHT + 5
                draw.text((51, y), number, font=self.font, fill="#758094", anchor="rt")
                self.draw_line(draw, line, y)
            scrollable = sum(not number or int(number) < line_number + len(lines) for number, _ in rows)
            self.sections.append((body, scrollable))
        self.mask = Image.new("L", (PANEL_W, PANEL_H))
        ImageDraw.Draw(self.mask).rounded_rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), radius=17, fill=255)

    def draw_line(self, draw, content, y):
        cursor = 0
        for token in SYNTAX.finditer(content):
            self.draw_span(draw, content[cursor:token.start()], cursor, y, "#c3c9d5")
            self.draw_span(draw, token.group(), token.start(), y, TOKEN_COLORS[token.lastindex - 1])
            cursor = token.end()
        self.draw_span(draw, content[cursor:], cursor, y, "#c3c9d5")

    def draw_span(self, draw, text, column, y, color):
        draw.text((70 + column * self.advance, y), text, font=self.font, fill=color, anchor="lt")

    def draw(self, frame, time):
        panel = Image.new("RGB", (PANEL_W, PANEL_H), "#0b1520")
        part = time / (DURATION / len(self.sections))
        section = min(len(self.sections) - 1, int(part))
        lines, count = self.sections[section]
        scroll = min(20, max(2, count - 7)) * (part - section)
        top = round(scroll * LINE_HEIGHT)
        body = lines.crop((0, top, PANEL_W, top + PANEL_H - HEADER_H - 3))
        highlight = round((int(scroll) + 6 - scroll) * LINE_HEIGHT)
        ribbon = body.crop((0, highlight, PANEL_W, highlight + LINE_HEIGHT))
        tint = Image.new("RGB", ribbon.size, "#3c4a69")
        body.paste(Image.blend(ribbon, tint, 0.23), (0, highlight))
        panel.paste(body, (0, HEADER_H))

        draw = ImageDraw.Draw(panel)
        draw.line((0, HEADER_H - 1, PANEL_W, HEADER_H - 1), fill="#25404e", width=2)
        for index, color in enumerate(("#ff665c", "#ffbe4c", "#38c66d")):
            left = 18 + index * 24
            draw.ellipse((left, 17, left + 13, 30), fill=color)
        draw.text((91, 23), "anemone.py", font=self.font, fill="#d6dbe3", anchor="lm")
        draw.text((1001, 23), f"running   {FPS} fps", font=self.font, fill="#a4adbe", anchor="rm")
        draw.ellipse((763, 18, 774, 29), fill="#52ce73")
        draw.rounded_rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), radius=17, outline="#2d4655", width=2)
        frame.paste(panel, (30, 1364), self.mask)


def make_glyph_atlas():
    font = mono(16)
    masks = []
    for character in GLYPHS:
        mask = Image.new("L", (CELL_W, CELL_H))
        draw = ImageDraw.Draw(mask)
        left, top, right, bottom = font.getbbox(character)
        origin = ((CELL_W - right + left) / 2 - left, (CELL_H - bottom + top) / 2 - top)
        draw.text(origin, character, font=font, fill=255)
        if character == "-":
            draw.line((2, CELL_H // 2, CELL_W - 3, CELL_H // 2), fill=241)
        masks.append(np.asarray(mask, dtype=np.float32) / 255)
    return np.stack(masks)


def make_preview(film, output):
    samples = (.35, 1.80, 3.15, 5.34, 7.62, 10.45, 13.82, 17.89)
    sheet = Image.new("RGB", (1080, 1011), "#111a23")
    draw = ImageDraw.Draw(sheet)
    for index, time in enumerate(samples):
        frame = film.render(time)
        x, y = index % 4 * 270, index // 4 * 505
        draw.text((x + 14, y + 4), f"{time:05.2f}s", fill="white")
        sheet.paste(frame.resize((270, 480), Image.Resampling.LANCZOS), (x, y + 20))
        if index in (0, 3, 5, 6):
            frame.crop((79, 334, 1001, 1136)).save(output.with_name(f"{output.stem}-detail-{index}.png"))
    sheet.save(output, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    film = Film()
    if args.preview:
        make_preview(film, args.output)
        return

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to encode the video.")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "15",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-metadata", "comment=Procedural ASCII anemone and clownfish", str(args.output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for index in range(DURATION * FPS):
                encoder.stdin.write(film.render(index / FPS).tobytes())
                if (index + 1) % 150 == 0:
                    print(f"{(index + 1) // FPS}/{DURATION} seconds rendered", flush=True)
        finally:
            encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("FFmpeg could not encode the scene.")


if __name__ == "__main__":
    main()

