"""Look over Seoul's rooftops and down a lit avenue from above the city."""

import argparse
import inspect
import math
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1080, 1920
FPS, DURATION = 30, 20
CELL_W, CELL_H = 8, 10
COLS, ROWS = 135, 137
ART_HEIGHT = ROWS * CELL_H
GLYPHS = " .,:-~;=+*ox#@|/\\[]_"
SYMBOL = {glyph: index for index, glyph in enumerate(GLYPHS)}
X, Y = np.meshgrid(
    (np.arange(COLS, dtype=np.float32) + 0.5) * CELL_W,
    (np.arange(ROWS, dtype=np.float32) + 0.5) * CELL_H,
)
FOCAL, EYE, NEAR = 892, 167, 35
VANISH_X, VANISH_Y = 566, 420
STREETS = (84, 149, 224, 308, 414, 542, 693, 869, 1082, 1335, 1640, 1972, 2364, 2840)
ROAD_HALF = 17
WINDOW_COLOR = np.float32((
    (237, 173, 95), (248, 214, 155), (154, 206, 247), (119, 190, 195), (188, 168, 224),
    (234, 112, 103),
))


def clamp(value):
    return np.clip(value, 0, 1)


def smoothstep(value):
    value = clamp(value)
    return value * value * (3 - 2 * value)


def hash_grid(x, y):
    x = np.asarray(x, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)
    value = (x * np.uint32(1597334677)) ^ (y * np.uint32(3812015801))
    value ^= value >> np.uint32(16)
    value = np.multiply(value, np.uint32(2246822519), dtype=np.uint32)
    value ^= value >> np.uint32(13)
    return (value & np.uint32(65535)).astype(np.float32) / 65535


def noise(x, y):
    """Interpolate neighboring values so the flame can flow between frames."""
    ix, iy = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    dx, dy = smoothstep(x - ix), smoothstep(y - iy)
    low = hash_grid(ix, iy) * (1 - dx) + hash_grid(ix + 1, iy) * dx
    high = hash_grid(ix, iy + 1) * (1 - dx) + hash_grid(ix + 1, iy + 1) * dx
    return low * (1 - dy) + high * dy


def project(x, z, height):
    """A high camera sees rooftops and compresses the far end of the avenue."""
    scale = FOCAL / (z + NEAR)
    return VANISH_X + x * scale, VANISH_Y + (EYE - height) * scale


def avenue(z):
    return -20.5 - 0.080 * z + 21 * np.sin(0.012 * z + 0.10) * np.exp(-np.asarray(z) / 960)


def sky():
    """Keep Namsan's silhouette visible against the last violet light."""
    field = noise(X * 0.007 + 51, Y * 0.015 + 24)
    grain = noise(X * 0.097 + 18, Y * 0.094 + 62)
    amber = np.exp(-((X - 333) / 573) ** 2 - ((Y - 449) / 214) ** 2)
    blue = np.exp(-((X - 802) / 626) ** 2 - ((Y - 297) / 327) ** 2)
    horizon = np.exp(-((Y - 494) / 144) ** 2)
    base = np.full((*X.shape, 3), (5, 8, 17), dtype=np.float32)
    base += (amber * (0.46 + 0.54 * field))[..., None] * np.float32((31, 16, 19))
    base += (blue * (0.41 + 0.59 * field))[..., None] * np.float32((8, 19, 36))
    base += (horizon * 0.45)[..., None] * np.float32((18, 11, 10))
    points = (0.40 * field + 0.60 * grain) * (0.24 + 0.52 * horizon)
    marks = np.select([points > 0.50, points > 0.31, points > 0.16],
                      [SYMBOL["-"], SYMBOL[":"], SYMBOL["."]], default=0)
    rgb = points[..., None] * (np.float32((39, 43, 61)) + amber[..., None] * np.float32((29, 7, 2)))

    ridge_x = np.float32((0, 135, 273, 393, 492, 585, 654, 701, 743, 838, 972, 1080))
    ridge = np.interp(X[0], ridge_x, (552, 536, 550, 530, 499, 472, 454, 468, 486, 530, 517, 510))
    ridge = ridge[None, :] + 3.5 * np.sin(X * 0.066) + 5.5 * np.sin(X * 0.026)
    hill = Y > ridge
    back = (Y > 518 + 25 * np.sin(X * 0.011 + 0.31)) & ~hill
    marks[back], rgb[back], base[back] = SYMBOL[":"], (19, 26, 44), (10, 16, 27)
    marks[hill] = np.where(grain[hill] > 0.42, SYMBOL[":"], SYMBOL["."])
    rgb[hill] = (0.64 + 0.36 * field[hill, None]) * np.float32((29, 40, 53))
    base[hill] = (7, 11, 20)
    return marks, rgb, base


def namsan_tower(marks, rgb, base, time):
    """A warmly lit observation deck sits above the cool narrow shaft."""
    axis = 654
    stem = (np.abs(X - axis) < (7.8 + 0.052 * (Y - 306))) & (Y > 304) & (Y < 462)
    mast = (np.abs(X - axis) < 4.1) & (Y >= 188) & (Y < 275)
    mast_base = (np.abs(X - axis) < 10.8) & (Y >= 252) & (Y < 277)
    cap = (np.abs(X - axis) < 25.5) & (Y >= 277) & (Y < 313)
    belt = (np.abs(X - axis) < 20) & (Y >= 313) & (Y < 326)
    for section, label, shade in (
        (stem, "|", (105, 181, 207)), (mast, "|", (161, 195, 210)),
        (mast_base, "#", (169, 188, 192)), (cap, "[", (239, 219, 171)),
        (belt, "=", (151, 213, 224)),
    ):
        marks[section], rgb[section] = SYMBOL[label], shade
        base[section] = np.float32(shade) * 0.19
    right = cap & (np.arange(COLS)[None, :] % 2 == 0)
    marks[right] = SYMBOL["]"]
    ring = cap & (Y < 288)
    marks[ring], rgb[ring] = SYMBOL["="], (200, 233, 233)
    column = int(axis / CELL_W)
    beacon = 0.47 + 0.53 * (0.5 + 0.5 * math.cos(time * 3.24)) ** 5
    marks[18, column] = SYMBOL["*"]
    rgb[18, column] = np.float32((255, 103, 86)) * beacon
    base[18, column] += np.float32((57, 19, 16)) * beacon


def street_grid(marks, rgb, base):
    """Mark the avenue and cross streets from their position on the ground."""
    z = FOCAL * EYE / np.maximum(Y - VANISH_Y, 1) - NEAR
    x = (X - VANISH_X) * (z + NEAR) / FOCAL
    terrain = (Y > VANISH_Y) & (z < 2760 + 420 * np.sin(X * 0.009 + 0.40))
    way = x - avenue(z)
    center_road = terrain & (np.abs(way) < ROAD_HALF)
    cross = np.min(np.abs(z[..., None] - np.float32(STREETS)), axis=-1)
    street = terrain & (cross < 7.0) & (np.abs(x) < 120 + z * 0.65)
    paved = center_road | street
    grit = noise(X * 0.068 + 65, Y * 0.080 + 2)
    asphalt = (0.66 + 0.51 * grit) * (0.65 + 0.35 * np.exp(-z / 900))
    turf = terrain & ~paved
    marks[turf] = np.where(grit[turf] > 0.49, SYMBOL[":"], SYMBOL["."])
    rgb[turf] = (0.65 + 0.45 * grit[turf, None]) * np.float32((14, 21, 29))
    base[turf] = (6, 8, 14)
    marks[paved] = np.where(center_road[paved], SYMBOL["."], SYMBOL["-"])
    rgb[paved] = asphalt[paved, None] * np.float32((53, 44, 43))
    base[paved] = asphalt[paved, None] * np.float32((19, 16, 22))
    median = center_road & (np.abs(way) < 1.4 + z * 0.0017) & (z % 28 < 15) & ~street
    marks[median], rgb[median] = SYMBOL["|"], (107, 82, 53)
    curb = center_road & (np.abs(way) > ROAD_HALF - 1.7 - 0.0013 * z)
    marks[curb], rgb[curb] = SYMBOL[":"], (92, 76, 58)
    return marks, rgb, base


class Rooftops:
    def __init__(self):
        self.rng = np.random.default_rng(24777)
        self.faces = np.zeros((ROWS, COLS), dtype=np.uint8)
        self.ids = np.full((ROWS, COLS), -1, dtype=np.int16)
        self.depth = np.full((ROWS, COLS), np.inf, dtype=np.float32)
        self.marks = np.zeros((ROWS, COLS), dtype=np.intp)
        self.rgb = np.zeros((ROWS, COLS, 3), dtype=np.float32)
        self.base = np.zeros_like(self.rgb)
        self.candidates = []
        self.glint = noise(X * 0.071 + 91, Y * 0.079 + 54)
        self.facet = noise(X * 0.022 + 117, Y * 0.055)
        for index, (x, z, w, d, h, color) in enumerate(self.layout()):
            self.block(index, x, z, w, d, h, color)
        windows = [(row,col,when,gain,color,glyph) for owner,row,col,when,gain,color,glyph in self.candidates
                   if self.ids[row,col]==owner and self.faces[row,col] in (1,2)]
        _, unique = np.unique(np.array(windows)[:, :2], axis=0, return_index=True)
        self.rows, self.cols, self.start, self.power, shades, glyphs = np.float32(windows)[unique].T
        self.rows, self.cols = self.rows.astype(np.intp), self.cols.astype(np.intp)
        self.ink, self.glyphs = WINDOW_COLOR[shades.astype(np.intp)], glyphs.astype(np.intp)
        self.visible = self.ids >= 0

    def layout(self):
        """Leave the same avenue open as the blocks stack up toward the horizon."""
        specs = []
        for band, (start, stop) in enumerate(zip(STREETS, STREETS[1:])):
            for side in (-1, 1):
                z = start + 10
                depth = (stop - start - 13) * 0.68
                middle = float(avenue(z + depth / 2))
                edge = middle + side * (ROAD_HALF + 7 + abs(float(avenue(z) - middle)))
                width_limit = (z + NEAR) * 0.74 + 93
                while abs(edge) < width_limit:
                    width = self.rng.uniform(30, 67) if band < 4 else self.rng.uniform(48, 109)
                    x = edge - width if side < 0 else edge
                    z0 = z + self.rng.uniform(0, min(13, depth / 2))
                    rim = max(0, min(1, (abs(edge - middle) - 45) / 193))
                    base = 31 + 60 * self.rng.random() + rim * (24 + self.rng.random() * 48)
                    height = base + (26 if 4 <= band < 8 and self.rng.random() > 0.70 else 0)
                    if abs(edge - middle) < 45:
                        height *= 0.84 if band < 4 else 1.0
                    style = int(self.rng.choice(5, p=(0.27,0.27,0.26,0.13,0.07)))
                    specs.append((x, z0, width, depth * self.rng.uniform(0.70, 1.02), height, style))
                    edge += side * (width + self.rng.uniform(10, 23))
        specs.extend(((-318, 836, 94, 52, 225, 2), (-234, 686, 58, 61, 204, 1),
                      (205, 757, 83, 62, 226, 2), (341, 984, 85, 79, 202, 4),
                      (-144, 131, 82, 42, 123, 3), (74, 115, 67, 36, 107, 2)))
        return sorted(specs, key=lambda spec: spec[1], reverse=True)

    def polygon(self, index, depth, corners, kind, hue):
        projected = np.float32([project(*point) for point in corners])
        if projected[:,0].max() < 0 or projected[:,0].min() >= WIDTH:
            return
        if projected[:,1].max() < 0 or projected[:,1].min() >= ART_HEIGHT:
            return
        grid = np.rint(projected / (CELL_W,CELL_H) - 0.5).astype(np.int32)
        silhouette = np.zeros(X.shape, dtype=np.uint8)
        cv2.fillConvexPoly(silhouette, grid, 255)
        face = silhouette > 0
        border = np.zeros_like(silhouette)
        cv2.polylines(border, [grid], True, 255, 1, cv2.LINE_8)
        edge = (border > 0) & face
        texture = self.glint * 0.58 + self.facet * 0.42
        air = 0.54 + 0.46 * np.exp(-depth / 1035)
        self.faces[face], self.ids[face], self.depth[face] = kind, index, depth
        if kind == 3:
            self.marks[face] = np.where(texture[face] > .52, SYMBOL[":"], SYMBOL["."])
            self.marks[edge] = SYMBOL["_"]
        elif kind == 2:
            self.marks[face] = np.where(texture[face] > .49, SYMBOL["|"], SYMBOL["."])
            self.marks[edge] = SYMBOL["/"] if np.mean(projected[:,0]) > VANISH_X else SYMBOL[chr(92)]
        else:
            self.marks[face] = np.where(texture[face] > .60, SYMBOL[":"], SYMBOL["."])
            self.marks[edge] = SYMBOL["|"]
        self.rgb[face] = hue * (0.48 + 0.54 * texture[face,None]) * air
        self.rgb[edge] += np.float32((28, 37, 53)) * air
        self.base[face] = np.float32((5.8, 8.8, 17.0)) + self.rgb[face] * 0.11

    def block(self, owner, x, z, width, depth, height, color):
        shade = np.float32((32, 47, 72)) + WINDOW_COLOR[color] * 0.092
        front = ((x,z,0),(x+width,z,0),(x+width,z,height),(x,z,height))
        side_x = x+width if x+width<0 else x
        side = ((side_x,z,0),(side_x,z+depth,0),(side_x,z+depth,height),(side_x,z,height))
        self.polygon(owner,z,front,1,shade)
        if x*x+width*x >= 0:
            self.polygon(owner,z,side,2,shade * np.float32((0.56,0.69,0.84)))
        if height < EYE:
            roof=((x,z,height),(x+width,z,height),(x+width,z+depth,height),(x,z+depth,height))
            self.polygon(owner,z,roof,3,shade * np.float32((0.80,0.93,1.02)))
        for elevation in np.arange(9,height-4,7.5 + 1.4*(owner%3)):
            power = self.rng.uniform(.52,1.0)
            lights = self.rng.uniform(-8.0,13.5)
            facade = ((px,z) for px in np.arange(x+3.8,x+width-2.4,7.2))
            side_wall = ((side_x,pz) for pz in np.arange(z+4,z+depth-2,9.3)) if x*x+width*x>=0 else ()
            for px,pz in (*facade,*side_wall):
                if self.rng.random() > (0.78 if z < 980 else 0.68):
                    continue
                sx,sy=project(px,pz,elevation)
                row,col=int(sy//CELL_H),int(sx//CELL_W)
                if not (0<=row<ROWS and 0<=col<COLS):
                    continue
                strength=power*self.rng.uniform(0.68,1.19)
                hue=color if self.rng.random()>.16 else int(self.rng.integers(0,6))
                glyph = SYMBOL["[" if col%2==0 else "]"] if z<700 else SYMBOL[":"]
                self.candidates.append((owner,row,col,lights+self.rng.uniform(-1.8,2.4),strength,hue,glyph))

    def light(self, marks, rgb, base, time):
        """Switch on whole floors, keeping the front buildings in front."""
        visible=self.visible
        marks[visible],rgb[visible],base[visible]=self.marks[visible],self.rgb[visible],self.base[visible]
        power=self.power*smoothstep((time-self.start)/1.08)
        active=power>.095
        rows,cols=self.rows[active],self.cols[active]
        ink=self.ink[active] * (power[active,None] * 1.13)
        marks[rows,cols]=np.where(power[active]>0.37,self.glyphs[active],SYMBOL["."])
        rgb[rows,cols]=ink
        base[rows,cols]+=ink*.14
        return int(np.count_nonzero(active))


class Traffic:
    def __init__(self):
        rng=np.random.default_rng(7717)
        self.period = np.float32((250, 440, 707, 935, 1590))
        count=(17,19,18,14,12)
        self.z0 = np.concatenate([rng.uniform(0,length,n) for length,n in zip(self.period,count)]).astype(np.float32)
        self.distance=np.repeat(self.period,count)
        self.origin=np.repeat(np.float32((67,187,392,692,1010)),count)
        self.pick=rng.uniform(-1,1,len(self.z0)).astype(np.float32)
        self.speed=rng.uniform(16,40,len(self.z0)).astype(np.float32)*np.where(self.pick>0,1,-1)
        self.cross_z=np.float32((146,222,306,415,544,690,868,1084))
        self.cross_offset=rng.uniform(0,490,len(self.cross_z)).astype(np.float32)
        self.cross_speed=np.float32((-26,31,22,-37,-31,39,42,-33))

    def flows(self,time):
        """Project advancing headlights and red tail lights onto both streets."""
        z=self.origin+(self.z0+self.speed*time)%self.distance
        x=avenue(z)+np.where(self.speed<0,-7,7)
        rows_x=avenue(self.cross_z)+(self.cross_offset+self.cross_speed*time)%490-245
        z_all=np.r_[z,self.cross_z]
        sx,sy=project(np.r_[x,rows_x],z_all,np.zeros_like(z_all))
        head=np.r_[self.speed<0,self.cross_speed>0]
        return sx,sy,z_all,head

    def plot(self,marks,rgb,base,scene_depth,sx,sy,z,head,trail=False):
        col,row=(sx//CELL_W).astype(np.intp),(sy//CELL_H).astype(np.intp)
        inside=(col>=0)&(col<COLS)&(row>=0)&(row<ROWS)
        col,row,z,head=col[inside],row[inside],z[inside],head[inside]
        seen=z<scene_depth[row,col]+4
        col,row,z,head=col[seen],row[seen],z[seen],head[seen]
        hue = np.where(head[:,None],np.float32((255,240,192)),np.float32((249,94,79)))
        intensity=(0.66+0.34*np.exp(-z/710))*(0.32 if trail else 1)
        light=hue*intensity[:,None]
        marks[row,col]=np.where(trail,SYMBOL["-"],np.where(z<338,SYMBOL["*"],SYMBOL["+"]))
        rgb[row,col]=np.maximum(rgb[row,col],light)
        base[row,col]+=light*(0.063 if trail else 0.24)

    def draw(self, marks, rgb, base, scene_depth, time):
        for delay in (0.13,0.0):
            self.plot(marks,rgb,base,scene_depth,*self.flows(time-delay),trail=delay>0)


class Skyline:
    def __init__(self):
        self.marks,self.rgb,self.base=street_grid(*sky())
        self.buildings=Rooftops()
        self.traffic=Traffic()
        self.panel=CodePanel()
        self.atlas=make_glyph_atlas()
        self.font=mono(23)
        self.add_lamps()

    def add_lamps(self):
        z=np.arange(92,1880,35,dtype=np.float32)
        for side in (-1,1):
            sx,sy=project(avenue(z)+side*(ROAD_HALF+2.3),z,np.zeros_like(z))
            for x,y,depth in zip(sx,sy,z):
                col,row=int(x//CELL_W),int(y//CELL_H)
                if 0<=col<COLS and 0<=row<ROWS and depth<self.buildings.depth[row,col]+3:
                    strength=.47+.45*math.exp(-depth/590)
                    self.marks[row,col]=SYMBOL["+"] if depth<508 else SYMBOL["."]
                    self.rgb[row,col]=np.float32((240,170,104))*strength
                    self.base[row,col]+=np.float32((52,24,10))*strength

    def render(self,time):
        marks,rgb,base=self.marks.copy(),self.rgb.copy(),self.base.copy()
        namsan_tower(marks,rgb,base,time)
        count=self.buildings.light(marks,rgb,base,time)
        self.traffic.draw(marks,rgb,base,self.buildings.depth,time)
        glow=cv2.resize(base,(WIDTH,ART_HEIGHT),interpolation=cv2.INTER_LINEAR)
        ink=self.atlas[marks][...,None]*rgb[:,:,None,None,:]
        ink=ink.transpose(0,2,1,3,4).reshape(ART_HEIGHT,WIDTH,3)
        frame=Image.new("RGB",(WIDTH,HEIGHT),(5,8,17))
        frame.paste(Image.fromarray(np.uint8(np.clip(glow+ink,0,255))),(0,0))
        draw=ImageDraw.Draw(frame)
        draw.text((33,30),"seoul.py",font=self.font,fill="#dfe4ee")
        draw.text((33,60),f"windows {count:04d}",font=self.font,fill="#a2b1cd")
        draw.text((WIDTH-34,30),f"t {time:05.2f}s",font=self.font,fill="#a2b1cd",anchor="ra")
        self.panel.draw(frame,time)
        return frame



def mono(size):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def make_glyph_atlas():
    font = mono(13)
    atlas = []
    for character in GLYPHS:
        mask = Image.new("L", (CELL_W, CELL_H))
        draw = ImageDraw.Draw(mask)
        draw.text((CELL_W / 2, 9), character, font=font, anchor="ms", fill=255)
        atlas.append(np.asarray(mask, dtype=np.float32) / 255)
    return np.stack(atlas)


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
        for function in (project, Rooftops.light, Traffic.flows, Traffic.plot):
            lines, line_number = inspect.getsourcelines(function)
            start = max(0, line_number - 4)
            end = start + len(lines) + 15
            body = Image.new("RGB", (PANEL_W, (end - start) * LINE_HEIGHT), "#10131a")
            draw = ImageDraw.Draw(body)
            for index, line in enumerate(source[start:end]):
                y = index * LINE_HEIGHT + 5
                draw.text((51, y), str(start + index + 1), font=self.font, fill="#758094", anchor="rt")
                self.draw_line(draw, line[:66], y)
            self.sections.append((body, len(lines)))
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
        panel = Image.new("RGB", (PANEL_W, PANEL_H), "#10131a")
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
        draw.line((0, HEADER_H - 1, PANEL_W, HEADER_H - 1), fill="#292e3b", width=2)
        for index, color in enumerate(("#ff665c", "#ffbe4c", "#38c66d")):
            left = 18 + index * 24
            draw.ellipse((left, 17, left + 13, 30), fill=color)
        draw.text((91, 23), "seoul.py", font=self.font, fill="#d6dbe3", anchor="lm")
        draw.text((1001, 23), "running   30 fps", font=self.font, fill="#a4adbe", anchor="rm")
        draw.ellipse((763, 18, 774, 29), fill="#52ce73")
        draw.rounded_rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), radius=17, outline="#333b4b", width=2)
        frame.paste(panel, (30, 1364), self.mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, help="Preview one frame as a PNG.")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    scene = Skyline()
    if args.frame is not None:
        if not 0 <= args.frame < FPS * DURATION:
            parser.error("--frame must be between 0 and 599.")
        scene.render(args.frame / FPS).save(args.output)
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("FFmpeg was not found on PATH.")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for index in range(FPS * DURATION):
                encoder.stdin.write(scene.render(index / FPS).tobytes())
                if (index + 1) % 150 == 0:
                    print(f"{(index + 1) // FPS:02d}/{DURATION}s rendered", flush=True)
        finally:
            encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit("FFmpeg could not encode the video.")


if __name__ == "__main__":
    main()
