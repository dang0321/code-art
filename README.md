# code art

> Python 코드로 그린 ASCII 코드 아트의 소스코드와 음원
> Source code and music for ASCII code-art videos drawn with Python

▶️ **YouTube** [@Dot.atelier](https://www.youtube.com/@Dot.atelier) · 📸 **Instagram** [@code_art.ascii](https://www.instagram.com/code_art.ascii/)

모든 작품은 1080×1920 세로 화면, 30fps, 20초 길이의 음악이 있는 영상이며, 완성 영상은 유튜브와 인스타그램에서 볼 수 있습니다.
Every piece is a 1080×1920 (vertical), 30 fps, 20-second video with music. The finished videos are on YouTube and Instagram.

---

## 이 저장소를 만든 이유 · Why this repo exists

**한국어**
[인스타그램 릴스](https://www.instagram.com/code_art.ascii/)와 [유튜브 쇼츠](https://www.youtube.com/@Dot.atelier)로 올리는 ASCII 코드 아트 시리즈의 소스코드와 음원을 한곳에 모아두려고 만들었습니다.
영상 속 장면은 그림 도구가 아니라 코드로 그렸고, 화면에 흐르는 코드는 그 장면을 만든 실제 스크립트입니다.
"저건 어떻게 움직이는 걸까?" 하고 궁금해진 분들이 직접 코드를 읽고 실행해 볼 수 있도록 소스코드와 음원을 공개합니다. 새 작품은 영상이 올라간 뒤에 이곳에 추가됩니다.

**English**
This repo collects the source code and music behind an ASCII code-art series posted as [Instagram Reels](https://www.instagram.com/code_art.ascii/) and [YouTube Shorts](https://www.youtube.com/@Dot.atelier).
Every scene is drawn with code rather than a drawing tool, and the code scrolling on screen is the actual script that renders it.
The code and music are shared so anyone who wonders how a piece works can read the code and run it. New pieces are added here after their video is posted.

---

## 저장소에 있는 것 · What's inside

```
source_code/<작품 이름>/<스크립트>.py      영상을 그리는 코드          script that renders the video
음원/<작품 이름>/<작품 이름>_bgm.py        음원을 합성하는 코드        script that synthesises the music
음원/<작품 이름>/<작품 이름>_bgm.wav       음원                       the music
watermark.png, terminal-cursor-profile.png  로고 이미지              logo images
```

`음원` 폴더는 "music(음원)"이라는 뜻입니다. · The `음원` folder means "music".

### 작품 목록 · Works (22)

| 작품 · Work | 영상 · Video | 소스코드 · Source | 음원 · Music |
|---|---|---|---|
| anemone and clownfish | [▶ Shorts](https://www.youtube.com/shorts/F0yzZOQ6EIk) | [anemone.py](<source_code/anemone and clownfish/anemone.py>) | [bgm.py](<음원/anemone and clownfish/anemone and clownfish_bgm.py>) · [wav](<음원/anemone and clownfish/anemone and clownfish_bgm.wav>) |
| aurora | [▶ Shorts](https://www.youtube.com/shorts/PBRTtChh7hw) | [aurora.py](<source_code/aurora/aurora.py>) | [bgm.py](<음원/aurora/aurora_bgm.py>) · [wav](<음원/aurora/aurora_bgm.wav>) |
| beach afternoon | [▶ Shorts](https://www.youtube.com/shorts/zEkF6x9Wp7E) | [beach.py](<source_code/beach afternoon/beach.py>) | [bgm.py](<음원/beach afternoon/beach afternoon_bgm.py>) · [wav](<음원/beach afternoon/beach afternoon_bgm.wav>) |
| beach sunlit | [▶ Shorts](https://www.youtube.com/shorts/xH7t_6IMTQg) | [beach.py](<source_code/beach sunlit/beach.py>) | [bgm.py](<음원/beach sunlit/beach sunlit_bgm.py>) · [wav](<음원/beach sunlit/beach sunlit_bgm.wav>) |
| comet | [▶ Shorts](https://www.youtube.com/shorts/7Md7C2y3T1o) | [meteors.py](<source_code/comet/meteors.py>) | [bgm.py](<음원/comet/comet_bgm.py>) · [wav](<음원/comet/comet_bgm.wav>) |
| dew web | [▶ Shorts](https://www.youtube.com/shorts/jy1DnBng2TQ) | [dew.py](<source_code/dew web/dew.py>) | [bgm.py](<음원/dew web/dew web_bgm.py>) · [wav](<음원/dew web/dew web_bgm.wav>) |
| fireflies | [▶ Shorts](https://www.youtube.com/shorts/oqQchUUGo04) | [fireflies.py](<source_code/fireflies/fireflies.py>) | [bgm.py](<음원/fireflies/fireflies_bgm.py>) · [wav](<음원/fireflies/fireflies_bgm.wav>) |
| hanabi | [▶ Shorts](https://www.youtube.com/shorts/-OVcy_TaeRw) | [hanabi.py](<source_code/hanabi/hanabi.py>) | [bgm.py](<음원/hanabi/hanabi_bgm.py>) · [wav](<음원/hanabi/hanabi_bgm.wav>) |
| hanabi2 | [▶ Shorts](https://www.youtube.com/shorts/ToIlcQ23M9Q) | [hanabi2.py](<source_code/hanabi2/hanabi2.py>) | [bgm.py](<음원/hanabi2/hanabi2_bgm.py>) · [wav](<음원/hanabi2/hanabi2_bgm.wav>) |
| jelly fish | [▶ Shorts](https://www.youtube.com/shorts/XYw-FMAamtI) | [jellyfish.py](<source_code/jelly fish/jellyfish.py>) | [bgm.py](<음원/jelly fish/jelly fish_bgm.py>) · [wav](<음원/jelly fish/jelly fish_bgm.wav>) |
| jellyfish2 | [▶ Shorts](https://www.youtube.com/shorts/RWrmG4KUW3w) | [jellyfish2.py](<source_code/jellyfish2/jellyfish2.py>) | [bgm.py](<음원/jellyfish2/jellyfish2_bgm.py>) · [wav](<음원/jellyfish2/jellyfish2_bgm.wav>) |
| jellyfish 3 | [▶ Shorts](https://www.youtube.com/shorts/TK5RwgV6k08) | [jellyfish2.py](<source_code/jellyfish 3/jellyfish2.py>) | [bgm.py](<음원/jellyfish 3/jellyfish 3_bgm.py>) · [wav](<음원/jellyfish 3/jellyfish 3_bgm.wav>) |
| jellyfish 4 | [▶ Shorts](https://www.youtube.com/shorts/YVWisoi9nFY) | [jellyfish4.py](<source_code/jellyfish 4/jellyfish4.py>) | [bgm.py](<음원/jellyfish 4/jellyfish 4_bgm.py>) · [wav](<음원/jellyfish 4/jellyfish 4_bgm.wav>) |
| plankton | [▶ Shorts](https://www.youtube.com/shorts/Menp5frA-K8) | [plankton.py](<source_code/plankton/plankton.py>) | [bgm.py](<음원/plankton/plankton_bgm.py>) · [wav](<음원/plankton/plankton_bgm.wav>) |
| plasma ball | [▶ Shorts](https://www.youtube.com/shorts/nvAyd3_fDBU) | [plasma_ball.py](<source_code/plasma ball/plasma_ball.py>) | [bgm.py](<음원/plasma ball/plasma ball_bgm.py>) · [wav](<음원/plasma ball/plasma ball_bgm.wav>) |
| rainy window | [▶ Shorts](https://www.youtube.com/shorts/otlhOtxHKQk) | [rainwindow.py](<source_code/rainy window/rainwindow.py>) | [bgm.py](<음원/rainy window/rainy window_bgm.py>) · [wav](<음원/rainy window/rainy window_bgm.wav>) |
| sakura | [▶ Shorts](https://www.youtube.com/shorts/yuRYW6-1_eY) | [sakura.py](<source_code/sakura/sakura.py>) | [bgm.py](<음원/sakura/sakura_bgm.py>) · [wav](<음원/sakura/sakura_bgm.wav>) |
| seoul night | [▶ Shorts](https://www.youtube.com/shorts/nP22EsIEPv4) | [seoul.py](<source_code/seoul night/seoul.py>) | [bgm.py](<음원/seoul night/seoul night_bgm.py>) · [wav](<음원/seoul night/seoul night_bgm.wav>) |
| snow | [▶ Shorts](https://www.youtube.com/shorts/qG_dCg5hQ54) | [snowflake.py](<source_code/snow/snowflake.py>) | [bgm.py](<음원/snow/snow_bgm.py>) · [wav](<음원/snow/snow_bgm.wav>) |
| star trails | [▶ Shorts](https://www.youtube.com/shorts/MXVMPfSaPMU) | [star_trails.py](<source_code/star trails/star_trails.py>) | [bgm.py](<음원/star trails/star trails_bgm.py>) · [wav](<음원/star trails/star trails_bgm.wav>) |
| tokyo night | [▶ Shorts](https://www.youtube.com/shorts/-CmEH1_EVRI) | [tokyo.py](<source_code/tokyo night/tokyo.py>) | [bgm.py](<음원/tokyo night/tokyo night_bgm.py>) · [wav](<음원/tokyo night/tokyo night_bgm.wav>) |
| volcano | [▶ Shorts](https://www.youtube.com/shorts/tFDgneN8bGY) | [volcano.py](<source_code/volcano/volcano.py>) | [bgm.py](<음원/volcano/volcano_bgm.py>) · [wav](<음원/volcano/volcano_bgm.wav>) |

### 직접 실행해 보기 · Running the code

**한국어**
- 필요한 것: Python 3, numpy, Pillow, opencv-python(일부 스크립트), 그리고 PATH에 등록된 ffmpeg
- 영상 스크립트는 화면을, 음원 스크립트는 소리를 만듭니다. 둘을 합친 완성 영상은 위 표의 링크에서 볼 수 있습니다.
- 스크립트마다 실행 인자가 조금 다릅니다. 각 파일 맨 아래 `if __name__ == "__main__":` 부분이나 `--help`를 참고해 주세요.

**English**
- Requirements: Python 3, numpy, Pillow, opencv-python (some scripts), and ffmpeg on your PATH
- Video scripts render the picture and music scripts synthesise the sound. The finished pieces that combine both are linked in the table above.
- Arguments differ slightly between scripts. See the `if __name__ == "__main__":` block at the bottom of each file, or run it with `--help`.

```bash
pip install numpy pillow opencv-python
python source_code/aurora/aurora.py        # aurora.mp4 (무음 · silent)
```

---

## 저작권 안내 · Copyright

© 2026 dang0321. All rights reserved.

**한국어**
이 저장소의 음원과 소스코드, 그리고 이 코드로 만든 영상은 별도의 오픈소스 라이선스 없이 공개되어 있으며, 저작권은 제작자에게 있습니다.
공개 저장소이므로 GitHub 안에서 보거나 포크하는 것은 가능하지만, 그 밖의 이용은 아래 내용을 꼭 지켜주세요.

- 개인 학습과 참고를 위해 코드를 읽고 실행해 보는 것은 환영합니다.
- 영상과 음원을 허락 없이 다른 곳에 다시 올리거나, 편집해서 올리지 말아 주세요.
- 상업적으로 이용하거나, 자신이 만든 작품인 것처럼 게시하는 것은 허용하지 않습니다.
- 이 저장소를 참고해 만든 작업을 공개할 때는 이 저장소 링크를 출처로 남겨 주세요.
- 그 밖의 이용이 필요하면 GitHub Issues로 먼저 문의해 주세요.

**English**
The music and source code in this repo, and the videos made with them, are published without an open-source license, and all rights belong to the creator.
Because the repo is public, you can view and fork it on GitHub, but for anything beyond that, please respect the following.

- You are welcome to read and run the code for personal study and reference.
- Please do not re-upload the videos or music elsewhere, or post edited versions of them, without permission.
- Commercial use, or presenting these works as your own, is not allowed.
- If you publish work made with reference to this repo, please credit it with a link back here.
- For any other use, please ask first through GitHub Issues.
