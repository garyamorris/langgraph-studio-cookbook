#!/usr/bin/env python3
"""Contact sheet of frames from an mp4: sheet.py clip.mp4 out.png T1 T2 ... [--cols 2] [--w 960]. Uses rec/vtool for decoding."""
import subprocess, sys, os, tempfile
from PIL import Image, ImageDraw
a = sys.argv[1:]; cols = 2; w = 960
if '--cols' in a: i = a.index('--cols'); cols = int(a[i+1]); del a[i:i+2]
if '--w' in a: i = a.index('--w'); w = int(a[i+1]); del a[i:i+2]
clip, out, times = a[0], a[1], [float(t) for t in a[2:]]
here = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp()
ims = []
for t in times:
    p = os.path.join(tmp, f"f{t}.png")
    subprocess.run([os.path.join(here, "rec/vtool"), "frame", clip, str(t), p], capture_output=True)
    im = Image.open(p).convert("RGB"); h = int(w * im.height / im.width); im = im.resize((w, h), Image.LANCZOS)
    ImageDraw.Draw(im).text((10, h - 22), f"{os.path.basename(clip)} @ {t}s", fill=(255, 255, 0)); ims.append(im)
rows = (len(ims) + cols - 1) // cols; H = ims[0].height
sheet = Image.new("RGB", (cols * w, rows * H), (0, 0, 0))
for i, im in enumerate(ims): sheet.paste(im, ((i % cols) * w, (i // cols) * H))
sheet.save(out); print(out, sheet.size)
