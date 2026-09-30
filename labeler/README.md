# Change labeller

A web page for labelling before/after satellite image pairs (Sentinel-2, 10 m pixels, 2.56 km × 2.56 km, about 6 years apart).
For each pair you choose **Change**, **No change** or **Seasonal change**, and write a caption when there is a change.
Every Save is written straight to `labeler/labels.json`.

## 1. Get the code and the image set

```bash
git clone -b man1 git@github.com:Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection.git
cd Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection
```

The images are shared separately as a zip, for example `labeler_set1_dev_150.zip`.
Put the zip in the **repository root** (the folder that contains `labeler/`) and unzip it there:

```bash
unzip labeler_set1_dev_150.zip
```

It fills in `labeler/images/`, `labeler/pairs.js` and `labeler/pairs.json`. The folder should then look like this:

```
labeler/
  index.html
  server.py
  pairs.js
  pairs.json
  images/   (300 PNG files for 150 pairs)
```

## 2. Start the save server (required before saving)

The page cannot write files on its own. A small Python server does the saving.
**It must be running the whole time you label.** It needs only Python 3, with no packages to install.

```bash
python3 labeler/server.py
```

Leave this terminal open. Stopping it (Ctrl-C, closing the terminal, or restarting the computer) stops saving.
Start it again with the same command. Labels you already saved are kept.

## 3. Label

1. Open **http://localhost:8000** in a browser.
2. Compare the two images. **Blink** (B key) flips between the two dates in place, which is the easiest way to spot change.
3. Choose one answer:

   | Button | Key | When |
   |---|---|---|
   | **Change** | 1 | A lasting land-use change is visible: new or expanded buildings/settlement, roads, construction sites, quarries/mines, ponds or reservoirs, cleared forest, solar farms, new farmland on bare land |
   | **No change** | 2 | Nothing meaningful differs |
   | **Seasonal change** | 3 | Only crop colour, vegetation greenness or water level differs |

4. If you chose **Change**, a caption box appears. Write one plain sentence about **what** changed and **where**, for example:
   *"A cluster of large buildings appears on previously open ground in the north-east."*
   - Describe only what you can see. At 10 m a single house is only 1–2 pixels.
   - Don't guess a building's purpose (such as "factory") or count small objects.
5. Click **Save & next** (or press Ctrl+Enter). You move to the next unlabelled pair.

- The numbered boxes at the bottom jump to any pair. Green means saved.
- To correct a pair, open it, change the answer and save again. The new answer replaces the old one.
- ← / → move between pairs without saving.

## 4. Hand in your labels

Your answers are in `labeler/labels.json`.
When you finish, **rename it to `labels_<set>_<yourname>.json`** (for example `labels_set1_dev_priya.json`) and send it back.
Each person labels independently: don't look at anyone else's labels first.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Save failed" or "Server not running" | The server isn't running. Start `python3 labeler/server.py` (step 2), then reload the page with Ctrl+Shift+R |
| No images | The zip wasn't unzipped in the repository root. `labeler/images/` and `labeler/pairs.js` must exist |
| `Address already in use` | The server is already running in another terminal. Use that one |
| Labelling on another computer on the same network | Start the server with `python3 labeler/server.py --host 0.0.0.0` and open `http://<this-computer's-IP>:8000` |

`labels.json` format: `{"<patch_id>": {"label": "change" | "no_change" | "seasonal", "caption": "... or null", "ts": <unix time>}}`
