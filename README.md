# EyeSpeak — Run Instructions

## 1. Install dependencies

Open a terminal **in the project folder** — the same folder that contains `app.py`, `index.html`, and `requirements.txt` (e.g. `cd path/to/EyeSpeak` first if you're not already there) — then run:

```bash
pip install -r requirements.txt
```

## 1b. Place the OCR files

Two extra items need to sit in your project folder for the OpenVINO-accelerated OCR to work:

- `ocr_openvino.py` and `openvino_infer_engine.py` — put both directly next to `app.py` (same folder).
- The `ocr_models/` folder (containing the `.xml`/`.bin` model files and a `_characters.txt` file) — put this whole folder directly next to `app.py` too.

Your folder should look like:
```
EyeSpeak 2.0/
  app.py
  index.html
  ocr_openvino.py
  openvino_infer_engine.py
  ocr_models/
    ch_PP-OCRv4_det_infer.xml
    ch_PP-OCRv4_det_infer.bin
    ch_PP-OCRv4_rec_infer.xml
    ch_PP-OCRv4_rec_infer.bin
    ch_PP-OCRv4_rec_infer_characters.txt
    ch_ppocr_mobile_v2.0_cls_infer.xml
    ch_ppocr_mobile_v2.0_cls_infer.bin
```

(If `pip install` complains about system packages, add `--break-system-packages` to the command.)

## 2. Run the app

```bash
python app.py
```

First run will take a minute or two — it downloads the model and converts it to OpenVINO IR format the first time. Every run after that is fast (cached).

You'll see it print something like:
```
Converting/loading model with OpenVINO... (first run takes longer, later runs are fast)
Model ready — running on OpenVINO Runtime...
 * Running on https://127.0.0.1:5000
```

## 3. Test on your laptop

Open `https://localhost:5000` in your browser. Your browser will warn about the self-signed certificate — click "Advanced" → "Proceed anyway." This warning is expected and fine for a demo.

## 4. Test on a phone (e.g. your grandparents' phone)

1. Make sure the phone is on the **same Wi-Fi network** as the laptop running the app.
2. Find your laptop's local IP address:
   - Mac/Linux: `ifconfig` (look for something like `192.168.1.23`)
   - Windows: `ipconfig` (look for "IPv4 Address")
3. On the phone's browser, go to `https://<that-ip>:5000` — e.g. `https://192.168.1.23:5000`
4. Accept the certificate warning the same way as on the laptop.
5. Allow camera access when prompted.

**Important:** Camera access will NOT work over plain `http://` on a phone — this is a browser security rule, not a bug in the app. It only works over `https://` or on `localhost`.

## 5. Known things to double check before a live demo

- Start the server a few minutes before you need it — the first OpenVINO conversion takes time.
- Test each language button at least once beforehand — translation depends on an external service and occasionally times out; if the demo Wi-Fi is slow, the description will just fall back to English (a fixed, non-breaking failure).
- If Safari on iPhone doesn't auto-play the audio, tap the "🔊 Repeat" button that appears — this is expected mobile browser behavior, not a bug.
