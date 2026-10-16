from flask import Flask, request, jsonify, send_from_directory
import base64
import os
import io
import traceback

from PIL import Image
from gtts import gTTS  # noqa: F401
from tts_engine import text_to_speech
from optimum.intel import OVModelForZeroShotImageClassification
from transformers import CLIPProcessor
from ocr_openvino import read_label_text


app = Flask(__name__)


# ---------------------------------------------------------
# OPENVINO + CLIP MODEL
# ---------------------------------------------------------

MODEL_ID = "openai/clip-vit-base-patch16"

print(
    "Converting/loading model with OpenVINO... "
    "(first run takes longer, later runs are fast)"
)

model = OVModelForZeroShotImageClassification.from_pretrained(
    MODEL_ID,
    export=True
)

processor = CLIPProcessor.from_pretrained(MODEL_ID)

print("Model ready — running on OpenVINO Runtime.")


# ---------------------------------------------------------
# CLIP FALLBACK LABELS
# Used only when OCR cannot read useful text
# ---------------------------------------------------------

CANDIDATE_LABELS = [
    "a bottle of medicine pills",
    "a bottle of water",
    "a can of soup",
    "a box of cereal",
    "a bag of chips",
    "a carton of milk",
    "a jar of jam",
    "a loaf of bread",
    "a bottle of shampoo",
    "a tube of toothpaste",
    "a piece of fruit",
    "a vegetable",
]

CONFIDENCE_THRESHOLD = 0.35


# Translation language mapping
LANG_MAP = {
    "ar": "arabic",
    "ur": "urdu",
    "hi": "hindi",
    "tl": "filipino",
    "te": "telugu",
    "bn": "bengali",
    "ml": "malayalam",
    "ta": "tamil",
    "gu": "gujarati",
    "fr": "french",
    "es": "spanish",
    "de": "german",
}


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ---------------------------------------------------------
# IDENTIFY ENDPOINT
# ---------------------------------------------------------

@app.route("/identify", methods=["POST"])
def identify():

    try:
        # Get JSON sent from the browser
        data = request.get_json(silent=True) or {}

        image_data = data.get("image")

        # Default to English if no language was supplied
        language = data.get("language", "en")

        print("\n--- NEW IDENTIFICATION REQUEST ---")
        print("Language:", language)

        # -------------------------------------------------
        # CHECK IMAGE
        # -------------------------------------------------

        if not image_data:
            print("ERROR: No image received")

            return jsonify({
                "error": "No image received"
            }), 400

        if not isinstance(image_data, str):
            print("ERROR: Image data is not a string")

            return jsonify({
                "error": "Invalid image format"
            }), 400


        # -------------------------------------------------
        # REMOVE DATA URL PREFIX
        #
        # Browser normally sends:
        #
        # data:image/jpeg;base64,/9j/4AAQ...
        #
        # PIL only needs the Base64 section.
        # -------------------------------------------------

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]


        # -------------------------------------------------
        # DECODE CAMERA IMAGE
        # -------------------------------------------------

        try:
            image_bytes = base64.b64decode(image_data)

            print("Received image bytes:", len(image_bytes))
            print("First bytes:", image_bytes[:10])

            if len(image_bytes) == 0:
                raise ValueError("Decoded image contains no data")

            image = Image.open(io.BytesIO(image_bytes))

            # Force PIL to fully read the image
            image.load()

            # CLIP requires RGB
            image = image.convert("RGB")

            print(
                "Image successfully decoded:",
                image.size,
                image.mode
            )

        except Exception as e:

            print("IMAGE DECODING ERROR:", e)
            traceback.print_exc()

            return jsonify({
                "error": f"Could not decode camera image: {str(e)}"
            }), 400


        # -------------------------------------------------
        # STEP 1: OCR
        # Read exact text printed on packaging
        # -------------------------------------------------

        try:
            print("Running OCR...")

            ocr_text = read_label_text(image)

            print("OCR result:", ocr_text)

        except Exception as e:

            print("OCR failed:", e)
            traceback.print_exc()

            # Continue with CLIP instead of crashing
            ocr_text = None


        # -------------------------------------------------
        # IF OCR READS TEXT, USE IT
        # -------------------------------------------------

        if ocr_text and str(ocr_text).strip():

            label = str(ocr_text).strip()

            print("OCR reading used:", label)


        # -------------------------------------------------
        # OTHERWISE USE CLIP
        # -------------------------------------------------

        else:

            print("No useful OCR text.")
            print("Running CLIP object recognition...")

            inputs = processor(
                text=CANDIDATE_LABELS,
                images=image,
                return_tensors="pt",
                padding=True
            )

            outputs = model(**inputs)

            probs = outputs.logits_per_image.softmax(dim=1)

            best_idx = probs.argmax().item()

            confidence = probs[0][best_idx].item()

            detected_label = CANDIDATE_LABELS[best_idx]

            print(
                f"CLIP result: {detected_label} "
                f"({confidence:.1%} confidence)"
            )


            if confidence < CONFIDENCE_THRESHOLD:

                label = (
                    "I'm not sure what this is. "
                    "Please try again with better lighting."
                )

            else:

                label = detected_label


        # -------------------------------------------------
        # TRANSLATION
        # -------------------------------------------------

        if language != "en":

            try:

                from deep_translator import GoogleTranslator

                target_lang = LANG_MAP.get(
                    language,
                    language
                )

                print(
                    f"Translating from English to {target_lang}..."
                )

                translated_label = GoogleTranslator(
                    source="english",
                    target=target_lang
                ).translate(label)

                if translated_label:
                    label = translated_label

                print("Translated result:", label)

            except Exception as e:

                print(
                    "Translation failed. "
                    "Using English result:",
                    e
                )


        # -------------------------------------------------
        # TEXT TO SPEECH
        # -------------------------------------------------

        try:

            os.makedirs(
                "static",
                exist_ok=True
            )

            audio_path = os.path.join(
                "static",
                "output.mp3"
            )

            print("Generating speech...")

            text_to_speech(
                label,
                language,
                audio_path
            )

            print("Speech generated successfully.")

        except Exception as e:

            print("TEXT-TO-SPEECH ERROR:", e)
            traceback.print_exc()

            # Still return the identified text even if
            # audio generation fails.

            return jsonify({
                "description": label,
                "audio": None,
                "warning": "Object identified but audio generation failed."
            })


        # -------------------------------------------------
        # SEND RESULT BACK TO WEBSITE
        # -------------------------------------------------

        print("FINAL RESULT:", label)
        print("--- REQUEST COMPLETE ---\n")

        return jsonify({
            "description": label,
            "audio": "/static/output.mp3"
        })


    # -----------------------------------------------------
    # CATCH ANY OTHER UNEXPECTED ERROR
    # -----------------------------------------------------

    except Exception as e:

        print("IDENTIFY ERROR:", e)
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# STATIC FILES / AUDIO
# ---------------------------------------------------------

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(
        "static",
        filename
    )


# ---------------------------------------------------------
# START SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )