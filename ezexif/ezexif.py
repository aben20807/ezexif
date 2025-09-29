"""
ezexif: A tiny Tkinter GUI to drag-and-drop an image and copy formatted EXIF
information plus user-defined tags to the clipboard. Supports preset tag
blocks and an exif-to-tag mapping for camera/lens-specific hashtags.

Behavior note: This module is intentionally GUI-driven; refactors and comments
added here do not change the original behavior or UI layout.
"""

import base64
import configparser
import json
import os
import re
import sys
import threading
import tkinter.font as tkFont
from fractions import Fraction
from tkinter import *  # noqa: F401,F403 - keep wildcard for Tk constants and widgets
from tkinter import ttk
from typing import List, Optional, Tuple

import clipboard
from geopy.geocoders import Nominatim
from geopy.point import Point
from icon_data import ICON_BASE64
from PIL import Image, ImageTk
from PIL.ExifTags import GPSTAGS, TAGS
from tkinterdnd2 import DND_FILES, TkinterDnD
## transformers are imported lazily in _load_blip_cpu to allow AI-less builds

# Default configuration used to generate a config file the first time.
# The structure is a single "Settings" section with:
# - tag_curr: which preset is currently selected (0-4 or the string "exif2tag")
# - tags_0..tags_4: 5 preset tag text blocks
# - exif2tag: a JSON string mapping EXIF text values to additional tag strings
CONFIG = {
    "Settings": {
        "tag_curr": 0,
        "tags_0": "-\n",
        "tags_1": "-\n",
        "tags_2": "-\n",
        "tags_3": "-\n",
        "tags_4": "-\n",
        "exif2tag": '{"NIKON Z 6_2":"#nikon #nikontaiwan #nikonz6ii","NIKKOR Z 24-70mm f/4 S":"#nikkorz2470f4", "NIKKOR Z 14-24mm f/2.8 S": "#nikkorz1424mmf28s", "NIKKOR Z 35mm f/1.4": "#nikkorz35f14"}',
    }
}

# EXIF fields of interest (ordered) and their display labels.
# Key is the EXIF tag name; value is the label printed in the output.
NEEDED_EXIF = {
    "FileName": "File name",
    "DateTimeOriginal": "Date time",
    "Location": "Location",
    "Model": "Camera",
    "LensModel": "Lens model",
    "FNumber": "F-number",
    "ExposureTime": "Exposure time",
    "ISOSpeedRatings": "ISO",
    "FocalLength": "Focal length",
}

global_vars = {
    # Path of the last dropped image
    "img_path": None,
    # List[Text]: tag text boxes for each preset (0-4) and one for exif2tag mapping (index 5)
    "textbox_tags": None,
    # Base path for config file (differs in frozen vs script execution)
    "app_path": "",
    # Base64-encoded icon data used for the window icon
    "icon_base64": ICON_BASE64,
    # Output Text widget to display the generated content
    "output_text": None,
    # AI Assist toggle state (IntVar), 0=off, 1=on
    "ai_var": None,
    # AI quality selection (StringVar): 'Fast' or 'Accurate'
    "ai_quality_var": None,
}


class CustomText(Text):
    """Text widget subclass that emits a virtual event when modified.

    Creates a Tcl command proxy to intercept insert/delete/replace and generate
    a custom event (<<TextModified>>). Useful for auto-saving edits.

    Ref: https://stackoverflow.com/a/40618152
    """

    def __init__(self, *args, **kwargs):
        """A text widget that reports on internal widget commands."""
        Text.__init__(self, *args, **kwargs)
        # create a proxy for the underlying widget
        self._orig = self._w + "_orig"
        self.tk.call("rename", self._w, self._orig)
        self.tk.createcommand(self._w, self._proxy)

    def _proxy(self, command, *args):
        cmd = (self._orig, command) + args
        try:
            result = self.tk.call(cmd)
        except Exception:
            # avoid error (_tkinter.TclError: text doesn't contain any characters tagged with "sel")
            # when copying Ref: https://stackoverflow.com/a/53418346
            return None
        if command in ("insert", "delete", "replace"):
            self.event_generate("<<TextModified>>")
        return result


def normalize_dnd_path(raw_path: str) -> str:
    """Normalize a file path coming from a DnD event (Windows braces, quotes)."""
    path = raw_path or ""
    path = path.strip()
    if path.startswith("{") and path.endswith("}"):
        path = path[1:-1]
    if (path.startswith('"') and path.endswith('"')) or (
        path.startswith("'") and path.endswith("'")
    ):
        path = path[1:-1]
    # Normalize backslashes
    if os.name == "nt":
        path = path.replace("\\\\", "\\")
    return path


def _map_exif_to_named_dict(exif: dict | None) -> dict:
    """Convert a raw EXIF dict with numeric keys to human-readable tag names."""
    if not exif:
        return {}
    try:
        from PIL.ExifTags import TAGS
    except Exception:
        TAGS = {}
    result = {}
    for tag, value in exif.items():
        key = TAGS.get(tag, tag)
        result[key] = value
    return result


def _extract_gps_info(exif_named: dict) -> Optional[Tuple[float, float]]:
    """Extract latitude/longitude in decimal degrees from EXIF dict."""
    if not exif_named or "GPSInfo" not in exif_named:
        return None
    gps = exif_named.get("GPSInfo")
    if not isinstance(gps, dict):
        return None
    # Map numeric gps keys to names
    try:
        from PIL.ExifTags import GPSTAGS

        gps_named = {GPSTAGS.get(k, k): v for k, v in gps.items()}
    except Exception:
        gps_named = gps
    lat = gps_named.get("GPSLatitude")
    lat_ref = gps_named.get("GPSLatitudeRef")
    lon = gps_named.get("GPSLongitude")
    lon_ref = gps_named.get("GPSLongitudeRef")
    if not (lat and lon and lat_ref and lon_ref):
        return None

    def to_deg(value):
        try:
            d, m, s = value

            def r(x):
                try:
                    return x[0] / x[1]
                except Exception:
                    return float(x)

            return r(d) + r(m) / 60.0 + r(s) / 3600.0
        except Exception:
            return None

    lat_d = to_deg(lat)
    lon_d = to_deg(lon)
    if lat_d is None or lon_d is None:
        return None
    if str(lat_ref).upper() == "S":
        lat_d = -lat_d
    if str(lon_ref).upper() == "W":
        lon_d = -lon_d
    return (lat_d, lon_d)


_geolocator = None


def _reverse_geocode(lat: float, lon: float) -> str | None:
    """Reverse geocode lat/lon to a human-readable location using Nominatim.

    Returns a short, friendly display string or None on failure.
    """
    global _geolocator
    try:
        if _geolocator is None:
            _geolocator = Nominatim(user_agent="ezexif")
        # Request detailed address components; higher zoom favors street-level results
        loc = _geolocator.reverse(
            Point(lat, lon), language="en", zoom=18, addressdetails=True, timeout=5
        )
        if not loc:
            return None
        print(
            f"Reverse geocode: {lat},{lon} -> {loc.address if hasattr(loc, 'address') else loc}"
        )
        addr = loc.raw.get("address", {}) if hasattr(loc, "raw") else {}
        # Build a detailed label: house number + road + suburb/neighbourhood + city + state + postcode + country
        house_no = addr.get("house_number")
        road = addr.get("road") or addr.get("pedestrian") or addr.get("footway")
        suburb = addr.get("suburb") or addr.get("neighbourhood") or addr.get("quarter")
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("hamlet")
        )
        state = addr.get("state") or addr.get("region")
        postcode = addr.get("postcode")
        country = addr.get("country")

        road_components = [str(p).strip() for p in (house_no, road) if p]
        road_part = " ".join(road_components) if road_components else None
        parts = [p for p in [road_part, suburb, city, state, postcode, country] if p]
        return (
            ", ".join(parts)
            if parts
            else (loc.address if hasattr(loc, "address") else None)
        )
    except Exception:
        return None


# ---------------- AI Assist (BLIP) ----------------
# Cache BLIP models by size (Base/Large)
_blip_cache = {}


def _load_blip_cpu(model_size: str = "Base"):
    """Load BLIP captioning model on CPU.

    model_size: "Base" or "Large". Falls back to Base on failure.
    Returns a tuple (model, processor).
    """
    global _blip_cache
    key = (model_size or "Base").title()
    if key in _blip_cache:
        return _blip_cache[key]
    # Lazy import heavy deps via importlib to avoid bundling when excluded
    import importlib
    try:
        transformers = importlib.import_module("transformers")
        BlipForConditionalGeneration = getattr(transformers, "BlipForConditionalGeneration")
        BlipProcessor = getattr(transformers, "BlipProcessor")
        torch = importlib.import_module("torch")
    except Exception:
        print("AI modules not available (transformers/torch). BLIP disabled.")
        return None, None
    repo = (
        "Salesforce/blip-image-captioning-large"
        if key == "Large"
        else "Salesforce/blip-image-captioning-base"
    )
    try:
        print(f"Loading BLIP ({key}) captioning (CPU) from {repo}...")
        proc = BlipProcessor.from_pretrained(repo)
        model = BlipForConditionalGeneration.from_pretrained(repo)
        try:
            model.to("cpu").eval()
        except Exception:
            pass
        try:
            # Use half of available threads for responsiveness
            torch.set_num_threads(max(1, os.cpu_count() // 2))
        except Exception:
            pass
        _blip_cache[key] = (model, proc)
        print("BLIP loaded.")
        return model, proc
    except Exception as e:
        print(f"Failed to load BLIP {key}: {e}")
        if key != "Base":
            # Fallback to Base
            try:
                if "Base" in _blip_cache:
                    return _blip_cache["Base"]
                print("Falling back to BLIP Base...")
                proc = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                try:
                    model.to("cpu").eval()
                except Exception:
                    pass
                _blip_cache["Base"] = (model, proc)
                return model, proc
            except Exception as e2:
                print(f"Failed to load BLIP Base during fallback: {e2}")
        return None, None


_STOPWORDS = set(
    "a an the and or but if then when while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there all any both each few more most other some such no nor not only own same so than too very can will just don don’t should should’ve now i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing would could ought i’m you’re he’s she’s it’s we’re they’re i’ve you’ve we’ve they’ve i’d you’d he’d she’d we’d they’d i’ll you’ll he’ll she’ll we’ll they’ll isn’t aren’t wasn’t weren’t hasn’t haven’t hadn’t doesn’t don’t didn’t won’t wouldn’t shan’t shouldn’t can’t cannot couldn’t mustn’t let’s that’s who’s what’s here’s there’s when’s where’s why’s how’s".replace(
        "\n", " "
    ).split()
)


def _blip_caption_and_tags(
    pil_img: Image.Image, quality: str = "Fast", model_size: str = "Base"
) -> Tuple[Optional[str], Optional[List[str]]]:
    model, proc = _load_blip_cpu(model_size)
    if model is None or proc is None:
        return None, None
    try:
        # Decode parameters: trade speed vs accuracy
        if quality == "Very accurate":
            prompt_text = "a highly detailed professional photo of"
            num_beams = 8
            max_new = 70
            length_penalty = 1.3
            repetition_penalty = 1.05
            no_repeat_ngram_size = 3
        elif quality == "Accurate":
            prompt_text = "a detailed photo of"
            num_beams = 5
            max_new = 50
            length_penalty = 1.2
            repetition_penalty = 1.1
            no_repeat_ngram_size = 2
        else:  # Fast
            prompt_text = None
            num_beams = 3
            max_new = 24
            length_penalty = 1.0
            repetition_penalty = 1.1
            no_repeat_ngram_size = None
        # Use torch inference context if available
        try:
            import importlib as _il
            _torch = _il.import_module("torch")
            ctx = _torch.inference_mode()
        except Exception:
            from contextlib import nullcontext
            ctx = nullcontext()
        with ctx:
            if prompt_text:
                inputs = proc(images=pil_img, text=prompt_text, return_tensors="pt")
            else:
                inputs = proc(images=pil_img, return_tensors="pt")
            try:
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            except Exception:
                pass
            gen_kwargs = {
                "max_new_tokens": max_new,
                "num_beams": num_beams,
                "do_sample": False,
                "length_penalty": length_penalty,
                "repetition_penalty": repetition_penalty,
                "early_stopping": True,
            }
            if no_repeat_ngram_size:
                gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
            out_ids = model.generate(**inputs, **gen_kwargs)
            caption = proc.decode(out_ids[0], skip_special_tokens=True).strip()
        # Ensure the prompt_text itself is not included in the final caption
        if prompt_text and caption:
            try:
                caption = re.sub(
                    rf"^\s*{re.escape(prompt_text)}\s*[:,-]*\s*",
                    "",
                    caption,
                    flags=re.IGNORECASE,
                ).strip()
            except Exception:
                # Fallback: simple prefix check
                if caption.lower().startswith(prompt_text.lower()):
                    caption = caption[len(prompt_text) :].lstrip(" :,-")
        if not caption:
            return None, None
        # Improved tag extraction from caption
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", caption.lower())
        # Filter stopwords, very short tokens, numeric-like, and common adjectives/colors
        common_adjectives = {
            "beautiful",
            "stunning",
            "amazing",
            "gorgeous",
            "wonderful",
            "pretty",
            "lovely",
            "nice",
            "soft",
            "hard",
            "blurred",
            "blurry",
            "sharp",
            "vintage",
            "classic",
            "old",
            "new",
            "high",
            "low",
            "close",
            "wide",
        }
        colors = {
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "purple",
            "violet",
            "pink",
            "brown",
            "black",
            "white",
            "gray",
            "grey",
            "cyan",
            "magenta",
            "teal",
            "golden",
            "gold",
            "silver",
        }
        filtered = []
        for w in words:
            if len(w) < 3:
                continue
            if w in _STOPWORDS or w in common_adjectives or w in colors:
                continue
            if all(ch.isdigit() for ch in w):
                continue
            filtered.append(w)
        # Deduplicate while preserving order
        seen = set()
        filtered_unique = []
        for w in filtered:
            if w not in seen:
                seen.add(w)
                filtered_unique.append(w)
        # Choose up to 12 tags; prefer nouns-ish tokens by heuristic (endings)
        candidates = filtered_unique
        tags: List[str] = []
        for w in candidates:
            tags.append(w)
            if len(tags) >= 12:
                break
        return caption, tags
    except Exception as e:
        print(f"BLIP inference failed: {e}")
        return None, None


def _append_output_text(text: str):
    out_widget = global_vars.get("output_text")
    if out_widget is None:
        return
    try:
        out_widget.config(state="normal")
        # ensure a newline separation if there's existing text
        if out_widget.get("1.0", "end-1c"):
            out_widget.insert("end", "\n")
        out_widget.insert("end", text)
        out_widget.config(state="disabled")
    except Exception as e:
        print(f"Failed to append output text: {e}")


def _run_ai_assist(image_path: str):
    # Let the user know AI assist has started
    if global_vars.get("ws") is not None:
        global_vars["ws"].after(
            0, lambda: _append_output_text("AI assist: generating...")
        )
    try:
        start_ts = None
        with Image.open(image_path) as img:
            pil_img = img.convert("RGB")
        # Downscale to speed up CPU inference while preserving aspect ratio
        small = pil_img.copy()
        try:
            # Decide target size based on quality and model size
            q = (
                global_vars.get("ai_quality_var").get()
                if global_vars.get("ai_quality_var") is not None
                else "Fast"
            )
            msize = (
                global_vars.get("ai_model_size_var").get()
                if global_vars.get("ai_model_size_var") is not None
                else "Base"
            )
            if q == "Very accurate":
                target = 1280 if msize == "Large" else 1024
            elif q == "Accurate":
                target = 1024 if msize == "Large" else 896
            else:
                target = 512
            small.thumbnail((target, target), Image.LANCZOS)
        except Exception:
            if q == "Very accurate":
                target = 1280 if msize == "Large" else 1024
            elif q == "Accurate":
                target = 1024 if msize == "Large" else 896
            else:
                target = 512
            small.thumbnail((target, target))
        import time

        start_ts = time.time()

        # Progress reporter that posts to UI
        def _report(stage: str):
            msg_map = {
                "encoding": "AI: encoding image...",
                "encoded": "AI: image encoded",
                "captioning": "AI: generating caption...",
                "tagging": "AI: generating tags...",
                "done": "AI: generation finished",
            }
            m = msg_map.get(stage)
            if not m:
                return
            if global_vars.get("ws") is not None:
                global_vars["ws"].after(0, lambda: _append_output_text(m))

        state = {"done": False, "cancel": False}

        def _watchdog():
            if state["done"]:
                return
            state["cancel"] = True
            if global_vars.get("ws") is not None:
                global_vars["ws"].after(0, lambda: _append_output_text(timeout_msg))

        # schedule a timeout watchdog depending on quality/model size
        # Base/Fast: 60s, Accurate: 90s, Very accurate or Large: 120s
        timeout_ms = 60000
        timeout_msg = "AI assist: timed out after 60s"
        if q == "Accurate":
            timeout_ms = 90000
            timeout_msg = "AI assist: timed out after 90s"
        if q == "Very accurate" or msize == "Large":
            timeout_ms = 120000
            timeout_msg = "AI assist: timed out after 120s"
        if global_vars.get("ws") is not None:
            # Announce settings
            settings_line = f"AI: model={msize}, quality={q}, size~{target}px"
            global_vars["ws"].after(0, lambda: _append_output_text(settings_line))
            global_vars["ws"].after(timeout_ms, _watchdog)
        # BLIP-only path
        _report("captioning")
        quality = "Fast"
        if global_vars.get("ai_quality_var") is not None:
            try:
                quality = global_vars["ai_quality_var"].get()
            except Exception:
                quality = "Fast"
        model_size = (
            global_vars.get("ai_model_size_var").get()
            if global_vars.get("ai_model_size_var") is not None
            else "Base"
        )
        caption, tags = _blip_caption_and_tags(
            small, quality=quality, model_size=model_size
        )
        _report("done")
        state["done"] = True
        if state["cancel"]:
            # Skip posting results if we already timed out
            return
        elapsed = None
        try:
            elapsed = time.time() - start_ts if start_ts else None
        except Exception:
            pass
        if caption or tags:
            ai_lines = []
            if caption:
                ai_lines.append(f"AI caption: {caption}")
            if tags:
                ai_lines.append("AI tags: " + " ".join(f"#{t}" for t in tags))
            if elapsed is not None:
                ai_lines.append(f"AI assist done in {elapsed:.1f}s")
            text = "\n".join(ai_lines)
            # marshal back to UI thread
            if global_vars.get("ws") is not None:
                global_vars["ws"].after(0, lambda: _append_output_text(text))
            else:
                print(text)
        else:
            # Nothing produced by the model
            msg = "AI assist: no caption/tags produced."
            if global_vars.get("ws") is not None:
                global_vars["ws"].after(0, lambda: _append_output_text(msg))
            else:
                print(msg)
    except Exception as e:
        print(f"AI assist failed: {e}")
        if global_vars.get("ws") is not None:
            global_vars["ws"].after(
                0, lambda: _append_output_text(f"AI assist failed: {e}")
            )


def _format_exif_value(
    key: str, value: str, image_path: str, exif_named: dict | None = None
) -> str:
    """Apply original formatting rules for specific EXIF fields.

    - FileName comes from the dropped path
    - DateTimeOriginal replaces the first two ':' with '/' for readability
    - ExposureTime is represented as a simplified fraction (e.g., 1/200 s)
    - FocalLength is suffixed with ' mm'
    Other fields are passed through after stripping stray NULs.
    """
    # location is not a valid exif in this tool
    if key == "Location":
        # Attempt to get location from GPS info if available
        if exif_named:
            gps = _extract_gps_info(exif_named)
            if gps:
                lat, lon = gps
                loc = _reverse_geocode(lat, lon)
                if loc:
                    return loc
                return f"{lat:.6f}, {lon:.6f}"
        return "'Location'"
    if key == "FileName":
        return os.path.basename(image_path)
    # value cleanup (LensModel has some \x00 char...)
    v = str(value).strip("\x00") if value is not None else "<NO DATA>"
    if key == "DateTimeOriginal":
        v = v.replace(":", "/", 2)
    if key == "ExposureTime":
        try:
            v = f"{Fraction(v).limit_denominator()} s"
        except Exception:
            # keep original value if Fraction fails
            v = f"{v}"
    if key == "FocalLength":
        v = f"{v} mm"
    return v


def _lookup_additional_tags(value: str) -> str:
    """Return additional tag string from exif2tag mapping if value matches a key.

    The mapping is stored as a JSON string in CONFIG["Settings"]["exif2tag"],
    e.g. {"NIKON Z 6_2": "#nikon ..."}. When the formatted EXIF value exactly
    matches a key, the corresponding tag string is appended to the output.
    """
    try:
        exif2tag_dict = json.loads(CONFIG["Settings"]["exif2tag"])
        if value in exif2tag_dict.keys():
            return " " + exif2tag_dict[value]
    except Exception as e:
        print(
            f"Something wrong when using json.loads: {e}\nCheck if your exif2tag is a valid dict"
        )
    return ""


def extract_exif_and_copy(event):
    """DnD callback: extract EXIF from dropped image, build text, and copy to clipboard."""
    from PIL import Image

    image_path = normalize_dnd_path(event.data)
    global_vars["img_path"] = image_path
    print(f"-\n> Open '{image_path}'")

    exif_result = None
    try:
        # 1) Load image and extract raw EXIF
        with Image.open(image_path) as img:
            exif = img._getexif()

        if exif is None:
            print("Something wrong: exif is None")
            return

        # 2) Convert to human-readable keys
        exif_result = _map_exif_to_named_dict(exif)

        # 3) Build output lines in the desired order
        copy_str = ""
        exif2tag_additional_tags = ""
        for key in NEEDED_EXIF.keys():
            copy_str += f"{NEEDED_EXIF[key]}: "
            if key not in exif_result and key not in ("FileName", "Location"):
                value_display = "<NO DATA>"
            else:
                raw_value = exif_result.get(key, "")
                value_display = _format_exif_value(
                    key, raw_value, image_path, exif_result
                )
            copy_str += f"{value_display}\n"
            # Look up camera/lens specific extra tags
            exif2tag_additional_tags += _lookup_additional_tags(value_display)

        # 4) Append the current preset tags and any extra tags from the mapping
        copy_str = (
            copy_str
            + global_vars["textbox_tags"][int(CONFIG["Settings"]["tag_curr"])].get(
                "1.0", "end-1c"
            )
            + exif2tag_additional_tags
        )
        # If an output viewer exists, show the generated text
        out_widget = global_vars.get("output_text")
        if out_widget is not None:
            try:
                out_widget.config(state="normal")
                out_widget.delete("1.0", "end")
                out_widget.insert("1.0", copy_str)
                out_widget.config(state="disabled")
            except Exception:
                print("Failed to update output viewer")
                pass
        print(copy_str)
        clipboard.copy(copy_str)
        print("> Copied to clipboard!")

        # If AI assist is enabled, run it in the background to keep UI responsive
        ai_var = global_vars.get("ai_var")
        if ai_var is not None:
            try:
                if ai_var.get() == 1:
                    threading.Thread(
                        target=_run_ai_assist, args=(image_path,), daemon=True
                    ).start()
            except Exception:
                pass

    except Exception as e:
        print(f"Something wrong: {e}")
        import pprint

        if isinstance(exif_result, dict):
            # Avoid dumping large binary content
            if "MakerNote" in exif_result.keys():
                exif_result["MakerNote"] = ""
            pprint.pprint(exif_result)
        clipboard.copy(f"Something wrong: {e}")


def update_tags(event):
    """Update the in-memory CONFIG when any tag textbox is edited."""
    tags_val = event.widget.get("1.0", "end-1c")
    tag_curr = int(CONFIG["Settings"]["tag_curr"])
    tag_set_idx = f"tags_{tag_curr}" if tag_curr < 5 else "exif2tag"
    CONFIG["Settings"][tag_set_idx] = tags_val


def get_config_path():
    """Return the full path to ezexif_config.ini, handling frozen apps.

    When bundled with PyInstaller (sys.frozen), the config lives next to the
    executable; otherwise next to this source file.
    """
    if getattr(sys, "frozen", False):
        global_vars["app_path"] = os.path.dirname(sys.executable)
    elif __file__:
        global_vars["app_path"] = os.path.dirname(__file__)
    config_path = os.path.join(global_vars["app_path"], "ezexif_config.ini")
    return config_path


def read_config():
    config = configparser.ConfigParser(
        comment_prefixes=";"
    )  # Ensure '#' in tag text won't be treated as INI comments
    config.read(get_config_path(), encoding="utf-8-sig")
    CONFIG["Settings"]["tag_curr"] = config.get("Settings", "tag_curr")
    for i in range(5):
        CONFIG["Settings"][f"tags_{i}"] = config.get("Settings", f"tags_{i}")
    CONFIG["Settings"]["exif2tag"] = config.get("Settings", "exif2tag")


def write_config():
    config = configparser.ConfigParser(
        comment_prefixes=";"
    )  # Ensure '#' in tag text won't be treated as INI comments
    config.read(get_config_path(), encoding="utf-8-sig")
    config.set("Settings", "tag_curr", str(CONFIG["Settings"]["tag_curr"]))
    for i in range(5):
        config.set("Settings", f"tags_{i}", CONFIG["Settings"][f"tags_{i}"])
    config.set("Settings", "exif2tag", CONFIG["Settings"]["exif2tag"])
    with open(get_config_path(), "w", encoding="utf-8-sig") as configfile:
        config.write(configfile)


def gen_config():
    """Create a fresh config file using the defaults in CONFIG."""
    config = configparser.ConfigParser()
    config.read_dict(CONFIG)
    with open(get_config_path(), "w", encoding="utf-8-sig") as configfile:
        config.write(configfile)


def combobox_callback(event):
    """Switch visible textbox when the preset combobox changes.

    Note: The special "exif2tag" entry maps to index 5 in textbox list.
    The combobox returns strings; we convert to indices as needed.
    """
    tag_prev = CONFIG["Settings"]["tag_curr"]
    tag_prev_idx = 5 if tag_prev == "exif2tag" else int(tag_prev)
    global_vars["textbox_tags"][tag_prev_idx].pack_forget()

    tag_curr = global_vars["combobox"].get()
    tag_curr_idx = 5 if tag_curr == "exif2tag" else int(tag_curr)
    CONFIG["Settings"]["tag_curr"] = tag_curr
    global_vars["textbox_tags"][tag_curr_idx].pack()


def on_closing():
    """Persist config to disk and close the window."""
    write_config()
    global_vars["ws"].destroy()


def main():
    # Load or create configuration on startup
    if os.path.exists(get_config_path()):
        read_config()
    else:
        gen_config()
    print(f"Config: {CONFIG}")

    # Create main window with drag and drop support
    global_vars["ws"] = TkinterDnD.Tk()
    global_vars["ws"].title("ezexif")
    global_vars["ws"].geometry("360x600")
    global_vars["ws"].config(bg="#F5F5F5")
    # convert base64 into icon data
    global_vars["ws"].iconphoto(
        True, ImageTk.PhotoImage(data=base64.b64decode(global_vars["icon_base64"]))
    )

    # Input field that accepts DND for image path
    var_img_path = StringVar()
    Label(global_vars["ws"], text="Path of the Image", bg="#F5F5F5").pack(
        anchor=NW, padx=10
    )
    e_box = Entry(global_vars["ws"], textvar=var_img_path, width=80)
    e_box.pack(fill=X, padx=10)
    e_box.drop_target_register(DND_FILES)
    e_box.dnd_bind("<<Drop>>", extract_exif_and_copy)

    # tag preset
    # Combobox used to switch among 5 tag presets or the exif2tag mapping editor
    lframe_tagset = LabelFrame(global_vars["ws"], text="", bg="#F5F5F5")
    Label(lframe_tagset, text="tag preset:", bg="#F5F5F5").pack(side=LEFT)
    global_vars["combobox"] = ttk.Combobox(lframe_tagset, state="readonly")
    global_vars["combobox"]["values"] = ("0", "1", "2", "3", "4", "exif2tag")
    # Ensure index is int for ttk.Combobox.current
    try:
        _curr_index = 5 if CONFIG["Settings"]["tag_curr"] == "exif2tag" else int(CONFIG["Settings"]["tag_curr"])  # type: ignore[arg-type]
    except Exception:
        _curr_index = 0
    global_vars["combobox"].current(_curr_index)
    global_vars["combobox"].pack(side=RIGHT)
    global_vars["combobox"].bind("<<ComboboxSelected>>", combobox_callback)
    lframe_tagset.pack(fill=BOTH, expand=True, padx=10, pady=10)

    # AI assist toggle
    ai_frame = Frame(global_vars["ws"], bg="#F5F5F5")
    global_vars["ai_var"] = IntVar(value=1)  # checked by default
    Checkbutton(
        ai_frame,
        text="AI assist",
        variable=global_vars["ai_var"],
        bg="#F5F5F5",
        anchor=W,
    ).pack(side=LEFT)
    # Quality selector
    Label(ai_frame, text=" Quality:", bg="#F5F5F5").pack(side=LEFT)
    global_vars["ai_quality_var"] = StringVar(value="Accurate")
    ai_quality_combo = ttk.Combobox(
        ai_frame,
        state="readonly",
        width=10,
        textvariable=global_vars["ai_quality_var"],
        values=("Fast", "Accurate", "Very accurate"),
    )
    ai_quality_combo.current(2)  # default to "Very accurate"
    ai_quality_combo.pack(side=LEFT, padx=(4, 0))
    # Model size selector
    Label(ai_frame, text=" Model:", bg="#F5F5F5").pack(side=LEFT, padx=(8, 0))
    global_vars["ai_model_size_var"] = StringVar(value="Base")
    ai_model_combo = ttk.Combobox(
        ai_frame,
        state="readonly",
        width=8,
        textvariable=global_vars["ai_model_size_var"],
        values=("Base", "Large"),
    )
    ai_model_combo.current(1)  # default to "Large"
    ai_model_combo.pack(side=LEFT, padx=(4, 0))
    ai_frame.pack(fill=BOTH, expand=False, padx=10, pady=(0, 0))

    # Instructions section + tag textboxes area
    lframe = LabelFrame(global_vars["ws"], text="Instructions", bg="#F5F5F5")
    Label(
        lframe,
        bg="#F5F5F5",
        text="Drag your file in the text field.",
    ).pack(fill=BOTH, expand=True)

    # Text box for tags, multiple boxes for different presets
    # the last one, exif2tag, is used to set the mapping for
    # additional tag according to the camera
    global_vars["textbox_tags"] = [
        CustomText(lframe, height=4, width=50) for _ in range(6)
    ]

    for i in range(6):
        tag_set_idx = f"tags_{i}" if i < 5 else "exif2tag"
        global_vars["textbox_tags"][i].insert("end-1c", CONFIG["Settings"][tag_set_idx])
        global_vars["textbox_tags"][i].configure(
            font=tkFont.Font(family="Microsoft YaHei", size=10)
        )
        global_vars["textbox_tags"][i].drop_target_register(DND_FILES)
        global_vars["textbox_tags"][i].dnd_bind("<<Drop>>", extract_exif_and_copy)
        # store modified text to tags_val
        global_vars["textbox_tags"][i].bind("<<TextModified>>", update_tags)
    global_vars["textbox_tags"][int(CONFIG["Settings"]["tag_curr"])].pack(side=TOP)

    # Pack the instruction/tag frame (original layout restored)
    lframe.pack(fill=BOTH, expand=True, padx=10, pady=10)

    # Output viewer for generated text
    output_frame = LabelFrame(global_vars["ws"], text="Output", bg="#F5F5F5")
    output_text = Text(output_frame, height=24, width=50, wrap="word")
    output_text.configure(font=tkFont.Font(family="Microsoft YaHei", size=9))
    output_text.config(state="disabled")
    output_text.pack(fill=BOTH, expand=True)
    output_frame.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))
    global_vars["output_text"] = output_text
    # store tags_val to config file
    global_vars["ws"].protocol("WM_DELETE_WINDOW", on_closing)
    global_vars["ws"].mainloop()


if __name__ == "__main__":
    main()
