import os, sys
from tkinter import *
import tkinter.font as tkFont
from tkinterdnd2 import *
from PIL import Image
from PIL.ExifTags import TAGS
from fractions import Fraction
import clipboard
import configparser

# default configuration for generation first time
CONFIG = {
    "Settings": {
        "tags": "-\n#photography #photographylovers #photoshoot "
        "#outdoors #nature #micro_lens #mushroom #菇 #成大 #榕園 #ncku"
    }
}

# "tag name": "display name"
NEEDED_EXIF = {
    "DateTime": "Date time",
    "Location": "Location",
    "Model": "Camera",
    "LensModel": "Lens model",
    "FNumber": "F-number",
    "ExposureTime": "Exposure time",
    "ISOSpeedRatings": "ISO",
    "FocalLength": "Focal length",
}

var_img_path = None
textbox_tags = None
app_path = ""


def callback(event):
    global var_img_path
    var_img_path.set(event.data)

    image_path = event.data
    print(f"-\n> Open '{image_path}'")

    try:
        img = Image.open(image_path)
        exif = img._getexif()

        if exif is None:
            print(f"Something wrong: exif is None")
            return

        exif_result = {}
        for (tag, value) in exif.items():
            key = TAGS.get(tag, tag)
            exif_result[key] = str(value)

        copy_str = ""
        for key in NEEDED_EXIF.keys():
            copy_str += f"{NEEDED_EXIF[key]}: "
            value = ""
            if key == "Location":  # locaiton is not a valid exif
                value = "'Location'"
            else:
                value = str(exif_result[key]).strip(
                    "\x00"
                )  # LensModel has some \x00 char...

            # custom format
            if key == "DateTime":
                value = value.replace(":", "/", 2)
            if key == "ExposureTime":
                value = f"{Fraction(value).limit_denominator()} s"
            if key == "FocalLength":
                value += " mm"
            copy_str += f"{value}\n"

        # Append tags information
        copy_str = copy_str + textbox_tags.get("1.0", "end-1c")
        print(copy_str)
        clipboard.copy(copy_str)
        print(f"> Copied to clipboard!")

    except Exception as e:
        print(f"Something wrong: {e}")
        clipboard.copy(f"Something wrong: {e}")
    finally:
        img.close()


def get_config_path():
    global app_path
    if getattr(sys, "frozen", False):
        app_path = os.path.dirname(sys.executable)
    elif __file__:
        app_path = os.path.dirname(__file__)
    config_path = os.path.join(app_path, "ezexif_config.ini")
    return config_path


def read_config():
    config = configparser.ConfigParser(
        comment_prefixes=";"
    )  # the tags startswith # may be see as comment
    config.read(get_config_path(), encoding="utf-8-sig")
    CONFIG["Settings"]["tags"] = config.get("Settings", "tags")
    print(CONFIG["Settings"]["tags"])


def gen_config():
    config = configparser.ConfigParser()
    config.read_dict(CONFIG)
    with open(get_config_path(), "w", encoding="utf-8-sig") as configfile:
        config.write(configfile)


def main():
    if os.path.exists(get_config_path()):
        read_config()
    else:
        gen_config()
    print(f"Config: {CONFIG}")

    ws = TkinterDnD.Tk()
    ws.title("ezexif")
    ws.geometry("320x240")
    ws.config(bg="#F5F5F5")
    global app_path
    ws.iconbitmap(app_path + os.sep + ".." + os.sep + "ezexif.ico")

    global var_img_path
    var_img_path = StringVar()
    Label(ws, text="Path of the Image", bg="#F5F5F5").pack(anchor=NW, padx=10)
    e_box = Entry(ws, textvar=var_img_path, width=80)
    e_box.pack(fill=X, padx=10)
    e_box.drop_target_register(DND_FILES)
    e_box.dnd_bind("<<Drop>>", callback)

    lframe = LabelFrame(ws, text="Instructions", bg="#F5F5F5")
    Label(
        lframe,
        bg="#F5F5F5",
        text="Drag your file in the text field.",
    ).pack(fill=BOTH, expand=True)

    # Text box for tags
    global textbox_tags
    textbox_tags = Text(lframe, height=22, width=50)
    textbox_tags.insert("end-1c", CONFIG["Settings"]["tags"])
    textbox_tags.pack(side=TOP)
    textbox_tags.configure(font=tkFont.Font(family="Microsoft YaHei", size=10))
    textbox_tags.drop_target_register(DND_FILES)
    textbox_tags.dnd_bind("<<Drop>>", callback)

    lframe.pack(fill=BOTH, expand=True, padx=10, pady=10)

    ws.mainloop()


if __name__ == "__main__":
    main()
