import base64
import os, sys
from tkinter import *
from tkinter import ttk
import tkinter.font as tkFont
from tkinterdnd2 import *
from PIL import Image, ImageTk
from PIL.ExifTags import TAGS
from fractions import Fraction
import clipboard
import configparser
import json

from icon_data import ICON_BASE64

# default configuration for generation first time
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

# "tag name": "display name"
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
    "img_path": None,
    "textbox_tags": None,
    "app_path": "",
    "icon_base64": ICON_BASE64,
}


class CustomText(Text):
    # Ref: https://stackoverflow.com/a/40618152
    def __init__(self, *args, **kwargs):
        """A text widget that report on internal widget commands"""
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


def extract_exif_and_copy(event):
    image_path = event.data
    # Workaround for the folder with chinese word as end which causes wrong paths
    if "\\" in image_path:
        image_path = image_path.replace(os.sep, "/")
    if image_path.startswith("{"):
        image_path = image_path[1:]
    if image_path.endswith("}"):
        image_path = image_path[:-1]

    global_vars["img_path"] = image_path
    print(f"-\n> Open '{image_path}'")

    try:
        img = Image.open(image_path)
        exif = img._getexif()

        if exif is None:
            print(f"Something wrong: exif is None")
            return

        exif_result = {}
        for tag, value in exif.items():
            key = TAGS.get(tag, tag)
            exif_result[key] = str(value)

        copy_str = ""
        exif2tag_additional_tags = ""
        for key in NEEDED_EXIF.keys():
            copy_str += f"{NEEDED_EXIF[key]}: "
            value = ""
            if key == "Location":  # locaiton is not a valid exif
                value = "'Location'"
            elif key == "FileName":
                value = os.path.basename(image_path)
            elif key not in exif_result.keys():
                value = "<NO DATA>"
            else:
                value = str(exif_result[key]).strip(
                    "\x00"
                )  # LensModel has some \x00 char...

            # custom format
            if key == "DateTimeOriginal":
                value = value.replace(":", "/", 2)
            if key == "ExposureTime":
                value = f"{Fraction(value).limit_denominator()} s"
            if key == "FocalLength":
                value += " mm"
            copy_str += f"{value}\n"

            try:
                exif2tag_dict = json.loads(CONFIG["Settings"]["exif2tag"])
                if value in exif2tag_dict.keys():
                    exif2tag_additional_tags += " " + exif2tag_dict[value]
            except Exception as e:
                print(
                    f"Something wrong when using json.loads: {e}\nCheck if your exif2tag is a valid dict"
                )

        # Append tags information
        copy_str = (
            copy_str
            + global_vars["textbox_tags"][int(CONFIG["Settings"]["tag_curr"])].get(
                "1.0", "end-1c"
            )
            + exif2tag_additional_tags
        )
        print(copy_str)
        clipboard.copy(copy_str)
        print(f"> Copied to clipboard!")

    except Exception as e:
        print(f"Something wrong: {e}")
        import pprint

        if "MakerNote" in exif_result.keys():
            exif_result["MakerNote"] = ""
        pprint.pprint(exif_result)
        clipboard.copy(f"Something wrong: {e}")
    finally:
        img.close()


def update_tags(event):
    tags_val = event.widget.get("1.0", "end-1c")
    tag_curr = int(CONFIG["Settings"]["tag_curr"])
    tag_set_idx = f"tags_{tag_curr}" if tag_curr < 5 else "exif2tag"
    CONFIG["Settings"][tag_set_idx] = tags_val


def get_config_path():
    if getattr(sys, "frozen", False):
        global_vars["app_path"] = os.path.dirname(sys.executable)
    elif __file__:
        global_vars["app_path"] = os.path.dirname(__file__)
    config_path = os.path.join(global_vars["app_path"], "ezexif_config.ini")
    return config_path


def read_config():
    config = configparser.ConfigParser(
        comment_prefixes=";"
    )  # the tags startswith # may be see as comment
    config.read(get_config_path(), encoding="utf-8-sig")
    CONFIG["Settings"]["tag_curr"] = config.get("Settings", "tag_curr")
    for i in range(5):
        CONFIG["Settings"][f"tags_{i}"] = config.get("Settings", f"tags_{i}")
    CONFIG["Settings"]["exif2tag"] = config.get("Settings", "exif2tag")


def write_config():
    config = configparser.ConfigParser(
        comment_prefixes=";"
    )  # the tags startswith # may be see as comment
    config.read(get_config_path(), encoding="utf-8-sig")
    config.set("Settings", "tag_curr", str(CONFIG["Settings"]["tag_curr"]))
    for i in range(5):
        config.set("Settings", f"tags_{i}", CONFIG["Settings"][f"tags_{i}"])
    config.set("Settings", "exif2tag", CONFIG["Settings"]["exif2tag"])
    with open(get_config_path(), "w", encoding="utf-8-sig") as configfile:
        config.write(configfile)


def gen_config():
    config = configparser.ConfigParser()
    config.read_dict(CONFIG)
    with open(get_config_path(), "w", encoding="utf-8-sig") as configfile:
        config.write(configfile)


def combobox_callback(event):
    tag_prev = CONFIG["Settings"]["tag_curr"]
    tag_prev = tag_prev if tag_prev != "exif2tag" else 5
    global_vars["textbox_tags"][int(tag_prev)].pack_forget()

    tag_curr = global_vars["combobox"].get()
    tag_curr = tag_curr if tag_curr != "exif2tag" else 5
    CONFIG["Settings"]["tag_curr"] = tag_curr
    global_vars["textbox_tags"][int(tag_curr)].pack()


def on_closing():
    write_config()
    global_vars["ws"].destroy()


def main():
    if os.path.exists(get_config_path()):
        read_config()
    else:
        gen_config()
    print(f"Config: {CONFIG}")

    global_vars["ws"] = TkinterDnD.Tk()
    global_vars["ws"].title("ezexif")
    global_vars["ws"].geometry("320x300")
    global_vars["ws"].config(bg="#F5F5F5")
    # convert base64 into icon data
    global_vars["ws"].iconphoto(
        True, ImageTk.PhotoImage(data=base64.b64decode(global_vars["icon_base64"]))
    )

    var_img_path = StringVar()
    Label(global_vars["ws"], text="Path of the Image", bg="#F5F5F5").pack(
        anchor=NW, padx=10
    )
    e_box = Entry(global_vars["ws"], textvar=var_img_path, width=80)
    e_box.pack(fill=X, padx=10)
    e_box.drop_target_register(DND_FILES)
    e_box.dnd_bind("<<Drop>>", extract_exif_and_copy)

    # tag preset
    lframe_tagset = LabelFrame(global_vars["ws"], text="", bg="#F5F5F5")
    Label(lframe_tagset, text="tag preset:", bg="#F5F5F5").pack(side=LEFT)
    global_vars["combobox"] = ttk.Combobox(lframe_tagset, state="readonly")
    global_vars["combobox"]["values"] = ("0", "1", "2", "3", "4", "exif2tag")
    global_vars["combobox"].current(CONFIG["Settings"]["tag_curr"])
    global_vars["combobox"].pack(side=RIGHT)
    global_vars["combobox"].bind("<<ComboboxSelected>>", combobox_callback)
    lframe_tagset.pack(fill=BOTH, expand=True, padx=10, pady=10)

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
        CustomText(lframe, height=22, width=50) for _ in range(6)
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

    lframe.pack(fill=BOTH, expand=True, padx=10, pady=10)
    # store tags_val to config file
    global_vars["ws"].protocol("WM_DELETE_WINDOW", on_closing)
    global_vars["ws"].mainloop()


if __name__ == "__main__":
    main()
