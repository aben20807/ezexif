# ezexif

Copy exif and tags from a photo to the clipboard.

Download the pre-build binary for win10 from the [release](https://github.com/aben20807/ezexif/releases).

## Demo

![ezexif](https://user-images.githubusercontent.com/14831545/184575651-5403acfc-e46c-4056-a7cf-877aac785a88.gif)


## Build from source

+ Python 3.9.13

+ Windows powershell commands:
```powershell
> python -m virtualenv venv
> .\venv\Scripts\Activate.ps1
> pip install tkinterdnd2 Pillow clipboard pyinstaller
> pyinstaller -F -c --clean --collect-all tkinterdnd2 --add-data "ezexif.ico;." --icon=ezexif.ico .\ezexif\ezexif.py
```

## Icon credit

The [icon (ezexif.ico)](https://icon-icons.com/icon/Document-Image-images-picture/82883) is from [Madeby kking](https://icon-icons.com/users/lUybzhSQf3kZ7FimJzYlO/icon-sets/) who shares it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)