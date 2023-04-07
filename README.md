# ezexif

Copy the exif from a photo and predefined tags to the clipboard.

Download the pre-built binary for Win10 from the [release](https://github.com/aben20807/ezexif/releases).

## Demo

![ezexif](https://user-images.githubusercontent.com/14831545/230545066-ec100126-a415-4285-9184-b21e8ffbae3f.gif)

## Build from source

+ Python 3.9.13

+ Windows PowerShell commands:

```powershell
> python -m virtualenv venv
> .\venv\Scripts\Activate.ps1 # you may need 'Set-ExecutionPolicy RemoteSigned' before the command and run 'Set-ExecutionPolicy Restricted' after this command under the administrator mode
> pip install tkinterdnd2 Pillow clipboard pyinstaller
> pyinstaller -F -c --clean --collect-all tkinterdnd2 --exclude-module black --icon=ezexif.ico .\ezexif\ezexif.py
```

## Icon credit

The [icon (ezexif.ico)](https://icon-icons.com/icon/Document-Image-images-picture/82883) is from [Madeby kking](https://icon-icons.com/users/lUybzhSQf3kZ7FimJzYlO/icon-sets/) who shares it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
