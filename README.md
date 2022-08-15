## Build from source

```powershell
> python -m virtualenv venv
> .\venv\Scripts\Activate.ps1
> pip install tkinterdnd2 Pillow clipboard pyinstaller
> pyinstaller -F -c --clean --collect-all tkinterdnd2 --add-data "ezexif.ico;." --icon=ezexif.ico .\ezexif\ezexif.py
```

## Icon credit

The [icon (excp.ico)](https://icon-icons.com/icon/Document-Image-images-picture/82883) is from [Madeby kking](https://icon-icons.com/users/lUybzhSQf3kZ7FimJzYlO/icon-sets/) who shares it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)