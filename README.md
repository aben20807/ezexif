# ezexif

Copy the exif from a photo and predefined tags to the clipboard.

Download the pre-built binary for Win10 from the [release](https://github.com/aben20807/ezexif/releases).

## Demo

![ezexif](https://user-images.githubusercontent.com/14831545/230545066-ec100126-a415-4285-9184-b21e8ffbae3f.gif)

## Build from source

+ Python 3.12 (project requires >=3.12)

+ Windows PowerShell commands:

```powershell
# Create & activate venv
uv venv --python 3.12
.\.venv\Scripts\activate

# Install runtime/build deps declared in pyproject
uv pip install -e .

# Option A: PyInstaller (simple, larger exe)
pyinstaller -F -c --clean --collect-all tkinterdnd2 --exclude-module black --icon=ezexif.ico .\ezexif\ezexif.py

# Option B: Nuitka (smaller/faster) — install Nuitka first
uv pip install nuitka
# Optional: install UPX for better compression (one of the following)
winget install --id UPX.UPX -e
# scoop install upx
# choco install upx

# Then run one of the provided scripts
./build_lite.ps1   # excludes AI libs (smallest)
./build_full.ps1   # includes AI libs (BLIP)
```

```powershell
> uv run .\ezexif\ezexif.py
```

## Icon credit

The [icon (ezexif.ico)](https://icon-icons.com/icon/Document-Image-images-picture/82883) is from [Madeby kking](https://icon-icons.com/users/lUybzhSQf3kZ7FimJzYlO/icon-sets/) who shares it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
