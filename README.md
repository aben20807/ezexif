# ezexif

Copy the exif from a photo and predefined tags to the clipboard.

Download the pre-built binary for Win10 from the [release](https://github.com/aben20807/ezexif/releases).

## Features

- **EXIF Extraction**: Automatically extracts and formats key EXIF data (camera, lens, exposure settings, date/time, focal length, ISO, etc.)
- **GPS Location**: Converts GPS coordinates to human-readable addresses with reverse geocoding (street, city, state, country)
- **AI Image Analysis** (optional):
  - Automatic caption generation using BLIP models (Base/Large)
  - Smart tag extraction from image content
  - Quality presets: Fast, Accurate, Very Accurate
  - CPU-optimized inference with adaptive image sizing
- **Customizable Tag Presets**: 5 preset slots + camera/lens-specific tag mapping
- **Drag & Drop Interface**: Simple Tkinter GUI with drag-and-drop support
- **One-Click Copy**: Formatted output automatically copied to clipboard
- **Portable Builds**: Available as standalone executables (lite without AI, full with AI assist)

## Demo

![ezexif](https://user-images.githubusercontent.com/14831545/230545066-ec100126-a415-4285-9184-b21e8ffbae3f.gif)

## Build from source

- Python 3.12 (project requires >=3.12)

- Windows PowerShell commands:

```powershell
# Create & activate venv
uv venv --python 3.12
.\.venv\Scripts\activate

# Install runtime/build deps declared in pyproject
uv pip install -e .

# Recommended: PyInstaller (fast, reliable)
# Use the provided scripts for convenience
./build_lite.ps1  # excludes AI libs → ezexif_lite.exe (smallest binary)
./build_full.ps1  # includes AI libs (BLIP) → ezexif.exe (larger with AI assist)

# Optional: UPX can further compress the exe if installed and on PATH.
# You can install UPX via winget/scoop/choco; PyInstaller will auto-detect it.
```

```powershell
> uv run .\ezexif\ezexif.py
```

## Icon credit

The [icon (ezexif.ico)](https://icon-icons.com/icon/Document-Image-images-picture/82883) is from [Madeby kking](https://icon-icons.com/users/lUybzhSQf3kZ7FimJzYlO/icon-sets/) who shares it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
