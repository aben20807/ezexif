# Build with Nuitka (lite, excludes AI libs) + optional UPX compression
# Requirements:
#   - Python: pip/uv install nuitka (e.g., `uv pip install nuitka`)
#   - UPX (optional): install via winget/scoop/choco or from https://github.com/upx/upx/releases and ensure upx.exe is on PATH

$ErrorActionPreference = 'Stop'

# Read version from pyproject.toml
$pyproj = Get-Content -Raw -Path .\pyproject.toml
if ($pyproj -match 'version\s*=\s*"([^"]+)"') {
  $version = $Matches[1]
} else {
  $version = "0.1.0"
}

# Clean
Remove-Item -Recurse -Force .\dist_lite, .\build_lite -ErrorAction SilentlyContinue

# Compile (exclude torch/transformers; AI assist gracefully disabled due to lazy imports)
python -m nuitka `
  --onefile `
  --standalone `
  --follow-imports `
  --mingw64 `
  --assume-yes-for-downloads `
  --plugin-enable=tk-inter `
  --include-data-files=ezexif.ico=ezexif.ico `
  --output-dir=.\dist_lite `
  --company-name="ezexif" `
  --product-name="ezexif" `
  --windows-console-mode=attach `
  --product-version=$version `
  --file-version=$version `
  --nofollow-import-to=black,torch,transformers `
  ezexif\ezexif.py

# UPX compress
if (Get-Command upx -ErrorAction SilentlyContinue) {
  $exe = Get-ChildItem -Recurse .\dist_lite -Filter ezexif.exe | Select-Object -First 1
  if ($exe) { upx --best --lzma $exe.FullName }
} else {
  Write-Host "UPX not found on PATH; skipping compression. To install: winget install --id UPX.UPX -e (or 'scoop install upx' / 'choco install upx')." -ForegroundColor Yellow
}
