# Build with PyInstaller (full, includes AI libs)
# This bundles transformers/torch; the EXE will be larger.
param(
  [switch]$ForceUPX
)

$ErrorActionPreference = 'Stop'

# Clean
Remove-Item -Recurse -Force .\dist_full, .\build_full -ErrorAction SilentlyContinue

# Compile (bundle AI libs; add data/hidden imports)
pyinstaller `
  --onefile `
  --clean `
  --noupx `
  --console `
  --name ezexif `
  --icon ezexif.ico `
  --distpath .\dist_full `
  --workpath .\build_full `
  --collect-all tkinterdnd2 `
  --exclude-module black `
  ezexif\ezexif.py

# Optional: UPX compress the final exe only (avoid DLL compression errors)
if (Get-Command upx -ErrorAction SilentlyContinue) {
  $exe = Get-ChildItem -Recurse .\dist_full -Filter ezexif.exe | Select-Object -First 1
  if ($exe) {
    $upxArgs = @('--best','--lzma')
    if ($ForceUPX) { $upxArgs += '--force' } else { Write-Host 'UPX: GuardCF likely enabled; defaulting to safe mode (no --force). Use -ForceUPX to override.' -ForegroundColor Yellow }
    & upx @upxArgs $exe.FullName
    $code = $LASTEXITCODE
    if ($code -eq 0) {
      Write-Host "UPX compression complete." -ForegroundColor Green
    } else {
      Write-Warning "UPX could not compress the binary (exit code $code). Skipping."
    }
  }
} else {
  Write-Host "UPX not found on PATH; skipping exe compression." -ForegroundColor Yellow
}
