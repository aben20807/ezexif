param(
  [switch]$ForceUPX
)
# Build with PyInstaller (lite, excludes AI libs)
# Produces ezexif_lite.exe without torch/transformers for smallest binary size.

$ErrorActionPreference = 'Stop'

# Clean
Remove-Item -Recurse -Force .\dist_lite, .\build_lite -ErrorAction SilentlyContinue

# Compile (exclude AI libs; add data/hidden imports for core features)
pyinstaller `
  --onefile `
  --clean `
  --noupx `
  --console `
  --name ezexif_lite `
  --icon ezexif.ico `
  --distpath .\dist_lite `
  --workpath .\build_lite `
  --collect-all tkinterdnd2 `
  --collect-all PIL `
  --collect-all geopy `
  --exclude-module black `
  --exclude-module torch `
  --exclude-module transformers `
  ezexif\ezexif.py

# Optional: UPX compress the final exe only (safer than compressing DLLs)
if (Get-Command upx -ErrorAction SilentlyContinue) {
  $exe = Get-ChildItem -Recurse .\dist_lite -Filter ezexif_lite.exe | Select-Object -First 1
  if ($exe) {
    $upxArgs = @('--best','--lzma')
    if ($ForceUPX) { 
      $upxArgs += '--force' 
    } else { 
      Write-Host 'UPX: GuardCF likely enabled; defaulting to safe mode (no --force). Use -ForceUPX to override.' -ForegroundColor Yellow 
    }
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
