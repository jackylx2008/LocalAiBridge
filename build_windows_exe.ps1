$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment Python was not found: $python"
}

Push-Location $projectRoot
try {
    & $python -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name LocalAiBridge `
        --icon "icons\windows\LocalAIBridge.ico" `
        --paths "src" `
        --add-data "config.yaml;." `
        --add-data "icons\windows\LocalAIBridge.ico;icons\windows" `
        "main.py"

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    Write-Host "Built: $projectRoot\dist\LocalAiBridge.exe"
}
finally {
    Pop-Location
}
