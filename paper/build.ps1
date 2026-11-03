# Builds the IEEE paper with XeLaTeX (required for the Vietnamese examples).
# Usage: powershell -ExecutionPolicy Bypass -File paper/build.ps1

$ErrorActionPreference = "Stop"
$miktex = "C:\Users\ACER\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
if (Test-Path $miktex) { $env:PATH = "$env:PATH;$miktex" }

Push-Location $PSScriptRoot
try {
    foreach ($pass in 1, 2) {
        Write-Host "xelatex pass $pass"
        xelatex -interaction=nonstopmode -halt-on-error paper.tex | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "build failed; see paper.log"
            Select-String -Path paper.log -Pattern "^!" -Context 0,3 | Select-Object -First 5
            exit 1
        }
    }
    $pages = (Select-String -Path paper.log -Pattern "Output written on .* \((\d+) pages?").Matches[0].Groups[1].Value
    Write-Host "paper.pdf written: $pages pages"
} finally {
    Pop-Location
}
