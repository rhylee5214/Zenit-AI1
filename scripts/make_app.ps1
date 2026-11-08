# Builds ZENIT.exe in the project folder and puts a ZENIT shortcut on the desktop.
#   powershell -ExecutionPolicy Bypass -File scripts\make_app.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $csc)) { throw "C# compiler not found at $csc" }

$exe = Join-Path $root "ZENIT.exe"
& $csc /nologo /target:exe /optimize+ "/out:$exe" (Join-Path $root "app\ZenitLauncher.cs")
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop "ZENIT.lnk"
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $exe
$sc.WorkingDirectory = $root
$sc.Description = "Start ZENIT"
$sc.Save()
Write-Host "Built $exe"
Write-Host "Shortcut: $lnk"
