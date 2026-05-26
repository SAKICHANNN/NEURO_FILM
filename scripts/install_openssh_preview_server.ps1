# Install standalone Microsoft OpenSSH Preview and configure sshd.
# Use this when Windows Optional Capability OpenSSH.Server fails.

$ErrorActionPreference = "Stop"

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "This script must run as Administrator."
}

Write-Host "=== Installing Microsoft OpenSSH Preview ===" -ForegroundColor Green

winget install --id Microsoft.OpenSSH.Preview --exact `
    --accept-source-agreements --accept-package-agreements `
    --silent

$candidatePaths = @(
    "$env:ProgramFiles\OpenSSH\sshd.exe",
    "$env:ProgramFiles\OpenSSH-Win64\sshd.exe",
    "$env:ProgramFiles\Microsoft OpenSSH\sshd.exe",
    "$env:WINDIR\System32\OpenSSH\sshd.exe"
)

$sshdExe = $candidatePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $sshdExe) {
    $sshdExe = Get-ChildItem "$env:ProgramFiles" -Recurse -Filter sshd.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $sshdExe) {
    throw "Installed package, but sshd.exe was not found."
}

Write-Host "Found sshd.exe at $sshdExe" -ForegroundColor Green

$installDir = Split-Path $sshdExe -Parent
$installScript = Join-Path $installDir "install-sshd.ps1"
if (Test-Path $installScript) {
    powershell.exe -ExecutionPolicy Bypass -File $installScript
} elseif (-not (Get-Service sshd -ErrorAction SilentlyContinue)) {
    New-Service -Name sshd -BinaryPathName "`"$sshdExe`"" -DisplayName "OpenSSH SSH Server" -StartupType Automatic
}

Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

$existingRule = Get-NetFirewallRule -Name "K-MCFM-sshd" -ErrorAction SilentlyContinue
if (-not $existingRule) {
    New-NetFirewallRule `
        -Name "K-MCFM-sshd" `
        -DisplayName "K-MCFM OpenSSH Server" `
        -Enabled True `
        -Direction Inbound `
        -Protocol TCP `
        -Action Allow `
        -LocalPort 22 | Out-Null
} else {
    Enable-NetFirewallRule -Name "K-MCFM-sshd"
}

$sshDir = Join-Path $env:USERPROFILE ".ssh"
New-Item -Path $sshDir -ItemType Directory -Force | Out-Null
icacls $sshDir /inheritance:r /grant "$env:USERNAME:(OI)(CI)(F)" | Out-Null

Write-Host "SSH server status:" -ForegroundColor Green
Get-Service sshd | Format-Table -AutoSize

Write-Host "Windows login target:" -ForegroundColor Green
Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object InterfaceAlias, IPAddress |
    Format-Table -AutoSize
