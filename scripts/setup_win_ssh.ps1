# K-MCFM Windows SSH setup.
# Run from an elevated PowerShell window.

$ErrorActionPreference = "Stop"

Write-Host "=== K-MCFM Windows SSH Setup ===" -ForegroundColor Green

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "This script must run as Administrator."
}

Write-Host "[1/4] Installing OpenSSH Server if needed..." -ForegroundColor Yellow
$capability = Get-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
if ($capability.State -ne "Installed") {
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
} else {
    Write-Host "  OpenSSH Server already installed." -ForegroundColor DarkGray
}

Write-Host "[2/4] Starting sshd and enabling autostart..." -ForegroundColor Yellow
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

Write-Host "[3/4] Opening Windows Firewall TCP/22..." -ForegroundColor Yellow
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

Write-Host "[4/4] Preparing user .ssh directory..." -ForegroundColor Yellow
$sshDir = Join-Path $env:USERPROFILE ".ssh"
New-Item -Path $sshDir -ItemType Directory -Force | Out-Null
icacls $sshDir /inheritance:r /grant "$env:USERNAME:(OI)(CI)(F)" | Out-Null

Write-Host ""
Write-Host "SSH server status:" -ForegroundColor Green
Get-Service sshd | Format-Table -AutoSize

Write-Host "Windows login target:" -ForegroundColor Green
$addresses = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object InterfaceAlias, IPAddress
$addresses | Format-Table -AutoSize

Write-Host "From Mac, test with:" -ForegroundColor Green
foreach ($addr in $addresses) {
    Write-Host "  ssh $env:USERNAME@$($addr.IPAddress)"
}
