# Windows 一键环境安装脚本
# 在 PowerShell（管理员）中运行

Write-Host "=== K-MCFM Windows Setup ===" -ForegroundColor Green

# 1. OpenSSH Server
Write-Host "[1/5] Installing OpenSSH Server..." -ForegroundColor Yellow
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
Write-Host "  SSH server ready. Port 22 open." -ForegroundColor Green

# 2. Git clone
Write-Host "[2/5] Cloning repository..." -ForegroundColor Yellow
$repoPath = "C:\projects\NEURO_FILM"
if (!(Test-Path $repoPath)) {
    New-Item -ItemType Directory -Path C:\projects -Force
    git clone https://github.com/SAKICHANNN/NEURO_FILM.git $repoPath
    cd $repoPath
    git checkout codex/experiment-log
} else {
    cd $repoPath
    git pull
}
Write-Host "  Repo ready at $repoPath" -ForegroundColor Green

# 3. Python venv
Write-Host "[3/5] Creating Python environment..." -ForegroundColor Yellow
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
Write-Host "  .venv created" -ForegroundColor Green

# 4. Dependencies
Write-Host "[4/5] Installing dependencies (this takes 5-10 min)..." -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install diffusers transformers accelerate peft safetensors huggingface_hub xformers
pip install datasets bitsandbytes
pip install lpips piq kornia
Write-Host "  Dependencies installed" -ForegroundColor Green

# 5. Verify
Write-Host "[5/5] Verifying CUDA..." -ForegroundColor Yellow
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB)')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "  CUDA ready!" -ForegroundColor Green
} else {
    Write-Host "  CUDA FAILED. Check PyTorch installation." -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. Download models: python scripts/download_models.py"
Write-Host "  2. Get data from Mac: rsync -avz mac_ip:~/neuro_film/data/ data/"
Write-Host "  3. Test: python scripts/translate.py test.jpg --style portra_400"
