Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Q3: Reward Switching - Auto Tuning" -ForegroundColor Cyan
Write-Host "Automated Temperature Tuning" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  - Experiment: Q3 Auto Tuning"
Write-Host "  - Seeds: 15 (0-14)"
Write-Host "  - Workers: 8 parallel"
Write-Host "  - Timesteps: 500,000 per seed"
Write-Host "  - Buffer: 100K (fixed, per TA)"
Write-Host "  - Hover reward: +200 → -100 (switch at 250K)"
Write-Host "  - Alpha: Automated (adapts to reward change)"
Write-Host "  - Estimated time: ~6-9 hours"
Write-Host ""
$confirm = Read-Host "Start Q3 Auto Tuning experiments? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Aborted." -ForegroundColor Red
    exit
}
Write-Host ""
Write-Host "Launching 15 seeds..." -ForegroundColor Green
Write-Host "Note: 8 will run in parallel at a time" -ForegroundColor Yellow
Write-Host ""
for ($i=0; $i -lt 15; $i++) {
    Write-Host "Starting seed $i..." -ForegroundColor Cyan
    Start-Process python -ArgumentList "train_sac.py --experiment q3_auto --seed $i" -NoNewWindow -WorkingDirectory $PSScriptRoot
    Start-Sleep -Milliseconds 500  # Small delay to stagger starts
}
Write-Host ""
Write-Host "All seeds launched!" -ForegroundColor Green
Write-Host "Monitor progress in outputs/ folder" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop this script (training will continue)" -ForegroundColor Yellow
Write-Host ""
while ($true) {
    Start-Sleep -Seconds 60
    $completed = (Get-ChildItem outputs -Filter "SAC_Q3_*auto_seed*" -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path (Join-Path $_.FullName "final_model.pt") }).Count
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Completed: $completed / 15" -ForegroundColor Cyan
    if ($completed -eq 15) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Q3 AUTO TUNING EXPERIMENTS COMPLETE!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        break
    }
}
