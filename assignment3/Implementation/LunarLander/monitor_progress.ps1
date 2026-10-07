param(
    [int]$RefreshSeconds = 30
)
function Get-ExperimentStats {
    param([string]$Pattern)
    $dirs = Get-ChildItem outputs -Filter $Pattern -Directory -ErrorAction SilentlyContinue
    $completed = ($dirs | Where-Object { Test-Path (Join-Path $_.FullName "final_model.pt") }).Count
    $total = $dirs.Count
    return @{
        Total = $total
        Completed = $completed
        Running = $total - $completed
    }
}
while ($true) {
    Clear-Host
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "LunarLander Training Progress Monitor" -ForegroundColor Cyan
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    $q1 = Get-ExperimentStats "SAC_default_auto_seed*"
    Write-Host "Q1/Q2 (Default Environment):" -ForegroundColor Yellow
    Write-Host "  Total: $($q1.Total) | Completed: $($q1.Completed) | Running: $($q1.Running)" -ForegroundColor White
    if ($q1.Total -gt 0) {
        $pct = [math]::Round(($q1.Completed / $q1.Total) * 100, 1)
        Write-Host "  Progress: $pct%" -ForegroundColor $(if ($pct -eq 100) { "Green" } else { "Yellow" })
    }
    Write-Host ""
    $q3f = Get-ExperimentStats "SAC_Q3_*alpha0.01_seed*"
    Write-Host "Q3 Fixed Alpha (α=0.01):" -ForegroundColor Yellow
    Write-Host "  Total: $($q3f.Total) | Completed: $($q3f.Completed) | Running: $($q3f.Running)" -ForegroundColor White
    if ($q3f.Total -gt 0) {
        $pct = [math]::Round(($q3f.Completed / $q3f.Total) * 100, 1)
        Write-Host "  Progress: $pct%" -ForegroundColor $(if ($pct -eq 100) { "Green" } else { "Yellow" })
    }
    Write-Host ""
    $q3a = Get-ExperimentStats "SAC_Q3_*auto_seed*"
    Write-Host "Q3 Auto Tuning:" -ForegroundColor Yellow
    Write-Host "  Total: $($q3a.Total) | Completed: $($q3a.Completed) | Running: $($q3a.Running)" -ForegroundColor White
    if ($q3a.Total -gt 0) {
        $pct = [math]::Round(($q3a.Completed / $q3a.Total) * 100, 1)
        Write-Host "  Progress: $pct%" -ForegroundColor $(if ($pct -eq 100) { "Green" } else { "Yellow" })
    }
    Write-Host ""
    $totalRuns = $q1.Total + $q3f.Total + $q3a.Total
    $totalCompleted = $q1.Completed + $q3f.Completed + $q3a.Completed
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Overall Progress: $totalCompleted / $totalRuns" -ForegroundColor Cyan
    if ($totalRuns -gt 0) {
        $overallPct = [math]::Round(($totalCompleted / $totalRuns) * 100, 1)
        Write-Host "Completion: $overallPct%" -ForegroundColor $(if ($overallPct -eq 100) { "Green" } else { "Cyan" })
    }
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Latest Runs:" -ForegroundColor Yellow
    Get-ChildItem outputs -Directory -ErrorAction SilentlyContinue | 
        Sort-Object LastWriteTime -Descending | 
        Select-Object -First 5 | 
        ForEach-Object {
            $status = if (Test-Path (Join-Path $_.FullName "final_model.pt")) { "[DONE]" } else { "[RUNNING]" }
            $color = if ($status -eq "[DONE]") { "Green" } else { "Yellow" }
            Write-Host "  $status - $($_.Name)" -ForegroundColor $color
        }
    Write-Host ""
    Write-Host "Refreshing in $RefreshSeconds seconds... (Ctrl+C to exit)" -ForegroundColor Gray
    Start-Sleep -Seconds $RefreshSeconds
}
