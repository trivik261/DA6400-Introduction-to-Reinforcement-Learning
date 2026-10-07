param(
    [string]$ConfigFile = "config.json"
)
$ExperimentName = "5b_automatic_90_0.1x_adaptive_reward_12Torque"
$TargetAngle = 90
$RewardScale = 0.1
$RewardType = "adaptive"
$SacType = "automatic"
$MaxTorque = 12.0
$NumSeeds = 15
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "EXPERIMENT 5b: Automatic SAC - 90° - 0.1x Reward Scale - Adaptive Reward - 12.0 Torque" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Target Angle: $TargetAngle°"
Write-Host "Reward Type: $RewardType"
Write-Host "Reward Scale: ${RewardScale}x"
Write-Host "SAC Type: $SacType"
Write-Host "Max Torque: $MaxTorque N·m"
Write-Host "Seeds: 0-$($NumSeeds - 1)"
Write-Host "Output Directory: outputs/$ExperimentName"
Write-Host "================================================================================`n" -ForegroundColor Cyan
Write-Host "Loading config from: $ConfigFile" -ForegroundColor Yellow
$config = Get-Content $ConfigFile | ConvertFrom-Json
$config.training.output_dir = "outputs/$ExperimentName"
$config.environment.reward_type = $RewardType
$config.environment.reward_scale = $RewardScale
$config.environment.max_torque = $MaxTorque
$config.sac.type = $SacType
$config.features.enable_live_plot = $false
$config.features.enable_checkpoints = $true
$config.features.enable_early_stopping = $true
$tempConfigFile = "config_5b_auto_01x_temp.json"
$config | ConvertTo-Json -Depth 10 | Set-Content $tempConfigFile
Write-Host "Modified config saved to: $tempConfigFile`n" -ForegroundColor Green
Write-Host "Configuration Summary:" -ForegroundColor Yellow
Write-Host "  Reward Type: $($config.environment.reward_type)"
Write-Host "  Reward Scale: $($config.environment.reward_scale)"
Write-Host "  Max Torque: $($config.environment.max_torque) N·m"
Write-Host "  SAC Type: $($config.sac.type)"
Write-Host "  Checkpoints: $($config.features.enable_checkpoints)"
Write-Host "  Early Stopping: $($config.features.enable_early_stopping)"
Write-Host "  Live Plot: $($config.features.enable_live_plot)`n"
$existingProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($existingProcesses) {
    $processCount = ($existingProcesses | Measure-Object).Count
    Write-Host "WARNING: Found $processCount existing Python process(es) running." -ForegroundColor Yellow
    Write-Host "Waiting for existing processes to complete before starting new experiment...`n" -ForegroundColor Yellow
    while ($true) {
        $stillRunning = Get-Process python -ErrorAction SilentlyContinue
        if (-not $stillRunning) {
            Write-Host "All existing Python processes completed. Starting experiment...`n" -ForegroundColor Green
            break
        }
        $currentCount = ($stillRunning | Measure-Object).Count
        Write-Host "  Waiting... ($currentCount process(es) still running)" -ForegroundColor Gray
        Start-Sleep -Seconds 10
    }
}
$MaxConcurrent = 8
Write-Host "Starting $NumSeeds training processes (max $MaxConcurrent concurrent)..." -ForegroundColor Green
Write-Host "Processes will run in minimized windows.`n" -ForegroundColor Yellow
$allSeeds = 0..($NumSeeds - 1)
$runningProcesses = @{}
$completedSeeds = @()
$nextSeedIndex = 0
$initialBatch = [Math]::Min($MaxConcurrent, $NumSeeds)
for ($i = 0; $i -lt $initialBatch; $i++) {
    $seed = $allSeeds[$nextSeedIndex]
    Write-Host "  Starting Seed $seed..." -ForegroundColor Gray
    $arguments = "train.py --config $tempConfigFile --target_angle $TargetAngle --seed $seed"
    $process = Start-Process -FilePath "python" -ArgumentList $arguments -PassThru -WindowStyle Minimized
    $runningProcesses[$seed] = $process
    $nextSeedIndex++
    Start-Sleep -Milliseconds 500
}
Write-Host "`nInitial batch of $initialBatch processes started." -ForegroundColor Green
Write-Host "Monitoring processes and starting new ones as they complete...`n" -ForegroundColor Yellow
while ($completedSeeds.Count -lt $NumSeeds) {
    Start-Sleep -Seconds 5
    $completedNow = @()
    foreach ($seed in $runningProcesses.Keys) {
        $process = $runningProcesses[$seed]
        if ($process.HasExited) {
            $completedNow += $seed
            $completedSeeds += $seed
            Write-Host "  Seed $seed completed (Exit Code: $($process.ExitCode))" -ForegroundColor Green
        }
    }
    foreach ($seed in $completedNow) {
        $runningProcesses.Remove($seed)
        if ($nextSeedIndex -lt $NumSeeds) {
            $newSeed = $allSeeds[$nextSeedIndex]
            Write-Host "  Starting Seed $newSeed..." -ForegroundColor Cyan
            $arguments = "train.py --config $tempConfigFile --target_angle $TargetAngle --seed $newSeed"
            $process = Start-Process -FilePath "python" -ArgumentList $arguments -PassThru -WindowStyle Minimized
            $runningProcesses[$newSeed] = $process
            $nextSeedIndex++
            Start-Sleep -Milliseconds 500
        }
    }
    $remaining = $NumSeeds - $completedSeeds.Count
    if ($completedSeeds.Count % 3 -eq 0 -and $completedSeeds.Count -gt 0 -and $remaining -gt 0) {
        Write-Host "`n  Progress: $($completedSeeds.Count)/$NumSeeds completed, $remaining remaining, $($runningProcesses.Count) running`n" -ForegroundColor Yellow
    }
}
Write-Host "`nAll processes completed!" -ForegroundColor Green
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "EXPERIMENT 5b (Automatic 0.1x) COMPLETED" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
$resultFiles = Get-ChildItem -Path "outputs/$ExperimentName" -Recurse -Filter "results.json" -ErrorAction SilentlyContinue
$completedCount = $resultFiles.Count
$failedCount = $NumSeeds - $completedCount
Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "  Completed: $completedCount / $NumSeeds" -ForegroundColor $(if ($completedCount -eq $NumSeeds) { "Green" } else { "Yellow" })
Write-Host "  Failed/Incomplete: $failedCount / $NumSeeds" -ForegroundColor $(if ($failedCount -gt 0) { "Red" } else { "Green" })
if ($completedCount -gt 0) {
    Write-Host "`nResults saved in: outputs/$ExperimentName/" -ForegroundColor Cyan
} else {
    Write-Host "`nWARNING: No results found! Check for errors." -ForegroundColor Red
}
Write-Host "================================================================================`n" -ForegroundColor Cyan
if (Test-Path $tempConfigFile) {
    Remove-Item $tempConfigFile
    Write-Host "Cleaned up temporary config file: $tempConfigFile`n" -ForegroundColor Gray
}
Write-Host "Experiment 5b (Automatic 0.1x) complete!`n" -ForegroundColor Green
