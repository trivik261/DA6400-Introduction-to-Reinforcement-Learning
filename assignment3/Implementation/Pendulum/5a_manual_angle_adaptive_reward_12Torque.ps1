param(
    [string]$ConfigFile = "config.json"
)
$ExperimentName = "5a_manual_angle_adaptive_reward_12Torque"
$Angles = @(-60, 120, -150)
$Alphas = @(0.2, 0.1, 0.05, 0.025, 0.01)
$RewardScale = 1.0
$RewardType = "adaptive"
$SacType = "manual"
$MaxTorque = 12.0
$NumSeeds = 15
$MaxConcurrent = 8
$TotalRuns = $Angles.Count * $Alphas.Count * $NumSeeds
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "EXPERIMENT 5a: Manual SAC Alpha Tuning - Multiple Angles" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Angles: $($Angles -join ', ')°"
Write-Host "Alphas: $($Alphas -join ', ')"
Write-Host "Reward Type: $RewardType"
Write-Host "Reward Scale: $RewardScale"
Write-Host "Max Torque: $MaxTorque N·m"
Write-Host "Seeds per config: $NumSeeds"
Write-Host "Total runs: $TotalRuns"
Write-Host "Max concurrent: $MaxConcurrent"
Write-Host "Output Directory: outputs/$ExperimentName"
Write-Host "================================================================================`n" -ForegroundColor Cyan
Write-Host "Loading config from: $ConfigFile" -ForegroundColor Yellow
$baseConfig = Get-Content $ConfigFile | ConvertFrom-Json
$baseConfig.training.output_dir = "outputs/$ExperimentName"
$baseConfig.environment.reward_type = $RewardType
$baseConfig.environment.reward_scale = $RewardScale
$baseConfig.environment.max_torque = $MaxTorque
$baseConfig.sac.type = $SacType
$baseConfig.features.enable_live_plot = $false
$baseConfig.features.enable_checkpoints = $true
$baseConfig.features.enable_early_stopping = $true
$tempConfigFile = "config_5a_temp.json"
$baseConfig | ConvertTo-Json -Depth 10 | Set-Content $tempConfigFile
Write-Host "Modified config saved to: $tempConfigFile`n" -ForegroundColor Green
Write-Host "Configuration Summary:" -ForegroundColor Yellow
Write-Host "  Reward Type: $($baseConfig.environment.reward_type)"
Write-Host "  Reward Scale: $($baseConfig.environment.reward_scale)"
Write-Host "  Max Torque: $($baseConfig.environment.max_torque) N·m"
Write-Host "  SAC Type: $($baseConfig.sac.type)"
Write-Host "  Checkpoints: $($baseConfig.features.enable_checkpoints)"
Write-Host "  Early Stopping: $($baseConfig.features.enable_early_stopping)"
Write-Host "  Live Plot: $($baseConfig.features.enable_live_plot)`n"
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
Write-Host "Building job queue..." -ForegroundColor Green
$jobQueue = @()
$jobId = 0
foreach ($angle in $Angles) {
    foreach ($alpha in $Alphas) {
        for ($seed = 0; $seed -lt $NumSeeds; $seed++) {
            $jobQueue += @{
                Id = $jobId
                Angle = $angle
                Alpha = $alpha
                Seed = $seed
            }
            $jobId++
        }
    }
}
Write-Host "Total jobs queued: $($jobQueue.Count)" -ForegroundColor Green
Write-Host "Starting execution with max $MaxConcurrent concurrent processes...`n" -ForegroundColor Yellow
$runningProcesses = @{}
$completedJobs = @()
$nextJobIndex = 0
$startTime = Get-Date
$initialBatch = [Math]::Min($MaxConcurrent, $jobQueue.Count)
for ($i = 0; $i -lt $initialBatch; $i++) {
    $job = $jobQueue[$nextJobIndex]
    Write-Host "  Starting Job $($job.Id): Angle=$($job.Angle)° Alpha=$($job.Alpha) Seed=$($job.Seed)" -ForegroundColor Gray
    $arguments = "train.py --config $tempConfigFile --target_angle $($job.Angle) --alpha $($job.Alpha) --seed $($job.Seed)"
    $process = Start-Process -FilePath "python" -ArgumentList $arguments -PassThru -WindowStyle Minimized
    $runningProcesses[$job.Id] = @{
        Process = $process
        Job = $job
    }
    $nextJobIndex++
    Start-Sleep -Milliseconds 500
}
Write-Host "`nInitial batch of $initialBatch processes started." -ForegroundColor Green
Write-Host "Monitoring processes and starting new ones as they complete...`n" -ForegroundColor Yellow
while ($completedJobs.Count -lt $jobQueue.Count) {
    Start-Sleep -Seconds 5
    $completedNow = @()
    foreach ($jobId in $runningProcesses.Keys) {
        $entry = $runningProcesses[$jobId]
        if ($entry.Process.HasExited) {
            $completedNow += $jobId
            $completedJobs += $entry.Job
            $job = $entry.Job
            Write-Host "  Job $jobId completed: Angle=$($job.Angle)° Alpha=$($job.Alpha) Seed=$($job.Seed) (Exit: $($entry.Process.ExitCode))" -ForegroundColor Green
        }
    }
    foreach ($jobId in $completedNow) {
        $runningProcesses.Remove($jobId)
        if ($nextJobIndex -lt $jobQueue.Count) {
            $job = $jobQueue[$nextJobIndex]
            Write-Host "  Starting Job $($job.Id): Angle=$($job.Angle)° Alpha=$($job.Alpha) Seed=$($job.Seed)" -ForegroundColor Cyan
            $arguments = "train.py --config $tempConfigFile --target_angle $($job.Angle) --alpha $($job.Alpha) --seed $($job.Seed)"
            $process = Start-Process -FilePath "python" -ArgumentList $arguments -PassThru -WindowStyle Minimized
            $runningProcesses[$job.Id] = @{
                Process = $process
                Job = $job
            }
            $nextJobIndex++
            Start-Sleep -Milliseconds 500
        }
    }
    $remaining = $jobQueue.Count - $completedJobs.Count
    $elapsed = (Get-Date) - $startTime
    if ($completedJobs.Count % 10 -eq 0 -and $completedJobs.Count -gt 0 -and $remaining -gt 0) {
        $avgTimePerJob = $elapsed.TotalSeconds / $completedJobs.Count
        $estimatedRemaining = [TimeSpan]::FromSeconds($avgTimePerJob * $remaining)
        Write-Host "`n  Progress: $($completedJobs.Count)/$($jobQueue.Count) completed, $remaining remaining, $($runningProcesses.Count) running" -ForegroundColor Yellow
        Write-Host "  Elapsed: $($elapsed.ToString('hh\:mm\:ss')) | Est. Remaining: $($estimatedRemaining.ToString('hh\:mm\:ss'))`n" -ForegroundColor Yellow
    }
}
$totalTime = (Get-Date) - $startTime
Write-Host "`nAll processes completed!" -ForegroundColor Green
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "EXPERIMENT 5a COMPLETED" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
$resultFiles = Get-ChildItem -Path "outputs/$ExperimentName" -Recurse -Filter "results.json" -ErrorAction SilentlyContinue
$completedCount = $resultFiles.Count
$failedCount = $jobQueue.Count - $completedCount
Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "  Total Jobs: $($jobQueue.Count)" -ForegroundColor White
Write-Host "  Completed: $completedCount" -ForegroundColor $(if ($completedCount -eq $jobQueue.Count) { "Green" } else { "Yellow" })
Write-Host "  Failed/Incomplete: $failedCount" -ForegroundColor $(if ($failedCount -gt 0) { "Red" } else { "Green" })
Write-Host "  Total Time: $($totalTime.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "`nBreakdown by Configuration:" -ForegroundColor Cyan
foreach ($angle in $Angles) {
    foreach ($alpha in $Alphas) {
        $pattern = "*theta$angle*alpha$alpha*"
        $count = (Get-ChildItem -Path "outputs/$ExperimentName" -Directory -Filter $pattern -ErrorAction SilentlyContinue | Measure-Object).Count
        $status = if ($count -eq $NumSeeds) { "OK" } else { "INCOMPLETE ($count/$NumSeeds)" }
        Write-Host "  Angle=$($angle)° Alpha=$alpha : $status" -ForegroundColor $(if ($count -eq $NumSeeds) { "Green" } else { "Red" })
    }
}
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
Write-Host "Experiment 5a complete!`n" -ForegroundColor Green
