# PowerShell Script: Cleanup Old Claude Code Branches
# This script helps you delete old Claude branches after merging to main

param(
    [string]$RepoUrl = "https://github.com/ItMeDiaTech/dt-cli.git",
    [string]$LocalPath = ".\dt-cli-cleanup",
    [switch]$DryRun = $false,
    [int]$KeepNewest = 1,
    [switch]$Force = $false
)

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Claude Code Branch Cleanup" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "DRY RUN MODE - No changes will be made" -ForegroundColor Yellow
    Write-Host ""
}

# Clone repository if it doesn't exist
if (-not (Test-Path $LocalPath)) {
    Write-Host "Cloning repository..." -ForegroundColor Yellow
    git clone $RepoUrl $LocalPath
} else {
    Write-Host "Using existing repository..." -ForegroundColor Yellow
}

Push-Location $LocalPath
git fetch --all --prune

# Get all claude branches
$branches = git branch -r | Where-Object { $_ -match "origin/claude/" } | ForEach-Object { $_.Trim() }

Write-Host "Found $($branches.Count) Claude Code branches" -ForegroundColor Green
Write-Host ""

# Collect branch information
$branchInfo = @()
foreach ($branch in $branches) {
    $branchName = $branch -replace "origin/", ""
    $commitDate = git log -1 --format="%ai" $branch

    $branchInfo += [PSCustomObject]@{
        FullName = $branch
        BranchName = $branchName
        Date = [DateTime]::Parse($commitDate)
    }
}

# Sort by date (newest first)
$branchInfo = $branchInfo | Sort-Object -Property Date -Descending

# Determine which to keep and which to delete
$toKeep = $branchInfo | Select-Object -First $KeepNewest
$toDelete = $branchInfo | Select-Object -Skip $KeepNewest

Write-Host "Cleanup Plan:" -ForegroundColor Cyan
Write-Host ""
Write-Host "KEEPING ($KeepNewest newest):" -ForegroundColor Green
foreach ($branch in $toKeep) {
    Write-Host "  ✓ $($branch.BranchName)" -ForegroundColor Green
    Write-Host "    $($branch.Date.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
}
Write-Host ""

if ($toDelete.Count -gt 0) {
    Write-Host "DELETING ($($toDelete.Count) old branches):" -ForegroundColor Red
    foreach ($branch in $toDelete) {
        Write-Host "  ✗ $($branch.BranchName)" -ForegroundColor Red
        Write-Host "    $($branch.Date.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
    }
    Write-Host ""

    if (-not $Force -and -not $DryRun) {
        Write-Host "WARNING: This will permanently delete $($toDelete.Count) branches!" -ForegroundColor Yellow
        $confirm = Read-Host "Are you sure? (type 'DELETE' to confirm)"
        if ($confirm -ne "DELETE") {
            Write-Host "Aborted." -ForegroundColor Red
            Pop-Location
            exit
        }
    }

    Write-Host ""
    Write-Host "Deleting branches..." -ForegroundColor Yellow

    foreach ($branch in $toDelete) {
        $remoteBranch = $branch.BranchName

        if (-not $DryRun) {
            Write-Host "Deleting $remoteBranch..." -NoNewline
            git push origin --delete $remoteBranch 2>$null

            if ($LASTEXITCODE -eq 0) {
                Write-Host " ✓" -ForegroundColor Green
            } else {
                Write-Host " ✗ Failed" -ForegroundColor Red
            }
        } else {
            Write-Host "[DRY RUN] Would delete: $remoteBranch" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "No branches to delete (only $($branchInfo.Count) branch(es) exist)" -ForegroundColor Green
}

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

if (-not $DryRun) {
    Write-Host "Kept: $($toKeep.Count) branch(es)" -ForegroundColor Green
    Write-Host "Deleted: $($toDelete.Count) branch(es)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Cleanup complete!" -ForegroundColor Green
} else {
    Write-Host "DRY RUN complete - no changes made" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run without -DryRun to execute:" -ForegroundColor White
    Write-Host "  .\Cleanup-ClaudeBranches.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "  -KeepNewest 2     Keep 2 newest branches" -ForegroundColor White
    Write-Host "  -Force            Skip confirmation" -ForegroundColor White
}

Write-Host ""

Pop-Location
