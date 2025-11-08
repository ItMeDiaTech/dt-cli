# PowerShell Script: Intelligently Merge Claude Code Branches to Main
# This script analyzes branches and creates a clean main branch

param(
    [string]$RepoUrl = "https://github.com/ItMeDiaTech/dt-cli.git",
    [string]$LocalPath = ".\dt-cli-merge",
    [switch]$DryRun = $false,
    [switch]$Force = $false
)

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Claude Code Branch Merger" -ForegroundColor Cyan
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
    Push-Location $LocalPath
    git fetch --all --prune
    Pop-Location
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
    $commitHash = git rev-parse $branch
    $commitMessage = git log -1 --format="%s" $branch

    $branchInfo += [PSCustomObject]@{
        FullName = $branch
        BranchName = $branchName
        CommitHash = $commitHash
        Date = [DateTime]::Parse($commitDate)
        Message = $commitMessage
    }
}

# Sort by date (oldest to newest for proper merging)
$branchInfo = $branchInfo | Sort-Object -Property Date

Write-Host "Branch Timeline (oldest to newest):" -ForegroundColor Cyan
Write-Host ""
foreach ($info in $branchInfo) {
    Write-Host "  $($info.Date.ToString('yyyy-MM-dd HH:mm')) - " -NoNewline -ForegroundColor Gray
    Write-Host "$($info.BranchName -replace 'claude/', '' -replace '-011.*', '')" -ForegroundColor White
}
Write-Host ""

# Strategy: Use the most recent branch as the base for main
$mostRecent = $branchInfo | Select-Object -Last 1

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Merge Strategy" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "RECOMMENDED APPROACH:" -ForegroundColor Green
Write-Host "Use the most recent branch as base for 'main'" -ForegroundColor White
Write-Host ""
Write-Host "Most recent: " -NoNewline
Write-Host "$($mostRecent.BranchName)" -ForegroundColor Yellow
Write-Host "Date: $($mostRecent.Date.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
Write-Host "Reason: Most likely contains all cumulative changes" -ForegroundColor Gray
Write-Host ""

if (-not $Force -and -not $DryRun) {
    $confirm = Read-Host "Continue with this strategy? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Aborted." -ForegroundColor Red
        Pop-Location
        exit
    }
}

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Creating Main Branch" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if main already exists
$mainExists = git ls-remote --heads origin main

if ($mainExists -and -not $Force) {
    Write-Host "WARNING: 'main' branch already exists on remote!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "1. Abort (safe)" -ForegroundColor White
    Write-Host "2. Create main-new branch instead" -ForegroundColor White
    Write-Host "3. Force overwrite main (dangerous!)" -ForegroundColor Red
    Write-Host ""
    $choice = Read-Host "Choose (1/2/3)"

    switch ($choice) {
        "1" {
            Write-Host "Aborted." -ForegroundColor Red
            Pop-Location
            exit
        }
        "2" {
            $targetBranch = "main-new"
            Write-Host "Will create '$targetBranch' instead" -ForegroundColor Yellow
        }
        "3" {
            $targetBranch = "main"
            Write-Host "WARNING: Will overwrite main!" -ForegroundColor Red
        }
        default {
            Write-Host "Invalid choice. Aborted." -ForegroundColor Red
            Pop-Location
            exit
        }
    }
} else {
    $targetBranch = "main"
}

if (-not $DryRun) {
    Write-Host "Creating branch '$targetBranch' from $($mostRecent.BranchName)..." -ForegroundColor Yellow

    # Create local branch from most recent
    git checkout -B $targetBranch $mostRecent.FullName

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Local branch created" -ForegroundColor Green

        # Push to remote
        Write-Host "Pushing to remote..." -ForegroundColor Yellow

        if ($Force -and $targetBranch -eq "main") {
            git push -f origin $targetBranch
        } else {
            git push -u origin $targetBranch
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Successfully pushed to origin/$targetBranch" -ForegroundColor Green
        } else {
            Write-Host "✗ Failed to push to remote" -ForegroundColor Red
            Write-Host "You may need to push manually with:" -ForegroundColor Yellow
            Write-Host "  git push -u origin $targetBranch" -ForegroundColor White
        }
    } else {
        Write-Host "✗ Failed to create local branch" -ForegroundColor Red
    }
} else {
    Write-Host "[DRY RUN] Would create '$targetBranch' from:" -ForegroundColor Yellow
    Write-Host "  $($mostRecent.BranchName)" -ForegroundColor White
}

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

if (-not $DryRun) {
    Write-Host "✓ '$targetBranch' branch created successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "This branch contains code from:" -ForegroundColor White
    Write-Host "  $($mostRecent.BranchName)" -ForegroundColor Yellow
    Write-Host "  Last updated: $($mostRecent.Date.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Review the branch on GitHub" -ForegroundColor White
    Write-Host "2. Test the installation:" -ForegroundColor White
    Write-Host "   git clone https://github.com/ItMeDiaTech/dt-cli.git" -ForegroundColor Gray
    Write-Host "   cd dt-cli" -ForegroundColor Gray
    Write-Host "   ./ubuntu-install.sh" -ForegroundColor Gray
    Write-Host "3. Optionally clean up old branches with:" -ForegroundColor White
    Write-Host "   .\Cleanup-ClaudeBranches.ps1" -ForegroundColor Yellow
} else {
    Write-Host "DRY RUN complete - no changes made" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run without -DryRun to execute:" -ForegroundColor White
    Write-Host "  .\Merge-ClaudeBranches.ps1" -ForegroundColor Yellow
}

Write-Host ""

Pop-Location
