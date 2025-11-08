# PowerShell Script: Analyze All Claude Code Branches
# This script shows you what's in each branch and which is most recent

param(
    [string]$RepoUrl = "https://github.com/ItMeDiaTech/dt-cli.git",
    [string]$LocalPath = ".\dt-cli-analysis"
)

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Claude Code Branch Analyzer" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Clone repository if it doesn't exist
if (-not (Test-Path $LocalPath)) {
    Write-Host "Cloning repository..." -ForegroundColor Yellow
    git clone $RepoUrl $LocalPath
} else {
    Write-Host "Updating repository..." -ForegroundColor Yellow
    Push-Location $LocalPath
    git fetch --all
    Pop-Location
}

Push-Location $LocalPath
git fetch --all --prune

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Branch Analysis" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Get all claude branches
$branches = git branch -r | Where-Object { $_ -match "origin/claude/" } | ForEach-Object { $_.Trim() }

$branchInfo = @()

foreach ($branch in $branches) {
    $branchName = $branch -replace "origin/", ""

    # Get last commit info
    $commitHash = git rev-parse $branch
    $commitDate = git log -1 --format="%ai" $branch
    $commitMessage = git log -1 --format="%s" $branch
    $commitAuthor = git log -1 --format="%an" $branch
    $commitCount = git rev-list --count $branch

    # Get files changed in last commit
    $filesChanged = git diff-tree --no-commit-id --name-only -r $commitHash | Measure-Object | Select-Object -ExpandProperty Count

    $branchInfo += [PSCustomObject]@{
        Branch = $branchName
        ShortName = ($branchName -replace "claude/", "" -replace "-011.*", "")
        CommitHash = $commitHash.Substring(0, 7)
        Date = [DateTime]::Parse($commitDate)
        Message = $commitMessage
        Author = $commitAuthor
        TotalCommits = $commitCount
        FilesInLastCommit = $filesChanged
    }
}

# Sort by date (newest first)
$branchInfo = $branchInfo | Sort-Object -Property Date -Descending

# Display summary
Write-Host "Found $($branchInfo.Count) Claude Code branches:" -ForegroundColor Green
Write-Host ""

$counter = 1
foreach ($info in $branchInfo) {
    $age = (Get-Date) - $info.Date
    $ageStr = if ($age.Days -gt 0) { "$($age.Days)d ago" } else { "$($age.Hours)h ago" }

    Write-Host "[$counter] " -NoNewline -ForegroundColor Yellow
    Write-Host "$($info.ShortName)" -ForegroundColor Cyan
    Write-Host "    Full: $($info.Branch)" -ForegroundColor Gray
    Write-Host "    Date: $($info.Date.ToString('yyyy-MM-dd HH:mm:ss')) ($ageStr)" -ForegroundColor Gray
    Write-Host "    Commits: $($info.TotalCommits) | Last commit: $($info.CommitHash)" -ForegroundColor Gray
    Write-Host "    Message: $($info.Message)" -ForegroundColor White
    Write-Host "    Files changed: $($info.FilesInLastCommit)" -ForegroundColor Gray
    Write-Host ""
    $counter++
}

# Find the most recent branch
$mostRecent = $branchInfo | Select-Object -First 1
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "RECOMMENDATION" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Most recent branch: " -NoNewline
Write-Host "$($mostRecent.Branch)" -ForegroundColor Yellow
Write-Host "Last updated: $($mostRecent.Date.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
Write-Host "This branch likely has the most up-to-date code!" -ForegroundColor Green
Write-Host ""

# Show unique files in each branch compared to oldest
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "File Comparison" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

$oldest = $branchInfo | Select-Object -Last 1
Write-Host "Comparing all branches to oldest: $($oldest.ShortName)" -ForegroundColor Gray
Write-Host ""

foreach ($info in $branchInfo | Select-Object -First 5) {
    git checkout $info.Branch -q 2>$null
    $uniqueFiles = git diff --name-only "origin/$($oldest.Branch)" | Measure-Object | Select-Object -ExpandProperty Count

    Write-Host "$($info.ShortName): " -NoNewline -ForegroundColor Cyan
    Write-Host "$uniqueFiles files different from oldest branch" -ForegroundColor White
}

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Next Steps" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Review the most recent branch (likely has all changes)" -ForegroundColor White
Write-Host "2. Run: " -NoNewline -ForegroundColor White
Write-Host ".\Merge-ClaudeBranches.ps1" -ForegroundColor Yellow
Write-Host "3. This will intelligently merge branches into main" -ForegroundColor White
Write-Host ""

# Export to CSV for detailed analysis
$csvPath = ".\branch-analysis.csv"
$branchInfo | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host "Detailed analysis exported to: $csvPath" -ForegroundColor Green

Pop-Location

Write-Host ""
Write-Host "Analysis complete!" -ForegroundColor Green
