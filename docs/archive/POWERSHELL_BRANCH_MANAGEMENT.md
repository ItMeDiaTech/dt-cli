# PowerShell Branch Management Guide

Managing multiple Claude Code web session branches made easy!

---

## 🎯 The Problem

Each Claude Code web session creates a new branch with a name like:
```
claude/ubuntu-installation-guide-011CUuijqz9a2fEsDv3PqkdC
claude/implement-logging-011CUue8RHA52jgqpMxdJahs
claude/complete-tasks-011CUugybmYNmBVd3bX2SeGj
```

After multiple sessions, you have **many branches** and it's hard to know:
- Which branch has the latest code?
- Which branches to merge?
- Which branches to delete?

## ✅ The Solution

Three PowerShell scripts to manage this:

1. **`Analyze-ClaudeBranches.ps1`** - Shows what's in each branch
2. **`Merge-ClaudeBranches.ps1`** - Creates main from the latest branch
3. **`Cleanup-ClaudeBranches.ps1`** - Deletes old branches

---

## 🚀 Quick Start

### Step 1: Analyze Your Branches

```powershell
# Download the scripts first (or use from repository)
# Then run:

.\Analyze-ClaudeBranches.ps1
```

**What it does:**
- Lists all Claude branches
- Shows dates, commit counts, and messages
- Tells you which branch is most recent
- Exports detailed CSV report

**Output example:**
```
Found 7 Claude Code branches:

[1] ubuntu-installation-guide
    Date: 2025-11-08 04:15:23 (2h ago)
    Commits: 15 | Last commit: 62c13d1
    Message: docs: Add branch status and merge guide
    Files changed: 2

[2] complete-tasks
    Date: 2025-11-07 15:30:12 (14h ago)
    Commits: 12 | Last commit: 20f9a1a
    Message: feat: Complete remaining tasks
    Files changed: 5

RECOMMENDATION:
Most recent branch: claude/ubuntu-installation-guide-011CUuijqz9a2fEsDv3PqkdC
This branch likely has the most up-to-date code!
```

### Step 2: Merge to Main

```powershell
# Dry run first (see what will happen)
.\Merge-ClaudeBranches.ps1 -DryRun

# Actually create main branch
.\Merge-ClaudeBranches.ps1
```

**What it does:**
- Identifies the most recent branch
- Creates `main` from that branch
- Pushes to GitHub
- Handles conflicts if main already exists

**Interactive prompts:**
```
RECOMMENDED APPROACH:
Use the most recent branch as base for 'main'

Most recent: claude/ubuntu-installation-guide-011CUuijqz9a2fEsDv3PqkdC
Date: 2025-11-08 04:15:23

Continue with this strategy? (yes/no): yes

Creating branch 'main' from ...
✓ Local branch created
✓ Successfully pushed to origin/main

✓ 'main' branch created successfully!
```

### Step 3: Clean Up Old Branches (Optional)

```powershell
# Dry run first
.\Cleanup-ClaudeBranches.ps1 -DryRun

# Delete all except the newest branch
.\Cleanup-ClaudeBranches.ps1

# Keep 2 newest branches
.\Cleanup-ClaudeBranches.ps1 -KeepNewest 2
```

**What it does:**
- Keeps N newest branches (default: 1)
- Deletes all older branches
- Requires confirmation before deleting

---

## 📖 Detailed Usage

### Analyze-ClaudeBranches.ps1

**Basic usage:**
```powershell
.\Analyze-ClaudeBranches.ps1
```

**Parameters:**
```powershell
# Use different repository
.\Analyze-ClaudeBranches.ps1 -RepoUrl "https://github.com/YourUser/YourRepo.git"

# Use different local path
.\Analyze-ClaudeBranches.ps1 -LocalPath "C:\Temp\analysis"
```

**Output:**
- Console: Branch list with dates and details
- File: `branch-analysis.csv` with complete data

**CSV includes:**
- Branch name
- Commit hash
- Date and time
- Commit message
- Total commits
- Files changed

---

### Merge-ClaudeBranches.ps1

**Basic usage:**
```powershell
# Dry run (recommended first)
.\Merge-ClaudeBranches.ps1 -DryRun

# Actually merge
.\Merge-ClaudeBranches.ps1

# Skip confirmation prompts
.\Merge-ClaudeBranches.ps1 -Force
```

**Parameters:**
```powershell
# Different repository
.\Merge-ClaudeBranches.ps1 -RepoUrl "https://github.com/YourUser/YourRepo.git"

# Different local path
.\Merge-ClaudeBranches.ps1 -LocalPath "C:\Temp\merge"

# Force overwrite existing main
.\Merge-ClaudeBranches.ps1 -Force
```

**What happens if main exists:**

The script will ask:
```
WARNING: 'main' branch already exists on remote!

Options:
1. Abort (safe)
2. Create main-new branch instead
3. Force overwrite main (dangerous!)

Choose (1/2/3):
```

**Strategy:**
- Finds the **most recent** branch by commit date
- Uses it as the base for `main`
- Assumption: Most recent = most up-to-date code

---

### Cleanup-ClaudeBranches.ps1

**Basic usage:**
```powershell
# Dry run
.\Cleanup-ClaudeBranches.ps1 -DryRun

# Delete (keep 1 newest)
.\Cleanup-ClaudeBranches.ps1

# Keep 3 newest branches
.\Cleanup-ClaudeBranches.ps1 -KeepNewest 3

# Skip confirmation
.\Cleanup-ClaudeBranches.ps1 -Force
```

**Parameters:**
```powershell
# Different repository
.\Cleanup-ClaudeBranches.ps1 -RepoUrl "https://github.com/YourUser/YourRepo.git"

# Keep more branches
.\Cleanup-ClaudeBranches.ps1 -KeepNewest 2

# Force delete without confirmation
.\Cleanup-ClaudeBranches.ps1 -Force
```

**Safety features:**
- Requires typing "DELETE" to confirm
- Dry run mode to preview
- Keeps N newest by default
- Only deletes `claude/` branches

---

## 🔄 Complete Workflow Example

```powershell
# Step 1: Analyze
.\Analyze-ClaudeBranches.ps1

# Review output, identify most recent branch

# Step 2: Merge (dry run first)
.\Merge-ClaudeBranches.ps1 -DryRun

# Step 3: Merge for real
.\Merge-ClaudeBranches.ps1

# Step 4: Verify on GitHub
# Go to: https://github.com/YourUser/YourRepo

# Step 5: Test installation
git clone https://github.com/YourUser/YourRepo.git
cd YourRepo
.\ubuntu-install.sh  # or install.sh

# Step 6: Clean up old branches
.\Cleanup-ClaudeBranches.ps1 -KeepNewest 1
```

---

## 🎨 Understanding the Output

### Branch Timeline

```
Branch Timeline (oldest to newest):
  2025-11-06 10:00 - local-rag-plugin-maf
  2025-11-07 08:30 - implement-logging
  2025-11-07 12:15 - audit-codebase-docs
  2025-11-07 15:30 - complete-tasks
  2025-11-08 04:15 - ubuntu-installation-guide  ← MOST RECENT
```

**Interpretation:**
- Each session created a new branch
- Most recent = `ubuntu-installation-guide`
- This branch likely has all cumulative changes

### File Comparison

```
Comparing all branches to oldest: local-rag-plugin-maf

ubuntu-installation-guide: 15 files different from oldest branch
complete-tasks: 12 files different from oldest branch
audit-codebase-docs: 8 files different from oldest branch
```

**Interpretation:**
- Shows how much changed since first session
- Higher number = more changes accumulated

---

## ⚠️ Important Notes

### About the Merge Strategy

The scripts use a **simple strategy**:
- **Use the most recent branch as `main`**

This assumes:
- ✅ Each Claude session builds on previous work
- ✅ Most recent branch has all accumulated changes
- ✅ No parallel development on different branches

**If you have parallel work:**
You may need to manually merge specific branches:

```powershell
git checkout -b main origin/claude/branch1
git merge origin/claude/branch2
git merge origin/claude/branch3
git push -u origin main
```

### About Branch Cleanup

**Before deleting branches:**
1. ✅ Verify main branch has all your code
2. ✅ Test the installation from main
3. ✅ Keep at least 1 recent branch as backup

**You can always recover if needed:**
- Deleted branches can be restored within 90 days (GitHub)
- Use GitHub's "Restore branch" feature

---

## 🐛 Troubleshooting

### "git not found"

**Solution:**
```powershell
# Install Git for Windows
winget install Git.Git

# Or download from: https://git-scm.com/download/win
```

### "Permission denied" when pushing

**Solution:**
```powershell
# Configure Git credentials
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Use GitHub Personal Access Token
# Settings → Developer settings → Personal access tokens
```

### "Main branch already exists"

**Solution 1: Create new branch**
```powershell
# Script will prompt to create 'main-new' instead
.\Merge-ClaudeBranches.ps1
# Choose option 2
```

**Solution 2: Force overwrite (dangerous!)**
```powershell
# Only if you're sure!
.\Merge-ClaudeBranches.ps1 -Force
# Choose option 3
```

### Script execution policy error

**Solution:**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or bypass for single script
powershell -ExecutionPolicy Bypass -File .\Analyze-ClaudeBranches.ps1
```

---

## 📊 Example Scenarios

### Scenario 1: First Time User

You have 7 branches, never created main:

```powershell
# 1. See what you have
.\Analyze-ClaudeBranches.ps1

# 2. Create main from most recent
.\Merge-ClaudeBranches.ps1

# 3. Clean up old branches
.\Cleanup-ClaudeBranches.ps1 -KeepNewest 1
```

### Scenario 2: Main Exists, Want to Update

Main exists but old, want to update from new session:

```powershell
# 1. Analyze to find newest
.\Analyze-ClaudeBranches.ps1

# 2. Create main-new from newest
.\Merge-ClaudeBranches.ps1
# Choose option 2 to create 'main-new'

# 3. Manually review and rename
# On GitHub: Delete old main, rename main-new to main
```

### Scenario 3: Keep Multiple Branches

You want to keep several recent branches as backup:

```powershell
# 1. Merge to main
.\Merge-ClaudeBranches.ps1

# 2. Keep 3 newest branches
.\Cleanup-ClaudeBranches.ps1 -KeepNewest 3
```

---

## 🔧 Advanced Usage

### Custom Branch Selection

If you want to use a specific branch (not the newest):

```powershell
# Manual approach
git clone https://github.com/YourUser/YourRepo.git
cd YourRepo
git fetch --all

# List branches
git branch -r

# Create main from specific branch
git checkout -b main origin/claude/specific-branch-name
git push -u origin main
```

### Comparing Specific Branches

```powershell
git clone https://github.com/YourUser/YourRepo.git
cd YourRepo
git fetch --all

# Compare two branches
git diff origin/claude/branch1 origin/claude/branch2

# See file list
git diff --name-only origin/claude/branch1 origin/claude/branch2
```

### Export Branch Data

```powershell
# The analyze script creates branch-analysis.csv
.\Analyze-ClaudeBranches.ps1

# Open in Excel
.\branch-analysis.csv

# Or use PowerShell
Import-Csv .\branch-analysis.csv | Out-GridView
```

---

## ✅ Best Practices

1. **Always run Analyze first**
   - Understand what you have before merging

2. **Use -DryRun before executing**
   - See what will happen without risk

3. **Keep one recent branch as backup**
   - Don't delete all Claude branches immediately

4. **Test main branch after creation**
   - Clone and run installation to verify

5. **Document your sessions**
   - Use meaningful commit messages in Claude sessions

6. **Regular cleanup**
   - Run cleanup after each few Claude sessions

---

## 📚 See Also

- [BRANCH_STATUS.md](./BRANCH_STATUS.md) - Current branch status
- [MERGE_TO_MAIN.md](./MERGE_TO_MAIN.md) - Manual merge instructions
- [README.md](./README.md) - Project overview
- [UBUNTU_DEPLOYMENT_GUIDE.md](./UBUNTU_DEPLOYMENT_GUIDE.md) - Installation guide

---

## 💡 Tips

**For Windows Users:**
- Use PowerShell (not Command Prompt)
- Run as Administrator if you get permission errors
- Install Git for Windows first

**For Git Bash Users:**
- These are PowerShell scripts (.ps1)
- Use PowerShell, not Git Bash
- Or convert to bash scripts if needed

**For Mac/Linux Users:**
- Use PowerShell Core (cross-platform)
- Or adapt scripts to bash/zsh

---

**Happy branch management!** 🚀

If you encounter issues, the scripts are designed to be safe with confirmations and dry-run modes.
