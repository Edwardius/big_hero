# Development Guide

## Using Dev Containers

This project includes VS Code dev container configurations to develop code in an isolated stable environment. This also simplifies and automates software installation of our stack.

### Prerequisites

- Docker rootless installed and running
- VS Code with the "Dev Containers" extension installed

### Getting Started

1. **Choose ROS distribution**:
   - Press `Ctrl+Shift+P` and run "Tasks: Run Task"
   - Select "Setup Devcontainer (CPU Only)" or "Setup Devcontainer (GPU)" and follow prompts
      - **Tested configurations:** Ubuntu 24.04 CUDA 12.9 ROS2 Jammy , Ubuntu 24.04 CUDA 13.0 ROS2 Jammy

2. **Rebuild and open in container**:
   - Press `Ctrl+Shift+P` and run "Dev Containers: Rebuild and Reopen in Container" or the other variants
   - The container will automatically rebuild and reopen your selected configuration, you are now inside a devcontainer!
      - changes inside the container will propagate outside the container
      - git works in the container
      - claude works in the container

### Container Features

- **Workspace**: Your code is mounted at `/workspaces/big_hero`
- **ROS sourcing**: ROS is automatically sourced in your shell
- **Build tools**: Includes `colcon` and `rosdep` for ROS development
- **Extensions**: C++, CMake, Python, and XML support pre-installed
- **Developer Tools**: Git, Claude, UnityHub are all pre-installed

---

## Monorepo Management

### Adding a New Repository

```bash
git remote add <name> <url>
git fetch <name> --no-tags
git subtree add --prefix=<directory> <name> <branch>
git remote remove <name>
```

**If repo has submodules:**
```bash
cat <directory>/.gitmodules  # Check what submodules exist
git remote add <sub-name> <sub-url>
git fetch <sub-name> --no-tags
rmdir <directory>/path/to/submodule
git add <directory>/path/to/submodule && git commit -m "Remove empty submodule"
git subtree add --prefix=<directory>/path/to/submodule <sub-name> <branch>
rm <directory>/.gitmodules && git add <directory>/.gitmodules && git commit -m "Remove .gitmodules"
```

### Pruning Large Files from History

```bash
# Find large files (>50MB)
git rev-list --objects --all | awk '{print $1}' | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '$1=="blob" && $3 > 50000000 {print $3/1024/1024 " MB", $4}'

# Find all paths the file appears at
git log --all --pretty=format: --name-only | grep "<filename>"

# Remove from each path
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force \
  --index-filter 'git rm --cached --ignore-unmatch "<path>"' \
  --prune-empty --tag-name-filter cat -- --all

# Cleanup and push
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force origin main
```

### Removing a Repository

**Simple removal (keeps history):**
```bash
git rm -r <directory>
git commit -m "Remove <directory>"
```

**Complete removal (rewrites history):**
```bash
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force \
  --index-filter 'git rm -rf --cached --ignore-unmatch <directory>' \
  --prune-empty --tag-name-filter cat -- --all
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force origin main
```
