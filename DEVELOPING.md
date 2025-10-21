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
