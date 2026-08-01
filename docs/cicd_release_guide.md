# CI/CD & Release Strategy (Step 148)

## 1. Automated Pipelines
- **CI Workflow (`.github/workflows/ci.yml`)**: Runs on pull requests. Executes linting and unit tests.
- **CD Workflow (`.github/workflows/cd.yml`)**: Runs on release tags (`vX.Y.Z`). Builds Docker image.
