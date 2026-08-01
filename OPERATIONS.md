# Enterprise OPERATIONS.md

## System Operations, Logging & Monitoring Manual

### Health Checks
- Liveness Probe: `python -c "from src.mlops.health import HealthChecker; print(HealthChecker.check_liveness())"`
- Readiness Probe: `python -c "from src.mlops.health import HealthChecker; print(HealthChecker.check_readiness())"`

### Log Files
- Log Path: `outputs/logs/app.log` (Rotating file handler, 10 MB per file).
