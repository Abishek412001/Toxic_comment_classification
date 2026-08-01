# Source Code Module Organization (Step 142)

## 1. Clean Architecture Layer Responsibilities

```
Presentation Layer (dashboard/, api/)
         ↓
Domain & Pipeline Layer (sentiment/, emotion/, xai/, visualization/)
         ↓
Model & Feature Layer (models/, features/, preprocessing/)
         ↓
Infrastructure Layer (mlops/, common/, utils/)
```

---

## 2. Package Boundaries & Import Strategy
- All imports follow explicit module references: `from src.<package>.<module> import <Class>`.
- Core abstract interfaces defined in `src/common/` and inherited across modules (`BaseModel`, `BaseExplainer`).
