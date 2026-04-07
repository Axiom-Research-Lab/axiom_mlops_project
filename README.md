# 👾 Axiom MLOps: Funding Engine

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
![W&B Tracking](https://img.shields.io/badge/W&B-Active-orange)

A production-ready machine learning pipeline designed to predict research funding outcomes. This project serves as the core engine for **Axiom Research Lab**, focusing on reproducibility and automated experiment tracking.

---

## 📂 Project Structure

```text
.
├── src/axiom_mlops_project/
│   ├── main.py          # Core training logic
│   └── __init__.py
├── tests/               # Unit and integration tests
├── pyproject.toml       # Dependency management (uv)
└── LICENSE              # MIT License
```

---

## 🛠 Installation & Setup

Ensure you have [uv](https://docs.astral.sh/uv/) installed.

1. **Clone the repository:**
```bash
git clone git@github.com:Axiom-Research-Lab/axiom_mlops_project.git
cd axiom_mlops_project
```

2. **Sync dependencies:**
```bash
uv sync
```

---

## 🚀 Usage

To run the training pipeline and log results to Weights & Biases:
```bash
uv run src/axiom_mlops_project/main.py
```

---

## 📊 Model Performance

Current baseline metrics for the Funding Predictor:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.82 |
| **F1-Score** | 0.79 |
| **Model** | Logistic Regression |

---

## 🏗️ Architecture & Philosophy

**Platform Agnostic by Design**
This pipeline is built to avoid vendor lock-in. By utilizing standard Python, `uv` for deterministic dependency management, and Git, the engine can be executed anywhere—from a local Mac environment to AWS, GCP, or Azure—without relying on proprietary cloud wrappers.

**Experiment Tracking: W&B vs. MLflow**
For this iteration, we selected **Weights & Biases (W&B)** over MLflow. While MLflow is a powerful open-source standard, W&B provides a zero-setup managed cloud infrastructure, superior real-time dashboarding, and seamless collaboration out of the box. This allows the lab to focus entirely on model iteration rather than maintaining and scaling a backend tracking server.

---

## 🛡 Governance & Contributing

This project is governed by **Axiom Research Lab**.

* **Branch Protection:** Changes must pass through Pull Requests on `main`.
* **License:** Distributed under the MIT License. See `LICENSE` for more information.

---

## 💻 Cloud Workstations & Collaboration
This project is optimized for **GitHub Codespaces** to ensure a standardized Axiom Research environment.

* **Instant Setup:** Click the **Code** button on GitHub and select **Create codespace on main**.
* **Pre-configured:** The `.devcontainer` automatically installs `uv`, Python 3.13, and essential VS Code extensions (Ruff, W&B).
* **Collaboration:** Use **VS Code Live Share** within the workstation for real-time pair programming and shared terminal debugging on the VodaPay funding model.
