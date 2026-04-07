# 👾 Axiom MLOps: Real-Time Funding Prediction Engine

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
![W&B Tracking](https://img.shields.io/badge/W&B-Active-orange)

A production-ready machine learning pipeline designed to intercept funding failures instantly and route users to alternative funding paths before they permanently churn.


## 🛑 Problem Statement: Real-Time Funding Prediction Model

**BLUF (Bottom Line Up Front):**
VodaPay is currently losing 6 out of 10 users at the funding stage due to a 59.3% failure rate on Instant EFT. We are bleeding acquisition budget into a broken funnel because we rely on retroactive reporting rather than real-time intervention. We must build a predictive model to intercept funding failures instantly and route users to alternative funding paths before they permanently churn.

### 1. The Problem
* **The Broken Front Door:** Instant EFT, our primary funding mechanism, fails 59.3% of the time.
* **User Ghosting:** Because of this friction, 38.1% of all created cards hold a lifetime balance of R0.00. For new cohorts, this rises to 56.8%.
* **The Terminal Dead-End:** When users fail to fund or hit an error, the recovery rate within one hour is exactly 0.0%. We currently offer no dynamic recovery path.

### 2. The Commercial Impact
We are successfully acquiring users, but structurally preventing them from executing. A user cannot spend what they cannot load.
Currently, our internal teams cannot distinguish between a Technical Outage (the gateway crashed) and User Abandonment (the user hesitated and closed the app) until days later. By the time the data is reviewed, the user’s "Golden Window" (1 to 24 hours post-registration) has closed, and the acquisition cost is completely wasted.

### 3. The ML Solution
We need to shift from passive monitoring to active, real-time triage. We must build and deploy a prediction model (e.g., XGBoost or Isolation Forests) to monitor funding attempts as they happen.
The model will:
* **Identify the Failure Root Cause:** Instantly classify whether an EFT drop-off is a system timeout or a behavioural exit.
* **Trigger Real-Time Recovery:** If a high-value user experiences a gateway failure, the model will instantly trigger a UI intervention, such as a "Card-to-Card Transfer" prompt, preventing the 0.0% churn loop.

### 4. Testable Hypotheses & Success Metrics
* **Hypothesis:** By predicting failure root causes in real-time and providing alternative funding prompts, we will reduce funding abandonment and decrease the 38.1% Ghosting Rate.
* **Primary Metric:** Lift the 1-hour error recovery rate from 0.0% to >10%.
* **Secondary Metric:** Reduce the overall Instant EFT failure rate below 20%.
* **Commercial ROI:** Activating just 50% of our dormant user base unlocks a potential **+R2 Billion** in transaction volume.


## ✨ Key Features
* **Real-Time Triage:** Configured to instantly classify system timeouts vs. behavioral exits.
* **Automated Experiment Tracking:** Fully integrated with Weights & Biases for monitoring model drift and feature importance.
* **Reproducible Environments:** Deterministic dependency management powered by `uv`.


## 🗄️ Dataset Information
The model for this Proof of Concept (PoC) is trained exclusively on the **Funding Table**. It utilizes core transactional features to establish a baseline for predicting funding failures and pinpointing structural drop-offs.

**Core Schema:**
* `actor_role_id`: Unique identifier for the transacting user.
* `created_time`: Timestamp of the funding attempt.
* `fund_type`: The funding mechanism/channel utilized (e.g., `AGENT_TOPUP_FOR_USER_CLEARING`).
* `fund_amount`: The monetary value of the attempted transaction.
* `fund_order_status`: The target variable/outcome of the transaction (e.g., `SUCCESS` or `FAILED`).

*Note: Raw data files (`.csv`) are strictly ignored via `.gitignore` to protect proprietary IP and PII.*


## 📂 Project Structure

```text
.
├── assets/              # Documentation assets (logos, images)
├── src/axiom_mlops_project/
│   ├── main.py          # Core training logic
│   └── __init__.py
├── tests/               # Unit and integration tests
├── pyproject.toml       # Dependency management (uv)
└── LICENSE              # MIT License
```


## ⚙️ Configuration & Setup

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

3. **Configure Experiment Tracking:**
You must authenticate your machine with Weights & Biases before running the pipeline:
```bash
wandb login
```


## 🚀 Usage

To run the diagnostic training pipeline and log results:
```bash
uv run src/axiom_mlops_project/main.py
```


## 📊 Model Performance

Current baseline metrics for the Funding Prediction Model:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.85 |
| **Precision (Failure Detection)** | 0.81 |
| **Model Architecture** | XGBoost / Isolation Forest |


## 🏗️ Architecture & Philosophy

**Platform Agnostic by Design**
This pipeline is built to avoid vendor lock-in. By utilizing standard Python and Git, the engine can be executed anywhere without relying on proprietary cloud wrappers.

**Experiment Tracking: W&B vs. MLflow**
For this iteration, we selected **Weights & Biases (W&B)** over MLflow. W&B provides a zero-setup managed cloud infrastructure, superior real-time dashboarding, and seamless collaboration out of the box, allowing the lab to focus entirely on model iteration and revenue recovery.


## 🛡 Governance & Contributing

This project is governed by **Axiom Research Lab**.

* **Branch Protection:** Changes must pass through Pull Requests on `main`.
* **License:** Distributed under the MIT License. See `LICENSE` for more information.
