"""
Axiom Funding Prediction Engine: Monitoring & Validation Job.

Performs drift analysis and performance validation. Automatically
triggers high-severity alerts if model quality falls below defined thresholds.
"""

from pathlib import Path
import pandas as pd
import wandb
from loguru import logger
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from axiom_mlops_project.services import alerts


class MonitoringJob:
    """
    Orchestrates model health checks and automated alerting.
    """

    def __init__(self, reference_path: str, current_path: str):
        self.base_dir = Path(__file__).resolve().parents[3]
        self.ref_data = self.base_dir / reference_path
        self.cur_data = self.base_dir / current_path
        self.report_out = self.base_dir / "reports" / "drift_report.html"
        self.report_out.parent.mkdir(exist_ok=True)

    def execute(self, current_precision: float, threshold: float = 0.80):
        """
        Runs the monitoring pipeline and validates performance.
        """
        logger.info("Initializing Axiom Monitoring Suite...")

        if current_precision < threshold:
            alerts.send(
                title="Model Performance Crisis",
                message=f"Precision dropped to {current_precision:.2f} (Threshold: {threshold})",
                severity="HIGH",
            )
            raise ValueError(f"Performance threshold breached: {current_precision}")

        run = wandb.init(project="axiom-mlops-project", job_type="monitoring")

        ref_df = pd.read_csv(self.ref_data)
        cur_df = pd.read_csv(self.cur_data)

        report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        report.save_html(str(self.report_out))

        run.log({"drift_analysis": wandb.Html(open(self.report_out).read())})

        drift_status = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
        run.summary["drift_detected"] = drift_status

        if drift_status:
            alerts.send(
                title="Data Drift Detected",
                message="Statistical divergence found in VodaPay input features.",
                severity="MEDIUM",
            )
        else:
            logger.success("Monitoring complete. No anomalies detected.")

        run.finish()


if __name__ == "__main__":
    job = MonitoringJob("data/funding_data.csv", "data/funding_data.csv")
    try:
        # Testing with a simulated failure (0.74 < 0.80)
        job.execute(current_precision=0.74)
    except ValueError:
        logger.error("Monitoring job halted due to performance failure.")
