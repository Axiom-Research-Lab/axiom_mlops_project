"""
Axiom Funding Prediction Engine: Alerting Service.

This module provides a unified interface for system notifications,
supporting local desktop alerts and cloud-based monitoring alerts via W&B.
"""

from typing import Literal
import wandb
from loguru import logger
from plyer import notification


class AlertService:
    """
    Handles multi-channel alerting for the Axiom MLOps pipeline.
    """

    def __init__(self, app_name: str = "AxiomEngine", enable_local: bool = True):
        self.app_name = app_name
        self.enable_local = enable_local

    def send(
        self, title: str, message: str, severity: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    ) -> None:
        """
        Dispatches an alert to enabled channels based on severity.
        """
        log_msg = f"[{severity}] {title}: {message}"
        if severity == "HIGH":
            logger.error(log_msg)
        elif severity == "MEDIUM":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        if self.enable_local:
            try:
                notification.notify(
                    title=f"{self.app_name} | {severity}", message=message, timeout=10
                )
            except Exception as e:
                logger.debug(f"Local notification failed: {e}")

        if wandb.run is not None:
            wb_level = wandb.AlertLevel.ERROR if severity == "HIGH" else wandb.AlertLevel.WARN
            wandb.alert(title=title, text=f"Severity: {severity}\n\n{message}", level=wb_level)


alerts = AlertService()
