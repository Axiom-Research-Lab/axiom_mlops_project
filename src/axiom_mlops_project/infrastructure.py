import psutil
from prefect import task, get_run_logger


@task(name="Log Infrastructure Health")
def log_system_metrics():
    logger = get_run_logger()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_usage = psutil.disk_usage("/")

    logger.info(f"CPU: {cpu_usage}% | RAM: {memory_info.percent}%")

    return {"cpu": cpu_usage, "memory": memory_info.percent, "disk": disk_usage.percent}


if __name__ == "__main__":
    from prefect import flow

    @flow(name="Axiom Health Check")
    def check_health():
        log_system_metrics()

    check_health()
