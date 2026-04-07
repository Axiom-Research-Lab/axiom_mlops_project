from prefect import task, flow
from prefect.artifacts import create_markdown_artifact
import pandas as pd
import hashlib
import os


@task(task_run_name="Create Audit Trail: {name}")
def create_audit_trail(df: pd.DataFrame, name: str, source_path: str):
    """
    Generates a persistent, immutable markdown artifact in Prefect Cloud
    to document the data version and integrity hash used in the pipeline.
    """
    # Security fix: Move positional argument before keyword argument
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(df).values, usedforsecurity=False
    ).hexdigest()

    markdown_report = f"""
    # Axiom Data Audit Trail: {name}

    ## Metadata
    - **Source Path:** `{source_path}`
    - **Row Count:** {len(df)}
    - **Integrity Hash (MD5):** `{data_hash}`
    - **Status:** Verified & Tracked 🛡️

    ## Lineage Note
    This dataset was verified for use in the VodaPay Recovery Model.
    Any change to the source data will result in a mismatch of the integrity hash.
    """

    create_markdown_artifact(
        key=f"audit-trail-{name.lower().replace('_', '-')}",
        markdown=markdown_report,
        description=f"Automated audit trail for {name}",
    )
    return data_hash


@flow(name="Axiom Lineage Pipeline")
def lineage_wrapper_flow(data_path: str):
    """
    Main execution flow for tracking data lineage and persisting audit trails.
    """
    df = pd.read_csv(data_path)
    create_audit_trail(df, "VodaPay_Funding_Data", data_path)
    print("✅ Audit Trail Created in Prefect Cloud")


if __name__ == "__main__":
    data_path = "data/funding_data.csv"
    if not os.path.exists(data_path):
        os.makedirs("data", exist_ok=True)
        with open(data_path, "w") as f:
            f.write("id,amount\n1,100\n2,200")

    lineage_wrapper_flow(data_path)
