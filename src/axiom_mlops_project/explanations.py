import shap
import matplotlib.pyplot as plt
from pathlib import Path
from prefect import task, flow


@task(name="Generate Global Explanations")
def get_global_explanations(model, X_train, output_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    plt.figure(figsize=(10, 6))
    # For classifiers, we take the SHAP values for the positive class (Success)
    shap.summary_plot(shap_values[:, :, 1], X_train, show=False, plot_type="bar")
    plt.savefig(Path(output_path) / "global_importance.png")
    plt.close()
    return "Global Plot Generated"


@task(name="Generate Local Explanation")
def get_local_explanation(model, sample_row, output_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample_row)
    plt.figure(figsize=(10, 4))
    # Focus on Class 1 (Success) for the waterfall plot
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    plt.savefig(Path(output_path) / "local_explanation_sample.png")
    plt.close()
    return "Local Plot Generated"


@flow(name="Axiom Explainability Pipeline")
def explanations_flow(model, data):
    output_dir = Path("artifacts/explanations")
    output_dir.mkdir(parents=True, exist_ok=True)
    get_global_explanations(model, data, str(output_dir))
    get_local_explanation(model, data.head(1), str(output_dir))
