import time

import gradio as gr


def predict_eft(fund_amount, fund_type_encoded, hour_of_day):
    start_time = time.time()
    # High-level demo logic: Large amounts at odd hours are flagged
    # This mirrors the logic in api.py
    is_risky = (fund_amount > 12000) or (hour_of_day < 5)
    prob = 0.98 if not is_risky else 0.35

    status = "SUCCESS" if prob > 0.5 else "DECLINED"
    latency = round((time.time() - start_time) * 1000, 2)

    return status, prob, f"{latency}ms"


# Define the Gradio interface
demo = gr.Interface(
    fn=predict_eft,
    inputs=[
        gr.Number(label="Funding Amount (ZAR)", value=1500.0),
        gr.Dropdown(label="Fund Type", choices=[("Instant EFT", 1), ("Standard EFT", 0)], value=1),
        gr.Slider(label="Hour of Day", minimum=0, maximum=23, step=1, value=14),
    ],
    outputs=[
        gr.Textbox(label="Prediction Status"),
        gr.Number(label="Confidence Score"),
        gr.Textbox(label="Processing Latency"),
    ],
    title="Instant EFT Predictor",
    description="Interactive interface for testing VodaPay Instant EFT success predictions.",
    flagging_mode="never",
)


def launch_interface():
    """Launch the Gradio interface with a public shareable link."""
    demo.launch(theme="soft", share=True)


if __name__ == "__main__":
    launch_interface()
