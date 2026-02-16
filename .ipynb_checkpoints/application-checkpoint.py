import gradio as gr
import pickle
import pandas as pd

# Load model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_usage(
    screen_time_hours,
    video_hours,
    social_hours,
    online_classes_hours,
    downloads,
    device_type,
    internet_type
):
    input_data = pd.DataFrame([{
        "screen_time_hours": screen_time_hours,
        "video_hours": video_hours,
        "social_hours": social_hours,
        "online_classes_hours": online_classes_hours,
        "downloads": downloads,
        "device_type": device_type,
        "internet_type": internet_type
    }])

    prediction = model.predict(input_data)[0]
    prediction_mb = float(prediction)
    prediction_gb = prediction_mb / 1024

    return (
        f"{prediction_mb:.2f} MB per day",
        f"{prediction_gb:.2f} GB per day"
    )

demo = gr.Interface(
    fn=predict_usage,
    inputs=[
        gr.Slider(1, 12, value=5, step=0.1, label="Screen Time (hours/day)"),
        gr.Slider(0, 6, value=2, step=0.1, label="Video Streaming (hours/day)"),
        gr.Slider(0, 5, value=1, step=0.1, label="Social Media (hours/day)"),
        gr.Slider(0, 4, value=1, step=0.1, label="Online Classes (hours/day)"),
        gr.Number(value=1, label="Number of Downloads"),
        gr.Dropdown(["phone", "laptop"], label="Device Type"),
        gr.Dropdown(["4G", "5G", "WiFi"], label="Internet Type")
    ],
    outputs=[
        gr.Text(label="Prediction (MB)"),
        gr.Text(label="Prediction (GB)")
    ],
    title="📊 Daily Internet Usage Predictor",
    description="Estimate how much mobile data you use per day (Data Science Project)"
)

demo.launch(share=True)
