import streamlit as st
import pickle
import pandas as pd

# ----------------------------
# Load model
# ----------------------------
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Prediction function
# ----------------------------
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

    return prediction_mb, prediction_gb

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Daily Internet Usage Predictor",
    page_icon="💻",
    layout="centered"  # center content instead of wide
)

# Custom CSS for tablet-style container
st.markdown(
    """
    <style>
    .tablet-container {
        background-color: #fdfdfd; /* soft white background */
        max-width: 700px;           /* tablet width */
        margin: 30px auto;          /* center container with vertical space */
        padding: 30px;              /* internal padding */
        border-radius: 15px;        /* rounded edges */
        box-shadow: 0 10px 25px rgba(0,0,0,0.1); /* subtle shadow */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='tablet-container'>", unsafe_allow_html=True)

st.title("📊 Daily Internet Usage Predictor")
st.write("Estimate how much mobile data you use per day (Data Science Project)")

# ----------------------------
# Inputs in a single column
# ----------------------------
screen_time = st.slider("Screen Time (hours/day)", 1.0, 12.0, 5.0, 0.1)
video_hours = st.slider("Video Streaming (hours/day)", 0.0, 6.0, 2.0, 0.1)
social_hours = st.slider("Social Media (hours/day)", 0.0, 5.0, 1.0, 0.1)
online_classes = st.slider("Online Classes (hours/day)", 0.0, 4.0, 1.0, 0.1)
downloads = st.number_input("Number of Downloads", min_value=0, value=1)
device_type = st.selectbox("Device Type", ["phone", "laptop"])
internet_type = st.selectbox("Internet Type", ["4G", "5G", "WiFi"])

st.markdown("---")  # separator

# ----------------------------
# Predict button and results
# ----------------------------
if st.button("Predict Usage"):
    mb, gb = predict_usage(
        screen_time, video_hours, social_hours,
        online_classes, downloads, device_type, internet_type
    )
    
    st.markdown(
        f"""
        <div style='background-color:#E8F5E9; padding:20px; border-radius:10px;'>
            <h3 style='color:#388E3C; margin:0;'> ✅ Your Daily Usage Prediction</h3>
            <p style='font-size:20px; margin:5px 0;'> <strong>{mb:.2f} MB</strong> per day</p>
            <p style='font-size:20px; margin:5px 0;'> <strong>{gb:.2f} GB</strong> per day</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)
