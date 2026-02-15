import streamlit as st
import pickle
import pandas as pd

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Daily Internet Usage Predictor",
    page_icon="💻",
    layout="wide"
)

# ----------------------------
# Load model
# ----------------------------
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Soft UI styling
# ----------------------------
st.markdown("""
<style>
body {
    background-color: dark grey;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

.stButton>button {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("<h1 style='color:#4facfe;'>📊 Daily Internet Usage Predictor</h1>", unsafe_allow_html=True)
st.caption("Estimate how much mobile data you use per day")

st.info("This model was trained using university student data.")

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧾 Your Daily Activity")

    screen_time_hours = st.slider("Screen Time (hours/day)", 1.0, 12.0, 5.0, 0.1)
    video_hours = st.slider("Video Streaming (hours/day)", 0.0, 6.0, 2.0, 0.1)
    social_hours = st.slider("Social Media (hours/day)", 0.0, 5.0, 1.0, 0.1)
    online_classes_hours = st.slider("Online Classes (hours/day)", 0.0, 4.0, 1.0, 0.1)
    downloads = st.number_input("Number of Downloads", min_value=0, max_value=20, value=1)

    device_type = st.selectbox("Device Type", ["phone", "laptop"])
    internet_type = st.selectbox("Internet Type", ["4G", "5G", "WiFi"])

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("###  Prediction Result")

    if st.button("Predict My Data Usage"):
        input_data = pd.DataFrame([{
            "screen_time_hours": screen_time_hours,
            "video_hours": video_hours,
            "social_hours": social_hours,
            "online_classes_hours": online_classes_hours,
            "downloads": downloads,
            "device_type": device_type,
            "internet_type": internet_type
        }])

        prediction = model.predict(input_data)
        prediction_mb = prediction[0]
        prediction_gb = prediction_mb / 1024

        st.success(f"**{prediction_mb:.2f} MB per day**")
        st.metric("In Gigabytes", f"{prediction_gb:.2f} GB")

    else:
        st.write("Click the button to see your estimated usage.")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.caption("Data Science Project")
