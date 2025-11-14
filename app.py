import streamlit as st
import cv2
import torch
from torchvision import transforms
import tempfile
import os
from src.model import build_model
from src.predict import predict_video
import base64

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Action Recognition App", layout="wide")
st.title("🎬 Human Action Recognition (CNN + LSTM)")
st.write("Upload a video or use your webcam to detect human actions in real time!")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("models/action_recognition_model.pth", map_location="cpu")
    num_classes = len(checkpoint["class_to_idx"])
    model = build_model(num_classes)

    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing:
        st.warning(f"⚠️ Missing keys: {missing}")
    if unexpected:
        st.warning(f"⚠️ Unexpected keys: {unexpected}")

    model.idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
    model.eval()
    return model

model = load_model()

# ---------------- Transform ----------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.CenterCrop((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ---------------- Video Upload Section ----------------
st.header("📁 Upload a Video for Action Recognition")

uploaded_file = st.file_uploader("Upload a video", type=["mp4","avi"])

if uploaded_file is not None:
    # Read bytes once
    video_bytes = uploaded_file.read()

    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_bytes)
    tfile.close()
    video_path = tfile.name

    # Display video in 200x200 px and centered
    video_b64 = base64.b64encode(video_bytes).decode()
    st.markdown(f"""
    <div style="display: flex; justify-content: center;">
        <video width="500" height="250" controls>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
    </div>
""", unsafe_allow_html=True)

    st.write("⏳ Analyzing video...")

    # Predict top 2
    results = predict_video(video_path, model, transform, num_frames=16, topk=2)

    # Display top 2 predictions
    st.header("🎯 Top 2 Predicted Actions")
    for i, (cls, prob) in enumerate(results, 1):
        if i == 1:
            st.success(f"{i}. {cls} — {prob*100:.2f}%")
        else:
            st.write(f"{i}. {cls} — {prob*100:.2f}%")

    # Clean up temp file
    try:
        os.remove(video_path)
    except PermissionError:
        st.warning("Temporary video could not be deleted immediately on Windows.")

        
# ---------------- Live Webcam Section ----------------
st.header("📷 Live Webcam Action Recognition")

start_camera = st.button("Start Camera")
stop_camera = st.button("Stop Camera")

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

if start_camera:
    st.session_state.camera_active = True
if stop_camera:
    st.session_state.camera_active = False

stframe = st.empty()
buffer_size = 16
frame_buffer = []

if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        st.warning("Press STOP button to end webcam.")
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break

            frame_buffer.append(frame)
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)

            if len(frame_buffer) == buffer_size:
                results = predict_video(
                    frame_buffer,
                    model,
                    transform,
                    num_frames=buffer_size,
                    topk=3,
                    from_camera=True
                )
                top_class, top_prob = results[0]

                rgb_frame = cv2.cvtColor(frame_buffer[-1], cv2.COLOR_BGR2RGB)
                cv2.putText(
                    rgb_frame,
                    f"{top_class} ({top_prob*100:.1f}%)",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )
                stframe.image(rgb_frame, channels="RGB")

                st.markdown("**Top 3 Predictions:**")
                for i, (cls, prob) in enumerate(results, 1):
                    if i == 1:
                        st.markdown(f"<span style='color:green'>{i}. {cls} — {prob*100:.2f}%</span>", unsafe_allow_html=True)
                    else:
                        st.write(f"{i}. {cls} — {prob*100:.2f}%")

        cap.release()
        st.success("Webcam stopped successfully.")
