import cv2
import torch
import torch.nn.functional as F
import numpy as np
import imageio.v3 as iio

def predict_video(video_input, model, transform, num_frames=16, device='cpu', topk=3, from_camera=False):
    """
    Predict action class from video or webcam frames using CNN+LSTM model.
    
    Parameters:
        video_input: str (path) or list of frames (for webcam)
        model: trained CNN+LSTM model
        transform: torchvision transforms
        num_frames: number of frames to sample for prediction
        topk: number of top predictions
        from_camera: True if input is live frames from webcam
    Returns:
        list of tuples: [(class_name, probability), ...]
    """

    # -------- Webcam mode (list of frames) --------
    if from_camera:
        frames = video_input  # list of frames

    else:
        # -------- Read all frames from video path --------
        def safe_read_frames_opencv(path):
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            for _ in range(total):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            return frames

        frames = safe_read_frames_opencv(video_input)

        # fallback to imageio
        if len(frames) == 0:
            try:
                frames = [f for f in iio.imiter(video_input)]
            except:
                raise ValueError(f"Cannot read video: {video_input}")

    if len(frames) == 0:
        raise ValueError("No frames extracted!")

    # -------- Sample num_frames uniformly --------
    total = len(frames)
    idxs = np.linspace(0, total - 1, num_frames).astype(int)
    sampled = [frames[i] for i in idxs]

    # -------- Apply transforms --------
    processed = [transform(f) for f in sampled]
    video_tensor = torch.stack(processed).unsqueeze(0).to(device)  # (1,F,C,H,W)

    # -------- Forward pass --------
    model.eval()
    with torch.no_grad():
        logits = model(video_tensor)
        probs = F.softmax(logits, dim=1)
        top_prob, top_idx = probs.topk(topk, dim=1)

    top_prob = top_prob.cpu().numpy()[0]
    top_idx = top_idx.cpu().numpy()[0]

    # -------- Map class indices to names --------
    idx_to_class = getattr(model, "idx_to_class", None)
    if idx_to_class is None:
        raise ValueError("Model does not have idx_to_class mapping!")

    results = [(idx_to_class[int(i)], float(p)) for i, p in zip(top_idx, top_prob)]
    return results
