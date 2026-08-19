import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import joblib
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

st.set_page_config(
    page_title="Chest X-ray Triage System",
    layout="wide"
)

TARGET_LABELS = [
    "Lung Opacity",
    "Pleural Effusion",
    "Edema",
    "Cardiomegaly",
    "Pneumothorax",
]

FINAL_THRESHOLDS = {
    "Lung Opacity": 0.334,
    "Pleural Effusion": 0.347,
    "Edema": 0.350,
    "Cardiomegaly": 0.332,
    "Pneumothorax": 0.247,
}

CONF_THRESHOLD = 0.90

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "v2_masked_finetune_db4_t3_norm5_20260305_015938.pt")
RELIABILITY_PATH = os.path.join(BASE_DIR, "model", "maxprob_reliability_model.pkl")

class ChexpertDenseNet121(nn.Module):
    def __init__(self, num_labels=5):
        super().__init__()
        self.backbone = models.densenet121(weights=None)
        self.backbone.classifier = nn.Linear(
            self.backbone.classifier.in_features, num_labels
        )

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_models():
    device = torch.device("cpu")
    model = ChexpertDenseNet121(num_labels=len(TARGET_LABELS))
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    reliability_model = joblib.load(RELIABILITY_PATH)
    return model, reliability_model

model, reliability_model = load_models()

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def run_inference(pil_image):
    img_tensor = inference_transform(pil_image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    prob_dict = {TARGET_LABELS[i]: float(probs[i]) for i in range(len(TARGET_LABELS))}

    findings = [
        (label, prob) for label, prob in prob_dict.items()
        if prob >= FINAL_THRESHOLDS[label]
    ]
    findings = sorted(findings, key=lambda x: x[1], reverse=True)
    case_result = "Abnormality Detected" if findings else "No Abnormality Detected"

    max_prob = max(prob_dict.values())
    X = np.array([[max_prob]])
    error_prob = reliability_model.predict_proba(X)[0, 1]
    confidence_score = 1 - error_prob

    if confidence_score >= CONF_THRESHOLD:
        decision = "Accepted"
        color = "green"
    else:
        decision = "Refer for Radiologist Review"
        color = "orange"

    return {
        "prob_dict": prob_dict,
        "findings": findings,
        "case_result": case_result,
        "confidence_score": confidence_score,
        "decision": decision,
        "color": color,
        "max_prob": max_prob,
    }


def generate_gradcam(pil_image, target_label_idx):
    img_tensor = inference_transform(pil_image.convert("RGB")).unsqueeze(0)
    target_layer = [model.backbone.features.denseblock4]
    cam = GradCAM(model=model, target_layers=target_layer)
    targets = [ClassifierOutputTarget(target_label_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    
    img_resized = pil_image.convert("RGB").resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization

st.title("Chest X-ray Triage System")
st.caption("AI-assisted decision support for chest X-ray screening")

st.divider()


uploaded_file = st.file_uploader(
    "Upload chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a frontal chest X-ray in JPG or PNG format"
)


if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(pil_image, use_container_width=True)

    with col2:
        st.subheader("Analysis")

        with st.spinner("Running inference..."):
            result = run_inference(pil_image)

        
        if result["case_result"] == "Abnormality Detected":
            st.error(f"⚠️ {result['case_result']}")
        else:
            st.success(f"✓ {result['case_result']}")

        st.divider()

        
        st.metric(
            label="Confidence Score",
            value=f"{result['confidence_score']*100:.1f}%"
        )
        st.progress(float(result["confidence_score"]))

        st.divider()

       
        if result["color"] == "green":
            st.success(f"✓ {result['decision']}")
        else:
            st.warning(f"⚠ {result['decision']}")

    st.divider()

    
    with st.expander("Find out more ", expanded=False):

        exp_col1, exp_col2 = st.columns([1, 1])

        with exp_col1:
            st.subheader("GradCAM Heatmap")
            st.caption("Regions the model focused on when making its prediction\n"
             "Note: This heatmap highlights influential regions but should not be interpreted as a definitive explanation."
            )

            if result["findings"]:
                top_label = result["findings"][0][0]
                top_label_idx = TARGET_LABELS.index(top_label)
                st.caption(f"Visualising attention for: **{top_label}**")
            else:
                top_label_idx = int(np.argmax([result["prob_dict"][l] for l in TARGET_LABELS]))
                top_label = TARGET_LABELS[top_label_idx]
                st.caption(f"Visualising attention for: **{top_label}** (highest probability label)")

            with st.spinner("Generating heatmap..."):
                heatmap = generate_gradcam(pil_image, top_label_idx)
            st.image(heatmap, use_container_width=True)

        with exp_col2:
            st.subheader("Model Signals")
            st.caption("Probability scores for each pathology")

            sorted_probs = sorted(
                result["prob_dict"].items(),
                key=lambda x: x[1], reverse=True
            )

            for label, prob in sorted_probs:
                thresh = FINAL_THRESHOLDS[label]
                above = prob >= thresh
                bar_label = f"{'✓ ' if above else ''}{label}: {prob*100:.1f}%"
                st.progress(float(prob), text=bar_label)

            st.divider()
            st.caption(
                "**Research prototype only.** This system is not validated "
                "for clinical use."
            )

st.divider()
st.caption(
    "Not for clinical use · Final Year Project · "
)
