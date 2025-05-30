import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
import matplotlib.cm as cm
from io import BytesIO
import requests

st.title("Image Classification Application (PyTorch MobileNetV2)")
st.write("Upload an image and let the model classify it")

# โหลดโมเดล MobileNetV2 pretrained
model = models.mobilenet_v2(pretrained=True)
model.eval()

# โหลด labels จาก imagenet
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.splitlines()

# preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # เพิ่มมิติ batch

    # predict
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # top 3 prediction
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    st.write("### Predictions:")
    for i in range(top3_prob.size(0)):
        st.write(f"**{labels[top3_catid[i]]}** ({top3_prob[i].item()*100:.2f}%)")
        st.progress(int(top3_prob[i].item()*100))

    # ------- Grad-CAM -------

    st.write("### Grad-CAM Visualization")

    # ฟังก์ชันช่วยทำ Grad-CAM
    def generate_gradcam(model, input_tensor, class_idx):
        model.zero_grad()
        features = None
        gradients = None

        def save_features(module, input, output):
            nonlocal features
            features = output

        def save_gradients(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]

        # hook บันทึก features และ gradients ที่ layer สุดท้ายก่อน classifier
        target_layer = model.features[-1]  # Conv_1 equivalent
        hook_f = target_layer.register_forward_hook(save_features)
        hook_g = target_layer.register_backward_hook(save_gradients)

        output = model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        hook_f.remove()
        hook_g.remove()

        # global average pooling ของ gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # weight feature maps ด้วย pooled gradients
        for i in range(features.shape[1]):
            features[0, i, :, :] *= pooled_gradients[i]

        heatmap = features[0].detach().cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8  # normalize

        return heatmap

    # สร้าง heatmap Grad-CAM
    pred_class = top3_catid[0].item()
    heatmap = generate_gradcam(model, input_batch, pred_class)

    # ย่อ/ขยาย heatmap ให้เท่ากับขนาดรูปต้นฉบับ
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, resample=Image.BILINEAR)

    # ใช้ colormap ของ matplotlib
    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(np.array(heatmap_img) / 255.0)[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    colored_heatmap_img = Image.fromarray(colored_heatmap)

    # ผสม heatmap กับภาพต้นฉบับ
    blended = Image.blend(image, colored_heatmap_img, alpha=0.4)

    st.image(blended, caption="Grad-CAM Heatmap", use_column_width=True)
