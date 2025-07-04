import gradio as gr
import spaces
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# ====== 模型加载 ======
print(f"Loading RT-DETR ... ", end="")
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd").to("cuda")
print("Done!")

# ====== 辅助函数 ======
def get_color(label):
    hash_value = hash(label)
    hue = (hash_value % 100) / 100.0
    saturation = 0.7
    value = 0.9
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(x * 255) for x in rgb)
@spaces.GPU
def draw_bounding_boxes(image, results: dict, model, threshold=0.3):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        if score > threshold:
            label = model.config.id2label[label_id.item()]
            box = [round(i, 2) for i in box.tolist()]
            color = get_color(label)
            draw.rectangle(box, outline=color, width=3)
            text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle(
                [box[0], box[1] - text_height - 4, box[0] + text_width, box[1]], fill=color
            )
            draw.text((box[0], box[1] - text_height - 4), text, fill="white", font=font)
    return image

# ====== 检测帧处理函数 ======
recognition_active = False  # 全局状态
@spaces.GPU
def detect_frame_controlled_for_display(frame, conf_threshold):
    global recognition_active
    if not recognition_active:
        return frame  # 显示原始帧
    resized = cv2.resize(frame, (640, 640))
    inputs = image_processor(images=[resized], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    boxes = image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[640, 640]]),
        threshold=conf_threshold,
    )
    pil_image = draw_bounding_boxes(Image.fromarray(resized), boxes[0], model, conf_threshold)
    return np.array(pil_image)  # RGB

def toggle_recognition():
    global recognition_active
    recognition_active = not recognition_active
    return "Stop Detection" if recognition_active else "Start Detection"

# ====== Gradio 界面结构 ======
from gradio_webrtc import WebRTC

with gr.Blocks() as demo:
    gr.HTML("<h2 style='text-align: center'>Live Webcam Object Detection (本地摄像头+实时推理结果)</h2>")
    with gr.Row():
        conf_slider = gr.Slider(
            label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.3,
        )
        toggle_btn = gr.Button("Start Detection")

    with gr.Row():
        # 左侧：本地摄像头预览（永远零延迟）
        with gr.Column():
            gr.HTML(
                """
                <video id="webcamPreview" width="100%" height="320" autoplay playsinline></video>
                <script>
                (async () => {
                    const video = document.getElementById('webcamPreview');
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                        video.srcObject = stream;
                    } catch (e) {
                        video.parentElement.innerHTML = "<b style='color:red;'>无法获取摄像头权限</b>";
                    }
                })();
                </script>
                """
            )
        # 右侧：推理结果
        with gr.Column():
            detection_result = gr.Image(label="🎯 Detection Result", interactive=False)

    # 隐藏的 WebRTC 组件，仅供帧采集推理使用
    stream_view = WebRTC(
        label=None,
        mode="sendrecv",
        modality="video",
        visible=False  # 这个组件隐藏，不显示在页面上
    )

    stream_view.stream(
        fn=detect_frame_controlled_for_display,
        inputs=[stream_view, conf_slider],
        outputs=[detection_result]
    )
    toggle_btn.click(fn=toggle_recognition, inputs=[], outputs=[toggle_btn])

# ====== 启动服务 ======
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=40001, debug=True)
