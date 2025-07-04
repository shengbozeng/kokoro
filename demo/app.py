import spaces
from components.audio import (
    AUDIO_CHOICES,
    DEFAULT_VOICE,
    TOKEN_NOTE,
    generate_all,
    generate_first,
    get_frankenstein,
    get_gatsby,
    get_random_quote,
    predict,
    tokenize_first,
)
import gradio as gr
import os
import torch
import colorsys
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import uuid

SUBSAMPLE = 2

# os.environ["http_proxy"] = "http://127.0.0.1:20171"
# os.environ["https_proxy"] = "http://127.0.0.1:20171"
#
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
os.chdir(script_dir)  # 更改工作目录到脚本所在目录

css = """.my-group {max-width: 600px !important; max-height: 600px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}"""
repo_id = ("hexgrad/kokoro-82M",)
config_json = ("/models/hexgrad/kokoro-82M/config.json",)
model_pth = "/models/hexgrad/kokoro-82M/kokoro-v1_0.pth"

CUDA_AVAILABLE = torch.cuda.is_available()


print(f"Loading image process model... ... ", end="")
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd").to("cuda")
print(f"Done!")


def get_color(label):
    # Simple hash function to generate consistent colors for each label
    hash_value = hash(label)
    hue = (hash_value % 100) / 100.0
    saturation = 0.7
    value = 0.9
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(x * 255) for x in rgb)


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

            # Draw bounding box
            draw.rectangle(box, outline=color, width=3)  # type: ignore

            # Prepare text
            text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Draw text background
            draw.rectangle(
                [box[0], box[1] - text_height - 4, box[0] + text_width, box[1]],  # type: ignore
                fill=color,  # type: ignore
            )

            # Draw text
            draw.text((box[0], box[1] - text_height - 4), text, fill="white", font=font)

    return image


@spaces.GPU
def stream_object_detection_from_video(video, conf_threshold):
    cap = cv2.VideoCapture(video)

    # This means we will output mp4 videos
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    desired_fps = fps // SUBSAMPLE
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

    iterating, frame = cap.read()

    n_frames = 0

    # Use UUID to create a unique video file
    output_video_name = f"output_{uuid.uuid4()}.mp4"

    # Output Video
    output_video = cv2.VideoWriter(output_video_name, video_codec, desired_fps, (width, height))  # type: ignore
    batch = []

    while iterating:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if n_frames % SUBSAMPLE == 0:
            batch.append(frame)
        if len(batch) == 2 * desired_fps:
            inputs = image_processor(images=batch, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model(**inputs)

            boxes = image_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([(height, width)] * len(batch)),
                threshold=conf_threshold,
            )

            for i, (array, box) in enumerate(zip(batch, boxes)):
                pil_image = draw_bounding_boxes(
                    Image.fromarray(array), box, model, conf_threshold
                )
                frame = np.array(pil_image)
                # Convert RGB to BGR
                frame = frame[:, :, ::-1].copy()
                output_video.write(frame)

            batch = []
            output_video.release()
            yield output_video_name
            output_video_name = f"output_{uuid.uuid4()}.mp4"
            output_video = cv2.VideoWriter(output_video_name, video_codec, desired_fps, (width, height))  # type: ignore

        iterating, frame = cap.read()
        n_frames += 1


recognition_active = False  # 全局状态变量

@spaces.GPU
def detect_frame_controlled_for_display(frame, conf_threshold):
    print("Frame received:", frame.shape)
    """处理每帧：返回识别后图像（用于 Image 显示）"""
    if not recognition_active:
        return frame  # 返回原图像（RGB）

    resized = cv2.resize(frame, (640, 640))
    inputs = image_processor(images=[resized], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    boxes = image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[640, 640]]),
        threshold=conf_threshold,
    )

    # 绘制框
    pil_image = draw_bounding_boxes(Image.fromarray(resized), boxes[0], model, conf_threshold)
    print("Returning frame shape:", np.array(pil_image).shape)
    return np.array(pil_image)  # RGB ndarray


def detect_frame_controlled(frame, conf_threshold):
    if not recognition_active:
        return frame  # 不处理，原样返回

    resized = cv2.resize(frame, (640, 640))
    result = model.detect_objects(resized, conf_threshold)
    return cv2.resize(result, (320, 320))[:, :, ::-1]


# 假设你有 YOLOv10 + detect_objects 结构
@spaces.GPU
def detect_frame(frame: np.ndarray, conf_threshold: float = 0.3):
    resized = cv2.resize(frame, (640, 640))
    result = model.detect_objects(resized, conf_threshold)
    # 压缩返回帧（RGB）
    display_frame = cv2.resize(result, (320, 320))
    return display_frame[:, :, ::-1]  # BGR → RGB


def toggle_recognition():
    global recognition_active
    recognition_active = not recognition_active
    return "Stop Detection" if recognition_active else "Start Detection"

@spaces.GPU
def stream_object_detection_from_camera(conf_threshold=0.3):
    cap = cv2.VideoCapture(0)  # 使用默认摄像头

    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 15  # 摄像头理想帧率
    width = 320
    height = 240

    output_video_name = f"camera_output_{uuid.uuid4()}.mp4"
    output_video = cv2.VideoWriter(
        output_video_name, video_codec, fps // SUBSAMPLE, (width, height)
    )

    n_frames = 0
    batch = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if n_frames % SUBSAMPLE == 0:
            batch.append(frame)

        if len(batch) >= fps:  # 每次处理1秒的视频
            inputs = image_processor(images=batch, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model(**inputs)

            boxes = image_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([(height, width)] * len(batch)),
                threshold=conf_threshold,
            )

            for array, box in zip(batch, boxes):
                pil_image = draw_bounding_boxes(
                    Image.fromarray(array), box, model, conf_threshold
                )
                output_frame = np.array(pil_image)[:, :, ::-1]  # RGB -> BGR
                output_video.write(output_frame)

            yield output_video_name
            output_video_name = f"camera_output_{uuid.uuid4()}.mp4"
            output_video = cv2.VideoWriter(
                output_video_name, video_codec, fps // SUBSAMPLE, (width, height)
            )
            batch = []

        n_frames += 1

    cap.release()
    output_video.release()


with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(
        label="Output Audio", interactive=False, streaming=False, autoplay=True
    )
    generate_btn = gr.Button("Generate", variant="primary")
    with gr.Accordion("Output Tokens", open=True):
        out_ps = gr.Textbox(
            interactive=False,
            show_label=False,
            info="Tokens used to generate the audio, up to 510 context length.",
        )
        tokenize_btn = gr.Button("Tokenize", variant="secondary")
        gr.Markdown(TOKEN_NOTE)
        predict_btn = gr.Button("Predict", variant="secondary", visible=False)

STREAM_NOTE = [
    "⚠️ There is an unknown Gradio bug that might yield no audio the first time you click `Stream`."
]
STREAM_NOTE = "\n\n".join(STREAM_NOTE)

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(
        label="Output Audio Stream", interactive=False, streaming=True, autoplay=True
    )
    with gr.Row():
        stream_btn = gr.Button("Stream", variant="primary")
        stop_btn = gr.Button("Stop", variant="stop")
    with gr.Accordion("Note", open=True):
        gr.Markdown(STREAM_NOTE)
        gr.DuplicateButton()

with gr.Blocks(css=css) as stream_video:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Video Object Detection with <a href='https://huggingface.co/PekingU/rtdetr_r101vd_coco_o365' target='_blank'>RT-DETR</a>
    </h1>
    """
    )
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Video Source")
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.30,
            )
        with gr.Column():
            output_video = gr.Video(
                label="Processed Video", streaming=True, autoplay=True
            )

    video.upload(
        fn=stream_object_detection_from_video,
        inputs=[video, conf_threshold],
        outputs=[output_video],
    )
from gradio_webrtc import WebRTC

with gr.Blocks(css=css) as live_webcam_demo:
    gr.HTML("<h1 style='text-align: center'>Live Webcam Object Detection</h1>")

    with gr.Row():
        conf_slider = gr.Slider(
            label="Confidence Threshold",
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=0.3,
        )
        toggle_btn = gr.Button("Start Detection")
    
    with gr.Row():    
        with gr.Column():
            # 实时视频输入与结果显示
            stream_view = WebRTC(
                label="Live Detection",
                mode="sendrecv",  # 必须设置
                modality="video",  # 启用视频处理
                # streaming=True           # 必须打开
            )
            
        with gr.Column():
            image_output = gr.Image(label="🎯 Detection Result", interactive=False)
            
    # 将处理结果输出到右侧 Image 组件
    stream_view.stream(
        fn=detect_frame_controlled_for_display,
        inputs=[stream_view, conf_slider],
        outputs=[image_output]
    )
    # 实时帧处理绑定
    # stream_view.stream(
    #     fn=detect_frame_controlled, inputs=[stream_view, conf_slider], outputs=stream_view
    # )
    # 控制识别状态的按钮
    toggle_btn.click(fn=toggle_recognition, inputs=[], outputs=[toggle_btn])


API_OPEN = True
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="Input Text", info=f"Arbitrarily many characters supported"
            )
            with gr.Row():
                voice = gr.Dropdown(
                    list(AUDIO_CHOICES.items()),
                    value=DEFAULT_VOICE,
                    label="Voice",
                    info="Quality and availability vary by language",
                )
                use_gpu = gr.Dropdown(
                    [("ZeroGPU 🚀", True), ("CPU 🐌", False)],
                    value=CUDA_AVAILABLE,
                    label="Hardware",
                    info="GPU is usually faster, but has a usage quota",
                    interactive=CUDA_AVAILABLE,
                )
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label="Speed")
            random_btn = gr.Button("🎲 Random Quote 💬", variant="secondary")
            with gr.Row():
                gatsby_btn = gr.Button("🥂 Gatsby 📕", variant="secondary")
                frankenstein_btn = gr.Button("💀 Frankenstein 📗", variant="secondary")
        with gr.Column():
            gr.TabbedInterface(
                [generate_tab, stream_tab, stream_video, live_webcam_demo],
                ["Generate", "Stream", "Stream_Video", "live_webcam_demo"],
            )
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text])
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text])
    generate_btn.click(
        fn=generate_first,
        inputs=[text, voice, speed, use_gpu],
        outputs=[out_audio, out_ps],
    )
    tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps])
    stream_event = stream_btn.click(
        fn=generate_all, inputs=[text, voice, speed, use_gpu], outputs=[out_stream]
    )
    stop_btn.click(fn=None, cancels=stream_event)
    predict_btn.click(fn=predict, inputs=[text, voice, speed], outputs=[out_audio])


if __name__ == "__main__":
    app.queue(api_open=API_OPEN).launch(
        server_name="0.0.0.0",
        server_port=40001,
        show_api=API_OPEN,
        share=True,
        debug=True,
    )
