"""************************************************************
* @Author: Zeng Shengbo shengbo.zeng@ailingues.com
* @Date: 2025-06-15 12:05:19
* @LastEditors: Zeng Shengbo shengbo.zeng@ailingues.com
* @LastEditTime: 2025-06-15 13:42:46
* @FilePath: \\kokoro\\demo\\components\\webcam.py
* @Description:
**********************************************************"""
import json
from pathlib import Path
import time

import cv2
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastrtc import Stream, get_twilio_turn_credentials
from gradio.utils import get_space
from huggingface_hub import hf_hub_download
import numpy as np
from pydantic import BaseModel, Field
from inference import YOLOv10
# try:
#     from demo.object_detection.inference import YOLOv10
# except (ImportError, ModuleNotFoundError):
#     from inference import YOLOv10


cur_dir = Path(__file__).parent

model_file = hf_hub_download(
    repo_id="onnx-community/yolov10n", filename="onnx/model.onnx"
)

model = YOLOv10(model_file)
model.session.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
print("model backend:", model.session.get_providers())

_last_frame = None
_last_infer_time = 0  # 全局变量记录上次推理时间
infer_interval = 0.02  # 最短间隔：单位秒（例如 0.5s = 2FPS）
_frame_diff_threshold = 8  # 越小越敏感（建议10~20）

def detection(image, conf_threshold=0.3):
    global _last_infer_time, _last_frame
    now = time.time()
    if now - _last_infer_time < infer_interval:
        return image

    if _last_frame is not None:
        # 图像差异（平均灰度差）
        diff = np.abs(image.astype(np.int16) - _last_frame.astype(np.int16))
        mean_diff = diff.mean()
        if mean_diff < _frame_diff_threshold:
            return image  # 图像差异太小，跳过推理

    _last_infer_time = now
    _last_frame = image.copy()
    
    image = cv2.resize(image, (image.shape[1], image.shape[0]))
    print("conf_threshold", conf_threshold)
    new_image = model.detect_objects(image, conf_threshold)
    return new_image


stream = Stream(
    handler=detection,
    modality="video",
    mode="send-receive",
    additional_inputs=[gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3)],
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=2 if get_space() else None,
)

app = FastAPI()

stream.mount(app)


@app.get("/")
async def _():
    rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = open(cur_dir / "index.html").read()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


class InputData(BaseModel):
    webrtc_id: str
    conf_threshold: float = Field(ge=0, le=1)


@app.post("/input_hook")
async def _(data: InputData):
    stream.set_input(data.webrtc_id, data.conf_threshold)


if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)