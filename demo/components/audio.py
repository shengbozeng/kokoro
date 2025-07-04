''' ************************************************************ 
 * @Author: Zeng Shengbo shengbo.zeng@ailingues.com
 * @Date: 2025-06-15 11:40:41
 * @LastEditors: Zeng Shengbo shengbo.zeng@ailingues.com
 * @LastEditTime: 2025-06-15 15:23:05
 * @FilePath: \\kokoro\\demo\\components\\audio.py
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 ********************************************************** '''
import os
import random
import spaces
import numpy as np
import gradio as gr
import torch
from kokoro import KModel, KPipeline
CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_VOICE ='af_heart'

print(f'Loading text to speech model... ... ',end='')
models = {gpu: KModel(
    repo_id='hexgrad/kokoro-82M',
    config="/models/hexgrad/kokoro-82M/config.json",
    model="/models/hexgrad/kokoro-82M/kokoro-v1_0.pth").to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
print(f'Done!')
print(f'Loading Pipeline... ... ',end='')
pipelines = {lang_code: KPipeline(lang_code=lang_code,repo_id='hexgrad/kokoro-82M', model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

print(f'Done!')


AUDIO_CHOICES = {
'🇺🇸 🚺 Heart ❤️': 'af_heart',
'🇺🇸 🚺 Bella 🔥': 'af_bella',
'🇺🇸 🚺 Nicole 🎧': 'af_nicole',
'🇺🇸 🚺 Aoede': 'af_aoede',
'🇺🇸 🚺 Kore': 'af_kore',
'🇺🇸 🚺 Sarah': 'af_sarah',
'🇺🇸 🚺 Nova': 'af_nova',
'🇺🇸 🚺 Sky': 'af_sky',
# '🇺🇸 🚺 Alloy': 'af_alloy',
# '🇺🇸 🚺 Jessica': 'af_jessica',
# '🇺🇸 🚺 River': 'af_river',
# '🇺🇸 🚹 Michael': 'am_michael',
# '🇺🇸 🚹 Fenrir': 'am_fenrir',
# '🇺🇸 🚹 Puck': 'am_puck',
# '🇺🇸 🚹 Echo': 'am_echo',
# '🇺🇸 🚹 Eric': 'am_eric',
# '🇺🇸 🚹 Liam': 'am_liam',
# '🇺🇸 🚹 Onyx': 'am_onyx',
# '🇺🇸 🚹 Santa': 'am_santa',
# '🇺🇸 🚹 Adam': 'am_adam',
# '🇬🇧 🚺 Emma': 'bf_emma',
# '🇬🇧 🚺 Isabella': 'bf_isabella',
# '🇬🇧 🚺 Alice': 'bf_alice',
# '🇬🇧 🚺 Lily': 'bf_lily',
# '🇬🇧 🚹 George': 'bm_george',
# '🇬🇧 🚹 Fable': 'bm_fable',
'🇬🇧 🚹 Lewis': 'bm_lewis',
'🇬🇧 🚹 Daniel': 'bm_daniel',
}
for v in AUDIO_CHOICES.values():
    print(f'Load voice {v}... ... ',end='')
    pipelines[v[0]].load_voice(v)
    print(f'Done!')

TOKEN_NOTE = '''
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`

💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`

⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`

⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
'''

def convert_float32_to_int16(audio_float):
    return (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)

@spaces.GPU(duration=30)
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def generate_first(text, voice=DEFAULT_VOICE, speed=1, use_gpu=CUDA_AVAILABLE):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU. To avoid this error, change Hardware to CPU.')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
         # ✅ 手动转 int16
        int16_audio = convert_float32_to_int16(audio.numpy())
        return (24000, int16_audio), ps
    return None, ''


# Arena API
def predict(text, voice=DEFAULT_VOICE, speed=1):
    return generate_first(text, voice, speed, use_gpu=False)[0]

def tokenize_first(text, voice=DEFAULT_VOICE):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ''

def generate_all(text, voice=DEFAULT_VOICE, speed=1, use_gpu=CUDA_AVAILABLE):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    first = True
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Switching to CPU')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        # ✅ 手动转 int16
        int16_audio = convert_float32_to_int16(audio.numpy())
        yield 24000, int16_audio
        
        if first:
            first = False
            yield 24000, convert_float32_to_int16(torch.zeros(1).numpy())

with open(os.path.join(os.getcwd(),'demo','en.txt'), 'r') as r:
    random_quotes = [line.strip() for line in r]

def get_random_quote():
    return random.choice(random_quotes)

def get_gatsby():
    with open('gatsby5k.md', 'r', encoding='utf-8', errors='ignore') as r:
        s=r.read()
        return s.strip()

def get_frankenstein():
    with open('frankenstein5k.md', 'r', encoding='utf-8', errors='ignore') as r:
        s=r.read()
        return s.strip()

