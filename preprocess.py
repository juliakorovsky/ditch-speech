import os
import torch
import gigaam
import resampy
import soundfile as sf
from patches import _patched_put_accent
from ruaccent import RUAccent
from ruaccent.accent_model import AccentModel


VAD_THRES = 0.50
MIN_SPEECH_DUR_MS = 500
MIN_SIL_DUR_MS = 50
SPEECH_PAD_MS = 0

SAMPLE_RATE = 24000
ASR_SAMPLE_RATE = 16000
DEVICE = 'cpu'

AccentModel.put_accent = _patched_put_accent
accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)

gigaam = gigaam.load_model("v3_e2e_ctc")
gigaam.to(DEVICE)

silero, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                               model='silero_vad',
                               force_reload=False)

(get_timestamps, _, _, _, _) = utils


def load_audio(audio_path):
    audio, sr = sf.read(audio_path, dtype='float32')
    audio_asr = resampy.resample(audio, sr, 16000) # yep, hardcoded
    if sr != SAMPLE_RATE:
        audio = resampy.resample(audio, sr, SAMPLE_RATE)
    return audio, audio_asr

def get_transcripts(audio, timestamps, device):
    transcripts = []
    for t in timestamps:
        start, end = t['start']*ASR_SAMPLE_RATE, t['end']*ASR_SAMPLE_RATE
        segment = audio[int(start): int(end)]
        segment = torch.from_numpy(segment).unsqueeze(0)
        length = torch.ones([1]) * segment.shape[-1]
        encoded, encoded_len = gigaam.forward(segment, length)
        text, ws = gigaam._decode(encoded, encoded_len, length, word_timestamps=False)[0]
        transcripts.append(text)
    return transcripts


def cut_to_segments(audio, timestamps, out_path):
    paths = []
    for t in timestamps:
        start, end = t['start'] * SAMPLE_RATE, t['end'] * SAMPLE_RATE
        start, end = int(start), int(end)
        filename = f'{out_path}/{start}_{end}.wav'
        sf.write(filename, audio[start:end], SAMPLE_RATE)
        paths.append(filename)
    return paths

        



def create_dataset(audio_path, out_path):
    audio, asr_audio = load_audio(audio_path)
    silero_audio = torch.from_numpy(asr_audio)
    bits = get_timestamps(silero_audio,
                          silero, threshold=VAD_THRES,
                          sampling_rate=ASR_SAMPLE_RATE,
                          min_speech_duration_ms=MIN_SPEECH_DUR_MS,
                          min_silence_duration_ms=MIN_SIL_DUR_MS,
                          speech_pad_ms=SPEECH_PAD_MS,
                          return_seconds=True) # audio to cut and audio for VAD are in different SR
    timestamps = [{"start": b["start"], "end": b["end"]} for b in bits]
    transcripts = get_transcripts(asr_audio, timestamps, DEVICE)
    meta_path = os.path.join(out_path, 'metadata.csv')
    out_path = os.path.join(out_path, 'segments')
    os.makedirs(out_path, exist_ok=True)
    file_paths = cut_to_segments(audio, timestamps, out_path)
    with open(meta_path, 'w', encoding='utf-8') as out:
        for path, text in zip(file_paths, transcripts):
            text = accentizer.process_all(text)
            line = f'{os.path.abspath(path)}|{text}\n'
            out.write(line)
    

create_dataset('./original.wav', './out')