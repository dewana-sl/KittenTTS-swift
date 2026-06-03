#!/usr/bin/env python3
import json
import os
import statistics
import sys
import time

import numpy as np
import soundfile as sf


TEXT = os.environ.get("KITTENTTS_BENCH_TEXT", "The quick brown fox jumps over the lazy dog.")
IPA = os.environ.get("KITTENTTS_BENCH_IPA", "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ.")
RUNS = int(os.environ.get("KITTENTTS_BENCH_RUNS", "3"))
WARMUP = int(os.environ.get("KITTENTTS_BENCH_WARMUP", "1"))
SPEED = float(os.environ.get("KITTENTTS_BENCH_SPEED", "1.0"))
CACHE_DIR = os.environ.get("KITTENTTS_BENCH_CACHE", "/tmp/kittentts-python-bench")
ENGINE_MODULE_PATH = os.environ.get(
    "KITTENTTS_NATIVE_ENGINE_PATH",
    "/Volumes/DewansSSD/dev/project/cpp-convnet-pvt/build-bench-py",
)
OUTPUT_DIR = os.environ.get("KITTENTTS_BENCH_OUTPUT_DIR")


class FixedPhonemizer:
    def phonemize(self, texts):
        return [IPA for _ in texts]


def patch_phonemizer(model):
    model.model.phonemizer = FixedPhonemizer()
    model.model.speed_priors = {}


def benchmark(model, engine):
    for _ in range(WARMUP):
        model.generate(TEXT, voice="Bella", speed=SPEED, clean_text=True)

    timings = []
    audio = None
    for _ in range(RUNS):
        start = time.perf_counter()
        audio = model.generate(TEXT, voice="Bella", speed=SPEED, clean_text=True)
        timings.append(time.perf_counter() - start)

    audio = np.asarray(audio).squeeze()
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sf.write(os.path.join(OUTPUT_DIR, f"python-{engine}.wav"), audio, 24_000)

    duration = float(audio.size) / 24_000.0
    mean = statistics.fmean(timings)
    std = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    return {
        "platform": "python",
        "engine": engine,
        "model": "nano/fp32_15m",
        "voice": "expr-voice-2-f",
        "text": TEXT,
        "speed": SPEED,
        "effectiveSpeed": SPEED,
        "runs": RUNS,
        "durationSeconds": duration,
        "generationTimeSecondsMean": mean,
        "generationTimeSecondsMin": min(timings),
        "generationTimeSecondsStd": std,
        "rtfMean": mean / duration,
        "sampleCount": int(audio.size),
    }


def main():
    sys.path.insert(0, "/Volumes/DewansSSD/dev/project/KittenTTS")
    from kittentts import KittenTTS

    onnx = KittenTTS(
        "KittenML/kitten-tts-nano-0.8",
        cache_dir=CACHE_DIR,
        backend="cpu",
        inference_engine="onnx",
    )
    patch_phonemizer(onnx)

    native = KittenTTS(
        "KittenML/kitten-tts-nano-0.8",
        cache_dir=CACHE_DIR,
        backend="cpu",
        inference_engine="native",
        engine_module_path=ENGINE_MODULE_PATH,
        native_variant="fp32_15m",
    )
    patch_phonemizer(native)

    print(json.dumps([benchmark(onnx, "onnx"), benchmark(native, "native")], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
