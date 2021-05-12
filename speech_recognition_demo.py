#!/usr/bin/env python3
#
# Copyright (C) 2019-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on deepspeech_openvino_0.5.py by Feng Yen-Chang at
# https://github.com/openvinotoolkit/open_model_zoo/pull/419, commit 529805d011d9b405f142b2b40f4d202bd403a4f1 on Sep 19, 2019.
#
import wave
import timeit
import argparse

import yaml
import numpy as np
from tqdm import tqdm

from utils.context_timer import Timer
from utils.deep_speech_pipeline import DeepSpeechPipeline, PROFILES

from recode import recode
from audiocut import mtaudiocut

import os
import logging

import argparse
from src.predict import OpenVINOPredictor
from pprint import pprint

audio = "output.wav"

myprofile = " -p mds08x_en"

mymodel = " -m /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/public/mozilla-deepspeech-0.8.2/mozilla-deepspeech-0.8.2/FP32/mozilla-deepspeech-0.8.2.xml"

mylm = " -L /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/public/mozilla-deepspeech-0.8.2/mozilla-deepspeech-0.8.2/deepspeech-0.8.2-models.kenlm"

myaudio = " -i ./output.wav"
    
mydevice = "CPU"

def build_argparser():
    parser = argparse.ArgumentParser(description="Speech recognition demo")
    parser.add_argument('-i', '--input', type=str, metavar="FILENAME", required=True,
                        help="Path to an audio file in WAV PCM 16 kHz mono format")
    parser.add_argument('-d', '--device', default='CPU', type=str,
                        help="Optional. Specify the target device to infer on, for example: CPU, GPU, FPGA, HDDL, MYRIAD or HETERO. "
                             "The sample will look for a suitable IE plugin for this device. (default is CPU)")
    parser.add_argument('-m', '--model', type=str, metavar="FILENAME", required=True,
                        help="Path to an .xml file with a trained model (required)")
    parser.add_argument('-L', '--lm', type=str, metavar="FILENAME",
                        help="path to language model file (optional)")
    parser.add_argument('-p', '--profile', type=str, metavar="NAME", required=True,
                        help="Choose pre/post-processing profile: "
                             "mds06x_en for Mozilla DeepSpeech v0.6.x, "
                             "mds07x_en or mds08x_en for Mozilla DeepSpeech v0.7.x/x0.8.x, "
                             "other: filename of a YAML file (required)")
    parser.add_argument('-b', '--beam-width', type=int, default=500, metavar="N",
                        help="Beam width for beam search in CTC decoder (default 500)")
    parser.add_argument('-c', '--max-candidates', type=int, default=1, metavar="N",
                        help="Show top N (or less) candidates (default 1)")

    parser.add_argument('-l', '--cpu_extension', type=str, metavar="FILENAME",
                        help="Optional. Required for CPU custom layers. "
                             "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                             " kernels implementations.")
    return parser


def get_profile(profile_name):
    if profile_name in PROFILES:
        return PROFILES[profile_name]
    with open(profile_name, 'rt') as f:
        profile = yaml.load(f)
    return profile


def main():
    start_time = timeit.default_timer()
    with Timer() as timer:
        args = build_argparser().parse_args()
        profile = get_profile(args.profile)

        stt = DeepSpeechPipeline(
            model = args.model,
            lm = args.lm,
            beam_width = args.beam_width,
            max_candidates = args.max_candidates,
            profile = profile,
            device = args.device,
            ie_extensions = [(args.device, args.cpu_extension)] if args.device == 'CPU' else [],
        )

        wave_read = wave.open(args.input, 'rb')
        channel_num, sample_width, sampling_rate, pcm_length, compression_type, _ = wave_read.getparams()
        assert sample_width == 2, "Only 16-bit WAV PCM supported"
        assert compression_type == 'NONE', "Only linear PCM WAV files supported"
        assert channel_num == 1, "Only mono WAV PCM supported"
        audio = np.frombuffer(wave_read.readframes(pcm_length * channel_num), dtype=np.int16).reshape((pcm_length, channel_num))
        wave_read.close()
    print("Loading, including network weights, IE initialization, LM, building LM vocabulary trie, loading audio: {} s".format(timer.elapsed))
    print("Audio file length: {} s".format(audio.shape[0] / sampling_rate))

    # Now it is enough to call:
    #   transcription = stt.recognize_audio(audio, sampling_rate)
    # if you don't need to access intermediate features like character probabilities or audio features.

    with Timer() as timer:
        audio_features = stt.extract_mfcc(audio, sampling_rate=sampling_rate)
    print("MFCC time: {} s".format(timer.elapsed))

    with Timer() as timer:
        character_probs = stt.extract_per_frame_probs(audio_features, wrap_iterator=tqdm)
    print("RNN time: {} s".format(timer.elapsed))

    with Timer() as timer:
        transcription = stt.decode_probs(character_probs)
    print("Beam search time: {} s".format(timer.elapsed))
    print("Overall time: {} s".format(timeit.default_timer() - start_time))

    print("\nTranscription and confidence score:")
    for candidate in transcription:
        print(
            "{}\t{}".format(
                candidate['conf'],
                candidate['text'],
            )
        )

def get_args():
    """
    Gets the arguments from the command line.
    """
    parser = argparse.ArgumentParser("python infer.py")
    required, optional = parser._action_groups

    required.add_argument("-i", "--input", help="Audio (.wav) file to diarize", required=True)
    required.add_argument("--xml", help="OpenVINO IR .xml file", required=True)
    required.add_argument("--bin", help="OpenVINO IR .bin file", required=True)
    required.add_argument("--config", help="model config file", required=True)
    optional.add_argument("--max-frame", help="Inference window length", default=45)
    optional.add_argument("--hop", help="Hop length of inference window", default=3)
    optional.add_argument("--plot", help="Plot the diarization result", default=True)
    args = parser.parse_args()
    return args

def infer_audio(args):
    """
    Perform speaker diarization
    """
    p = OpenVINOPredictor('./model/openvino/diarization.xml', './model/openvino/diarization.bin', './model/configs.pth', 45, 3)
    timestamps, speakers = p.predict('./tokyoleft.wav', plot=False)
    #print(timestamps, "timestamps===================>")

    result = [{"timestamp": timestamp.round(2), "speaker_id": speaker} for timestamp, speaker in zip(timestamps, speakers)] #透過round() 將timestamps進行四捨五入 透過zip 將timestamps 與 speakers 進行矩陣的行列互換
    pprint(result)
    
def cutaudiorun():
    
    smartspeakers = '-i ./tokyoleft.wav --xml ./model/openvino/diarization.xml --bin ./model/openvino/diarization.bin --config ./model/configs.pth'
    
    # recode.recoding()
    # get_args()
    infer_audio(smartspeakers)
    # mtaudiocut.cutaudio()
    
    start_time = timeit.default_timer()
    with Timer() as timer:
        args = build_argparser().parse_args()
        profile = get_profile(args.profile)

    stt = DeepSpeechPipeline(
        model = args.model,
        lm = args.lm,
        beam_width = args.beam_width,
        max_candidates = args.max_candidates,
        profile = profile,
        device = args.device,
        ie_extensions = [(args.device, args.cpu_extension)] if args.device == 'CPU' else [],
    )
        
    auu = "./chunks/"
    files = os.listdir(auu)
    files.sort(key = lambda x:int(x[:-4]))
        
    for filename in files:
        if filename.endswith(".wav"):
            file_directory = os.path.join(auu, filename)
            print(file_directory, "===============>")
            
            wave_read = wave.open(file_directory, 'rb')#(args.input, 'rb')
            channel_num, sample_width, sampling_rate, pcm_length, compression_type, _ = wave_read.getparams()
            assert sample_width == 2, "Only 16-bit WAV PCM supported"
            assert compression_type == 'NONE', "Only linear PCM WAV files supported"
            assert channel_num == 1, "Only mono WAV PCM supported"
            audio = np.frombuffer(wave_read.readframes(pcm_length * channel_num), dtype=np.int16).reshape((pcm_length, channel_num))
            #wave_read.close()
                
            print("Loading, including network weights, IE initialization, LM, building LM vocabulary trie, loading audio: {} s".format(timer.elapsed))
            print("Audio file length: {} s".format(audio.shape[0] / sampling_rate))

            # Now it is enough to call:
            #   transcription = stt.recognize_audio(audio, sampling_rate)
            # if you don't need to access intermediate features like character probabilities or audio features.

            with Timer() as timer:
                audio_features = stt.extract_mfcc(audio, sampling_rate=sampling_rate)
            print("MFCC time: {} s".format(timer.elapsed))

            with Timer() as timer:
                character_probs = stt.extract_per_frame_probs(audio_features, wrap_iterator=tqdm)
            print("RNN time: {} s".format(timer.elapsed))

            with Timer() as timer:
                transcription = stt.decode_probs(character_probs)
            print("Beam search time: {} s".format(timer.elapsed))
            print("Overall time: {} s".format(timeit.default_timer() - start_time))

            print("\nTranscription and confidence score:")
            for candidate in transcription:
                mylog = open("./mylog.log", "a+")
                mylog.write(candidate['text'])
                mylog.close()
                print(
                    "{}\t{}".format(
                        candidate['conf'],
                        candidate['text'],                            
                    )        
                )
    

if __name__ == '__main__':
    #main()
    cutaudiorun()
