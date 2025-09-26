import asyncio
import logging
import os
import subprocess
import time
import uuid
import requests
import json
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime
from gtts import gTTS
import edge_tts
import ChatTTS
import torch
import torchaudio
import soundfile as sf
from kokoro import KModel, KPipeline

logger = logging.getLogger(__name__)


class AbstractTTS(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_tts(self, text):
        pass


class GTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file")
        self.lang = config.get("lang")

    def _generate_filename(self, extension=".aiff"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"执行时间: {execution_time:.2f} 秒")

    def to_tts(self, text):
        tmpfile = self._generate_filename(".aiff")
        try:
            start_time = time.time()
            tts = gTTS(text=text, lang=self.lang)
            tts.save(tmpfile)
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.debug(f"生成TTS文件失败: {e}")
            return None


class MacTTS(AbstractTTS):
    """
    macOS 系统自带的TTS
    voice: say -v ? 可以打印所有语音
    """

    def __init__(self, config):
        super().__init__()
        self.voice = config.get("voice")
        self.output_file = config.get("output_file")

    def _generate_filename(self, extension=".aiff"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"执行时间: {execution_time:.2f} 秒")

    def to_tts(self, phrase):
        logger.debug(f"正在转换的tts：{phrase}")
        tmpfile = self._generate_filename(".aiff")
        try:
            start_time = time.time()
            res = subprocess.run(
                ["say", "-v", self.voice, "-o", tmpfile, phrase],
                shell=False,
                universal_newlines=True,
            )
            self._log_execution_time(start_time)
            if res.returncode == 0:
                return tmpfile
            else:
                logger.info("TTS 生成失败")
                return None
        except Exception as e:
            logger.info(f"执行TTS失败: {e}")
            return None


class EdgeTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file", "tmp/")
        self.voice = config.get("voice")

    def _generate_filename(self, extension=".wav"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")

    async def text_to_speak(self, text, output_file):
        communicate = edge_tts.Communicate(text, voice=self.voice)  # Use your preferred voice
        await communicate.save(output_file)

    def to_tts(self, text):
        tmpfile = self._generate_filename(".wav")
        start_time = time.time()
        try:
            asyncio.run(self.text_to_speak(text, tmpfile))
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.info(f"Failed to generate TTS file: {e}")
            return None


class CHATTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file", ".")
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)  # Set to True for better performance
        self.rand_spk = self.chat.sample_random_speaker()

    def _generate_filename(self, extension=".wav"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")

    def to_tts(self, text):
        tmpfile = self._generate_filename(".wav")
        start_time = time.time()
        try:
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=self.rand_spk,  # add sampled speaker
                temperature=.3,  # using custom temperature
                top_P=0.7,  # top P decode
                top_K=20,  # top K decode
            )
            params_refine_text = ChatTTS.Chat.RefineTextParams(
                prompt='[oral_2][laugh_0][break_6]',
            )
            wavs = self.chat.infer(
                [text],
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
            )
            try:
                torchaudio.save(tmpfile, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
            except:
                torchaudio.save(tmpfile, torch.from_numpy(wavs[0]), 24000)
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.error(f"Failed to generate TTS file: {e}")
            return None



# class KOKOROTTS(AbstractTTS):
#     def __init__(self, config):
#         from kokoro import KPipeline
#         self.output_file = config.get("output_file", ".")
#         self.lang = config.get("lang", "z")
#         self.pipeline = KPipeline(lang_code=self.lang)  # <= make sure lang_code matches voice
#         self.voice = config.get("voice", "zm_yunyang")
#
#     def _generate_filename(self, extension=".wav"):
#         return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")
#
#     def _log_execution_time(self, start_time):
#         end_time = time.time()
#         execution_time = end_time - start_time
#         logger.debug(f"Execution Time: {execution_time:.2f} seconds")
#
#     def to_tts(self, text):
#         tmpfile = self._generate_filename(".wav")
#         start_time = time.time()
#         try:
#             generator = self.pipeline(
#                 text, voice=self.voice,  # <= change voice here
#                 speed=1, split_pattern=r'\n+'
#             )
#             for i, (gs, ps, audio) in enumerate(generator):
#                 logger.debug(f"KOKOROTTS: i: {i}, gs：{gs}, ps：{ps}")  # i => index
#                 sf.write(tmpfile, audio, 24000)  # save each audio file
#             self._log_execution_time(start_time)
#             return tmpfile
#         except Exception as e:
#             logger.error(f"Failed to generate TTS file: {e}")
#             return None


class KOKOROTTS(AbstractTTS):
    def __init__(self, config):
        """
        config keys:
          - repo_id: HuggingFace repo ID for Kokoro model (e.g. 'hexgrad/Kokoro-82M-v1.1-zh')
          - lang:      'z' for Chinese, 'a' for multilingual/IPA fallback
          - voice:     e.g. 'zf_001' or 'zm_010'
          - output_dir: directory to write wav files to
          - sample_rate: audio sample rate (default 24000)
          - zero_padding: number of zeros between segments (default 5000)
        """
        self.repo_id      = config.get("repo_id", "hexgrad/Kokoro-82M-v1.1-zh")
        self.output_dir   = config.get("output_dir", ".")
        self.lang         = config.get("lang", "z")
        self.voice        = config.get("voice", "zf_001")
        self.sample_rate  = config.get("sample_rate", 24000)

        # device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load model if Chinese TTS
        self.model = None
        if self.lang == "z":
            self.model = KModel(repo_id=self.repo_id).to(self.device).eval()

        # set up pipelines
        self._setup_pipelines()

    def _setup_pipelines(self):
        # English/IPA fallback pipeline
        self.en_pipeline = KPipeline(
            lang_code="a", repo_id=self.repo_id, model=False
        )

        def en_callable(text):
            if text == "Kokoro":
                return "kˈOkəɹO"
            elif text == "Sol":
                return "sˈOl"
            return next(self.en_pipeline(text)).phonemes

        self.en_callable = en_callable

        # Main TTS pipeline
        self.pipeline = KPipeline(
            lang_code=self.lang,
            repo_id=self.repo_id,
            model=self.model,
            en_callable=self.en_callable
        )

    def _generate_filename(self, extension=".wav"):
        fname = f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}"
        return os.path.join(self.output_dir, fname)

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")
    @staticmethod
    def _speed_callable(len_ps: int) -> float:
        """
        piecewise linear function to slow down generation for long inputs
        """
        speed = 0.8
        if len_ps <= 83:
            speed = 1.0
        elif len_ps < 183:
            speed = 1.0 - (len_ps - 83) / 500.0
        return speed * 1.1

    def to_tts(self, text: str) -> str:
        """
        Generate TTS for `text`. Returns path to the combined WAV file.
        """
        output_file = self._generate_filename(".wav")
        start_time = time.time()

        try:
            generator = self.pipeline(
                text.replace("\n", " "),
                voice=self.voice,
                speed=self._speed_callable,
                #split_pattern=r"\n+"
            )
            result = next(generator)
            wav = result.audio
            sf.write(output_file, wav, self.sample_rate)

            self._log_execution_time(start_time)
            return output_file
        except Exception as e:
            logger.error(f"[KOKOROTTS] Failed to generate TTS: {e}")
            return ""


class CosyVoiceTTS(AbstractTTS):
    """
    CosyVoice TTS服务实现
    基于实际的CosyVoice API接口
    """
    
    def __init__(self, config):
        """
        初始化CosyVoice TTS服务
        
        config keys:
          - api_url: CosyVoice API服务地址 (默认: http://localhost:5001)
          - output_file: 输出文件目录
          - voice_type: 语音类型 (默认: "0")
          - reference_text: 参考文本 (默认: 示例文本)
          - prompt_audio_path: 提示音频文件路径
          - stream: 是否流式输出 (默认: False)
          - timeout: 请求超时时间 (默认: 30秒)
        """
        self.api_url = config.get("api_url", "http://localhost:5001")
        self.output_file = config.get("output_file", "tmp/")
        self.voice_type = config.get("voice_type", "0")
        self.reference_text = config.get("reference_text", 
            "今天有一张票是8662893339721，乘客叫罗婉晴，是银川到南宁的BK33号。你帮我看一下这个是不是取消保护到了3036。对银川到南宁的3162,6月21号。")
        self.prompt_audio_path = config.get("prompt_audio_path", 
            "/home/octopus/data/rwq/voice_generate/data/yxf_v1_cut.wav")
        self.stream = config.get("stream", False)
        self.timeout = config.get("timeout", 30)
        
        # 确保输出目录存在
        os.makedirs(self.output_file, exist_ok=True)
        
        # 验证API服务是否可用
        self._check_api_availability()

    def _check_api_availability(self):
        """检查CosyVoice API服务是否可用"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("CosyVoice API服务连接成功")
            else:
                logger.warning(f"CosyVoice API服务响应异常: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"无法连接到CosyVoice API服务: {e}")
            raise ConnectionError(f"CosyVoice API服务不可用: {e}")

    def _generate_filename(self, extension=".wav"):
        """生成输出文件名"""
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        """记录执行时间"""
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"CosyVoice TTS执行时间: {execution_time:.2f} 秒")

    def _prepare_request_data(self, text):
        """准备请求数据"""
        data = {
            "text": text,
            "voice_type": self.voice_type,
            "reference_text": self.reference_text,
            "prompt_audio_path": self.prompt_audio_path,
            "stream": self.stream
        }
        
        return data

    def _save_audio_response(self, response, output_file):
        """保存音频响应到文件"""
        try:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"保存音频文件失败: {e}")
            return False

    def to_tts(self, text: str) -> str:
        """
        将文本转换为语音
        
        Args:
            text (str): 要转换的文本
            
        Returns:
            str: 生成的音频文件路径，失败时返回空字符串（与KOKOROTTS保持一致）
        """
        if not text or not text.strip():
            logger.warning("输入文本为空")
            return ""
            
        output_file = self._generate_filename(".wav")
        start_time = time.time()
        
        try:
            # 准备请求数据
            data = self._prepare_request_data(text)
            
            # 发送请求到CosyVoice API
            response = requests.post(
                f"{self.api_url}/generate_voice",
                json=data,
                timeout=self.timeout,
                stream=self.stream
            )
            
            # 检查响应状态
            if response.status_code == 200:
                # 保存音频文件
                if self._save_audio_response(response, output_file):
                    self._log_execution_time(start_time)
                    logger.info(f"CosyVoice TTS生成成功: {output_file}")
                    return output_file
                else:
                    logger.error("保存音频文件失败")
                    return ""
            elif response.status_code == 400:
                error_msg = response.json().get("message", "请求参数不正确")
                logger.error(f"CosyVoice API参数错误: {error_msg}")
                return ""
            elif response.status_code == 500:
                error_msg = response.json().get("message", "服务器生成音频失败")
                logger.error(f"CosyVoice API服务器错误: {error_msg}")
                return ""
            else:
                logger.error(f"CosyVoice API请求失败: {response.status_code}")
                return ""
                
        except requests.exceptions.Timeout:
            logger.error("CosyVoice API请求超时")
            return ""
        except requests.exceptions.ConnectionError:
            logger.error("无法连接到CosyVoice API服务")
            return ""
        except Exception as e:
            logger.error(f"CosyVoice TTS生成失败: {e}")
            return ""

    def get_available_speakers(self):
        """
        获取可用的音色列表
        
        Returns:
            list: 可用音色列表，失败时返回空列表
        """
        try:
            response = requests.get(f"{self.api_url}/sft_spk", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("available_spks", [])
            else:
                logger.error(f"获取音色列表失败: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"获取音色列表异常: {e}")
            return []


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls = globals().get(class_name)
    if cls:
        # 创建并返回实例
        return cls(*args, **kwargs)
    else:
        raise ValueError(f"Class {class_name} not found")
