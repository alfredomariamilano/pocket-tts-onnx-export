from pocket_tts import TTSModel
import soundfile as sf

model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt('hf://kyutai/tts-voices/alba-mackenna/casual.wav')
audio = model.generate_audio(voice_state, "Hello onnx model", frames_after_eos=2, copy_state=True)
print('ref shape', audio.shape)
audio_np = audio.cpu().numpy()
# audio may be [1,samples] or [channels, samples]
if audio_np.ndim == 2 and audio_np.shape[0] == 1:
    audio_np = audio_np[0]
sf.write('reference.wav', audio_np, model.sample_rate)
print('wrote reference.wav')
