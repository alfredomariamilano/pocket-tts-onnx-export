from pocket_tts import TTSModel
import pprint

model = TTSModel.load_model()
voice = model.get_state_for_audio_prompt('hf://kyutai/tts-voices/alba-mackenna/casual.wav')
print('modules:', list(voice.keys()))
for module, state in voice.items():
    print(module)
    for k, t in state.items():
        if hasattr(t, 'shape'):
            print(' ', k, t.shape)
        else:
            print(' ', k, type(t))
