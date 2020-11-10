import torch
from fairseq.models.wav2vec import Wav2VecModel
import librosa

cp = torch.load('../models/wav2vec_small_960h.pt')
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])

signal, sr = librosa.load('../static/test.wav')
tensors = torch.from_numpy(signal).unsqueeze(0)

z = model.feature_extractor(tensors)
c = model.feature_aggregator(z)
print('c:', c)
