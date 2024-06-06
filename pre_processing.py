import numpy as np
import librosa
import PIL.Image as pimg

## Classe responsável por:
## - Converter o sinal com uso de STFT na escala Mel
## - Adequar o tamanho da arraym para um formato quadrado (128x128)

class PreProcessing:
  def __init__(self) -> None:
    self.hop_length = 512
    self.n_mels = 128
    self.sr = 16000

  def scale_minmax(self,x, min=0.0, max=1.0):
    x_std = (x - x.min()) / (x.max() - x.min())
    x_sca = x_std * (max - min) + min
    return x_sca

  def spectrogrtam_img(self, y):
    mels = librosa.feature.melspectrogram(
      y=y,
      sr=self.sr,
      n_mels=self.n_mels,
      n_fft=self.hop_length*2,
      hop_length=self.hop_length)
    mels = np.log(mels+1e-9)
    out = self.scale_minmax(mels,0, 255).astype(np.uint8)
    out = np.flip(out, axis=0)
    out = 255-out
    return out
  
  def pad_repeat(self, image:pimg.Image, width:int)->pimg.Image:
    if (image.width >= width):
      return image

    new_im = pimg.new('RGB', (width, image.height))
    offset = (width - image.width) // 2 % image.width

    if offset > 0:  # first part
      box = (image.width - offset, 0, image.width, image.height)
      new_im.paste(image.crop(box))

    while offset < width:
      new_im.paste(image, (offset, 0))
      offset += image.width
    return new_im

  def process(self, data):
    image = pimg.fromarray(self.spectrogrtam_img(data))
    image = self.pad_repeat(image,128)
    return np.array(image)