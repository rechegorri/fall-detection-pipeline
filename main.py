import pyaudio
import numpy as np
import time
from pynput import keyboard
from data_analyser import DataInput, DataManagement

#Global Vars
RATE=16000
CHUNK=256
CHANNELS=1
fulldata = np.array([])
KEY_INTERRUPT = False
CAPTURE_WINDOW = 1

## Função de keyListener
def on_press(key):
  if key == keyboard.Key.esc:
    global KEY_INTERRUPT
    KEY_INTERRUPT = True

## Controle do stream de entrada de dados
def callback(in_data, frame_count, time_info, flag):
  global fulldata
  audio_data = np.frombuffer(in_data, dtype=np.float32)
  fulldata = np.append(fulldata,audio_data)
  if KEY_INTERRUPT:
    callback_flag = pyaudio.paComplete
  else:
    callback_flag = pyaudio.paContinue
  return(audio_data,callback_flag)

class StreamController:
  def __init__(self) -> None:
    self.p = pyaudio.PyAudio()

  def open(self) -> None:
    self.stream = self.p.open(
      format=pyaudio.paFloat32,
      channels=CHANNELS,
      rate=RATE,
      input=True,
      frames_per_buffer=CHUNK,
      stream_callback=callback
    )

  def close(self) -> None:
    self.stream.stop_stream()
    self.stream.close()

  def terminate(self) -> None:
    self.p.terminate()

  def is_active(self) -> bool:
    return self.stream.is_active()

## Rotina principal
if __name__ ==  "__main__":
  #Setup
  sc = StreamController()
  key_lis = keyboard.Listener(on_press=on_press)
  dm = DataManagement(CAPTURE_WINDOW)

  sc.open()
  key_lis.start()

  print("RECORDING")
  #Main loop
  while sc.is_active():
    time.sleep(CAPTURE_WINDOW)
    ##Envio: fulldata com timestamp
    dm.push(DataInput(fulldata,time.localtime()))
    fulldata = np.array([])
  print("STOP RECORDING")
  
  dm.close()
  sc.close()
  sc.terminate()
