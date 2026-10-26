"""Create a deterministic PCM tone for an acoustic software smoke test.

This is not speech. No participant recording, text or voice is used.
"""
from array import array
import math
from pathlib import Path
import sys
import wave


def main():
    path=Path(__file__).resolve().parent/'project/data/wav/demo_voice/sample.wav'
    if path.exists():
        raise SystemExit('Demo WAV already exists; it has not been overwritten.')
    rate=16000
    samples=array('h')
    phase=0.
    for i in range(35200):
        t=i/rate
        if .1<=t<.9:
            hz=180.
            envelope=min(1.,(t-.1)/.02,(.9-t)/.02)
        elif 1.1<=t<2.:
            hz=215.
            envelope=min(1.,(t-1.1)/.02,(2.-t)/.02)
        else:
            hz=180.
            envelope=0.
        phase+=2.*math.pi*hz/rate
        samples.append(round(32767*.15*max(0.,envelope)*(math.sin(phase)+.2*math.sin(2*phase))))
    if sys.byteorder!='little':samples.byteswap()
    path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())
    print(path)


if __name__=='__main__':main()
