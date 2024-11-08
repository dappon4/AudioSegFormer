import librosa as lr
import matplotlib.pyplot as plt
import numpy as np

def preprocess_audio(audio_path, sr=16000):
    pass

if __name__ == '__main__':
    
    y, sr = lr.load('../datasets/DCASE2016_synthetic/dcase2016_task2_train_dev/dcase2016_task2_dev/sound/dev_1_ebr_-6_nec_1_poly_0.wav')
    print(sr)
    spectrogram = lr.feature.melspectrogram(y=y[:sr*10], sr=sr)
    print(y.shape)
    print(sr)
    print(lr.get_duration(y=y, sr=sr))
    print(spectrogram.shape)
    
    fig, ax = plt.subplots()
    
    S_dB = lr.power_to_db(spectrogram, ref=np.max)
    print(S_dB.shape)
    img = lr.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    
    plt.show()
    #print(spectrogram.shape)
    