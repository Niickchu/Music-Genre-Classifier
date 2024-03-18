import os
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, set_start_method

# Set the input directory
input_dir = '../data/dataset_audio'

# Set the output directory
output_dir = '../data/dataset_spec/'
os.makedirs(output_dir, exist_ok=True)

# Check if a GPU is available
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')

def process_audio_files(args):
    genre_folder, filename = args
    audio_file = os.path.join(input_dir, genre_folder, filename)
    try:
        audio, sr = librosa.load(audio_file)
    except Exception as e:
        print(e)
        return
    
    # Convert the audio to a PyTorch tensor
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    
    # The stft window size and hop length to achieve a 512x512 spectrogram
    n_fft = 1024  # Window size for STFT
    hop_length = int(len(audio) / 511)  # Calculate hop length to achieve 512 time steps
    
    # Compute the spectrogram using PyTorch's stft method
    spectrogram = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                             window=torch.hann_window(n_fft), return_complex=True)
    spectrogram_magnitude = torch.abs(spectrogram).squeeze(0)
    spectrogram_magnitude = spectrogram_magnitude[:-1, :]

    # Resize the spectrogram to be 512x512 if it's not already
    if spectrogram_magnitude.shape != (512, 512):
        print("WRONG SIZE!\n")
        spectrogram_magnitude = torch.nn.functional.interpolate(spectrogram_magnitude.unsqueeze(0).unsqueeze(0),
                                                                size=(512, 512), mode='bilinear', align_corners=False)
        spectrogram_magnitude = spectrogram_magnitude.squeeze(0).squeeze(0)
    
    # Save the spectrogram as a PNG file
    output_file = os.path.splitext(filename)[0] + '.png'
    output_path = os.path.join(output_dir, genre_folder, output_file)
    os.makedirs(os.path.join(output_dir, genre_folder), exist_ok=True)
    
    # Plot the spectrogram using Matplotlib and save as 512x512 PNG
    plt.figure(figsize=(7, 7), dpi=72)
    plt.axis('off')
    plt.pcolormesh(np.arange(spectrogram_magnitude.size(1)), np.linspace(0, sr/2, spectrogram_magnitude.size(0)),
                   librosa.amplitude_to_db(spectrogram_magnitude.cpu().numpy(), ref=np.max), shading='auto', cmap='viridis')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(output_path, dpi=512/7, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f'Spectrogram saved to: {output_path}')

if __name__ == '__main__':
    set_start_method('spawn')

    # Get the number of CPU cores available
    num_cores = cpu_count()
    print(f'Using {num_cores} CPU cores')

    # Loop through each genre folder in the 'dataset_audio' folder
    args = []
    for genre_folder in os.listdir(input_dir):
        genre_path = os.path.join(input_dir, genre_folder)
        if os.path.isdir(genre_path):
            for filename in os.listdir(genre_path):
                if filename.endswith('.wav') or filename.endswith('.mp3'):
                    args.append((genre_folder, filename))
    # Use multiprocessing to process the audio files in parallel
    with Pool(processes=num_cores) as p:
        p.map(process_audio_files, args)
