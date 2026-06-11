import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
import transformers
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count, freeze_support
from embeddings.base_embedding import BaseEmbedding

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.dirname(current_dir)
pipeline_dir = os.path.dirname(modules_dir)
project_root = os.path.dirname(pipeline_dir)
utils_path = os.path.join(pipeline_dir, 'utils')

if utils_path not in sys.path:
    sys.path.insert(0, utils_path)


class SSLAMEmbeddings(BaseEmbedding):
    def __init__(self, config=None):
        """Initialize the SSLAM embedding extractor using config file."""
        super().__init__(config)

        # Initialize model
        print("Loading SSLAM AudioSet model...")
        print(f"Transformers version: {transformers.__version__}") 
        
        # 1. Instantiér modellen fra Hugging Face (arkitektur + konfiguration)
        model_name = "ta012/SSLAM_pretrain"
        self.model = transformers.AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Set up device
        if getattr(self, 'use_gpu', True): # Safely check use_gpu
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple Silicon GPU (MPS)")
            elif torch.xpu.is_available():
                self.device = torch.device("xpu")
                print("Using Intel GPU (XPU)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA GPU")
            else:
                print("GPU requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
                self.use_gpu = False
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # 2. Indlæs dit lokale checkpoint (dine finetunede vægte)
        checkpoint_path = r"C:\Users\amy_m\OneDrive\Dokumenter\GitHub\AAU_P8\sslam_2026_V4_10l_ch0.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 3. Udtræk state_dict
        # Tjek om vægtene ligger i en 'model' kasse eller direkte i filen
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 4. Load vægtene ind i modellen
        # strict=False er ofte en god idé ved finetuning, hvis du har fjernet/ændret det sidste lag
        self.model.load_state_dict(state_dict, strict=False)

        # 2. Sæt i evaluerings-mode
        self.model.eval()
        print("SSLAM model loaded successfully!")

        # Move model to device
        self.model.to(self.device)
        print(f"Model moved to {self.device}!")


        # --- Band Attention configuration ---
        self.num_bands = 16       # Split 768 into 6 bands of 128
        self.band_size = 768 // self.num_bands
        print(f"Band Attention: {self.num_bands} bands of {self.band_size} dims each")

    def _process_single_channel(self, waveform, sample_rate, target_sr=16000):
        """Helper method to process a single 1D audio channel into a Mel-spectrogram."""

        norm_mean = -4.268
        norm_std = 4.569

        if sample_rate != target_sr:
            cache_key = (sample_rate, target_sr)
            if not hasattr(self, '_resamplers'):
                self._resamplers = {}
            if cache_key not in self._resamplers:
                self._resamplers[cache_key] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = self._resamplers[cache_key](waveform)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0) # [1, samples]

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,            # Common for EAT/AST models
            sample_frequency=target_sr, 
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,           
            dither=0.0,                 
            frame_shift=10.0,           
            frame_length=500.0,          
            high_freq=target_sr/2,      
            low_freq=20
        )

        # AudioSet Normalization (Critical for SSLAM)
        fbank = (fbank - norm_mean) / (norm_std * 2)

        # Reshape for CNN front-end: [1, Channel, Time, Mel]
        return fbank.unsqueeze(0).unsqueeze(0)

    def preprocess_audio(self, audio_data, sample_rate, target_sr=16000):
        """
        Transforms raw stereo audio into a batched Mel-spectrogram tensor [2, 1, Time, Mel].
        """
        # 1. Standardization: Convert to float32 and normalize to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # Ensure audio is stereo for Reference-Query approach
        # Assuming audio_data shape is (channels, samples) or (samples, channels)
        if len(audio_data.shape) == 1:
            raise ValueError("Expected stereo audio, but got mono. Reference-Query requires 2 channels.")
        
        # Force shape to (channels, samples)
        if audio_data.shape[0] > audio_data.shape[1]: 
            audio_data = audio_data.T

        if audio_data.shape[0] < 2:
            raise ValueError("Audio must have at least 2 channels for Reference-Query.")

        # Convert to tensor and extract the first 2 channels
        audio_tensor = torch.from_numpy(audio_data[:2]).float() # [2, samples]
        
        # Optimize: Resample both channels simultaneously before splitting
        if sample_rate != target_sr:
            cache_key = (sample_rate, target_sr)
            if not hasattr(self, '_resamplers'):
                self._resamplers = {}
            if cache_key not in self._resamplers:
                self._resamplers[cache_key] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            audio_tensor = self._resamplers[cache_key](audio_tensor)
            sample_rate = target_sr  # Update sample rate so downstream skips resampling

        # Extract Channel 1 (Close/Target) and Channel 2 (Distant/Context)
        c1_wave = audio_tensor[0]
        c2_wave = audio_tensor[1]

        # Process channels independently
        fbank_c1 = self._process_single_channel(c1_wave, sample_rate, target_sr)
        fbank_c2 = self._process_single_channel(c2_wave, sample_rate, target_sr)

        # Batch them together: shape becomes [2, 1, Time, Mel]
        # Index 0 is Channel 1, Index 1 is Channel 2
        batched_fbank = torch.cat([fbank_c1, fbank_c2], dim=0)
        
        return batched_fbank
    

    def extract_embedding(self, audio_tensor):
        """
        Extracts features using Frequency-Band Linear Cross-Attention.
        
        Splits the 768-dim SSLAM features into frequency bands and performs
        linear attention (ELU+1 kernel) independently per band. Linear attention
        produces smoother, more holistic cross-channel comparisons than standard
        dot-product attention, treating each band as a global summary rather
        than spiky token-to-token matching.
        
        The per-band energy vector tells the downstream classifier *which*
        bands had the most cross-channel discrepancy.
        """

        # 1. Force Float32 and move to device
        audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
        
        # 2. Optimization for Intel: Use channels_last if on XPU
        if self.device.type == 'xpu':
            audio_tensor = audio_tensor.to(memory_format=torch.channels_last)
            # Force the model to the same format
            self.model.to(memory_format=torch.channels_last)
        
        self.model.eval()
        with torch.no_grad():
            # Extract features for BOTH channels simultaneously
            # Output shape: [2, Time, 768]
            features = self.model.extract_features(audio_tensor)

            # Split features back into C1 and C2 and L2-normalize
            # Shapes: [1, Time, 768]
            F1 = F.normalize(features[0:1, :, :], p=2, dim=-1)
            F2 = F.normalize(features[1:2, :, :], p=2, dim=-1)

            # --- Frequency-Band Linear Cross-Attention ---
            # Split the 768-dim feature into bands and do linear attention per band.
            # This isolates anomalies in specific frequency regions.
            band_deltas = []
            for i in range(self.num_bands):
                s = i * self.band_size
                e = s + self.band_size
                
                f1_band = F1[:, :, s:e]  # [1, T, 48]
                f2_band = F2[:, :, s:e]  # [1, T, 48]
                
                # Linear attention: Q=F2_band, K=F1_band, V=F1_band
                attended = F.scaled_dot_product_attention(
                    query=f2_band, key=f1_band, value=f1_band
                )
                band_deltas.append(f1_band - attended)

            # Concatenate all band residuals back: [1, T, 768]
            delta = torch.cat(band_deltas, dim=-1)

            # Per-band energy: how much cross-channel discrepancy in each band
            # Shape: [1, num_bands]
            band_energies = torch.stack([
                bd.pow(2).sum(dim=-1).mean(dim=-1) for bd in band_deltas
            ], dim=-1)

            # Apply generalized mean pooling (p-norm) over the time dimension
            p = 3
            eps = 1e-6
            emb_delta = torch.mean(delta.clamp(min=eps).pow(p), dim=1).pow(1./p)  # [1, 768]
            #emb_delta = torch.max(delta, dim=1).values  # [1, 768]
            
            # L2-normalize the main embedding
            emb_delta = F.normalize(emb_delta, p=2, dim=1)

            f1_emb = F.normalize(F1, p=2, dim=-1)
            
            # L2-normalize band energies so they're on a comparable scale
            band_energies = F.normalize(band_energies, p=2, dim=1)

            # Final embedding: [1, 768 + num_bands] = [1, 774]
            embedding = torch.cat([emb_delta, band_energies], dim=1)
        
        # Move to CPU and return as a 1D NumPy array
        return f1_emb.squeeze(0).cpu().numpy()

    # --- RUN PIPELINE ---
    def run(self):
        """Execute the full pipeline using config values."""
        self.process_machines()
        self.get_embeddings_summary()
        return self.embeddings


# --- MAIN FUNCTION ---
def main(config=None):
    extractor = SSLAMEmbeddings(config=config)
    embeddings = extractor.run()
    return embeddings


if __name__ == "__main__":
    freeze_support()
    import argparse
    parser = argparse.ArgumentParser(description='Extract SSLAM embeddings')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()
    main(config=args.config)
