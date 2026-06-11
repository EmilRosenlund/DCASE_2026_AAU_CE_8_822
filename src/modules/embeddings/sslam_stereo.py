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

        # 2. Sæt i evaluerings-mode
        self.model.eval()
        print("SSLAM model loaded successfully!")

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

        self.model.to(self.device)
        print(f"Model moved to {self.device}!")

    def _process_single_channel(self, waveform, sample_rate, target_sr=16000):
        """Helper method to process a single 1D audio channel into a Mel-spectrogram."""
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0) # [1, samples]

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,            # Common for EAT/AST models
            sample_frequency=target_sr, 
            use_energy=False,
            window_type='povey',
            num_mel_bins=128,           
            dither=0.0,                 
            frame_shift=10.0,           
            frame_length=25.0,          
            high_freq=target_sr/2,      
            low_freq=20
        )

        # AudioSet Normalization (Critical for SSLAM)
        fbank = (fbank - (-4.268)) / 4.569

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

        # Extract Channel 1 (Close/Target) and Channel 2 (Distant/Context)
        c1_wave = torch.from_numpy(audio_data[0]).float()
        c2_wave = torch.from_numpy(audio_data[1]).float()

        # Process channels independently
        fbank_c1 = self._process_single_channel(c1_wave, sample_rate, target_sr)
        fbank_c2 = self._process_single_channel(c2_wave, sample_rate, target_sr)

        # Batch them together: shape becomes [2, 1, Time, Mel]
        # Index 0 is Channel 1, Index 1 is Channel 2
        batched_fbank = torch.cat([fbank_c1, fbank_c2], dim=0)
        
        return batched_fbank
    
    def extract_embedding(self, audio_tensor):
        """
        Extracts features, applies Cross-Attention, and returns the pooled residual embedding.
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
            
            # Split features back into C1 and C2
            # Shapes: [1, Time, 768]
            F1 = F.normalize(features[0:1, :, :], p=2, dim=-1) # Unit length
            F2 = F.normalize(features[1:2, :, :], p=2, dim=-1) # Unit length
            # --- 1. Cross-Attention (Residual / Anomaly Indicator) ---
            # Q=F2, K=F1, V=F1
            attended = F.scaled_dot_product_attention(query=F2, key=F1, value=F1)
            # What does F1 have that couldn't be explained by attending to F1 from F2's perspective?
            delta = F1 - attended
            
            # --- 2. Self-Attention (Spatial Filtering / Beamforming) ---
            # Combine the two channels along the feature dimension: [1, Time, 1536]
            # This allows the attention mechanism to look at both channels simultaneously at every time step
            combined = torch.cat([F1, F2], dim=-1) 
            spatial = F.scaled_dot_product_attention(query=combined, key=combined, value=combined)
            
            # Apply generalized mean pooling (p-norm) over the time dimension
            p = 3
            eps = 1e-6
            emb_residual = torch.mean(delta.clamp(min=eps).pow(p), dim=1).pow(1./p)
            emb_spatial = torch.mean(spatial.clamp(min=eps).pow(p), dim=1).pow(1./p)
            # L2-normalize both embeddings independently so they contribute equally to downstream distance metrics
            emb_residual = F.normalize(emb_residual, p=2, dim=1)
            emb_spatial = F.normalize(emb_spatial, p=2, dim=1)
            
            # Combine both normalized embeddings
            embedding = torch.cat([emb_residual, emb_spatial], dim=1)
        
        # Move to CPU and return as a 1D NumPy array
        return embedding.squeeze(0).cpu().numpy()

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