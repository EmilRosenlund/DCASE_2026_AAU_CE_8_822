import sys
import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count, freeze_support
from embeddings.base_embedding import BaseEmbedding

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.dirname(current_dir)  # pipeline/modules
pipeline_dir = os.path.dirname(modules_dir)  # pipeline
project_root = os.path.dirname(pipeline_dir)  # AAU_P8

# Add utils to path
sys.path.insert(0, os.path.join(pipeline_dir, 'utils'))

# Add beats to path
sys.path.insert(0, os.path.join(project_root, 'beats'))

from BEATs import BEATs, BEATsConfig

# Import worker function from utils (must be importable for multiprocessing on Windows)
from embedding_worker import process_single_file


class BeatsEmbeddings(BaseEmbedding):
    def __init__(self, config: dict):
        """
        Initialize the BEATs model and DataUtils using a loaded config dictionary.

        Args:
            config: YAML-loaded config dictionary
            num_workers: Number of parallel workers (None = auto, 0 = sequential, ignored if use_gpu=True)
            use_gpu: Force GPU usage (None = auto from config, True = use GPU, False = use CPU)
        """
        super().__init__(config)

        # Paths
        if self.models_folder_path:
            self.model_path = os.path.join(self.models_folder_path, 'beats_2026_V3_4l_ch0.pt')
        else:
            self.model_path = os.path.join(project_root, 'models', 'beats_2026_V3_4l_ch0.pt')
        
        self.model_path = os.path.normpath(self.model_path)

        # Set up device
        if self.use_gpu:
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

        # Load BEATs model
        print("Loading BEATs model...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model = self.model.to(self.device)
        print(f"BEATs model loaded successfully on {self.device}!")

    def preprocess_audio(self, audio_data, sample_rate, target_sr=16000):
        """
        Preprocess audio data for BEATs model.
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: current sample rate
            target_sr: target sample rate (BEATs expects 16kHz)
        
        Returns:
            torch tensor of preprocessed audio
        """
        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # Force shape to (channels, samples)
        if audio_data.shape[0] > audio_data.shape[1]: 
            audio_data = audio_data.T

        if audio_data.shape[0] < 2:
            raise ValueError("Audio must have at least 2 channels for Reference-Query.")

        # Extract Channel 1 (Close/Target) and Channel 2 (Distant/Context)
        c1_wave = torch.from_numpy(audio_data[0]).float().unsqueeze(0)
        c2_wave = torch.from_numpy(audio_data[1]).float().unsqueeze(0)
        
        # Resample if needed (simple version - for production use librosa.resample)
        if sample_rate != target_sr:
            print(f"Warning: Sample rate is {sample_rate}Hz, BEATs expects {target_sr}Hz")
            # For now, we'll just use the audio as-is and warn the user
                
        audio_tensor_batched = torch.cat([c1_wave, c2_wave], dim=0)

        return audio_tensor_batched
    
    def extract_embedding(self, audio_tensor):
        """
        Extract embedding from audio tensor using BEATs model.

        Args:
            audio_tensor: preprocessed audio tensor

        Returns:
            embedding tensor of shape (768,) - pooled across time dimension
        """
        with torch.no_grad():
            # Move audio to device
            audio_tensor = audio_tensor.to(self.device)

            # Create padding mask (all False = no padding)
            padding_mask = torch.zeros(audio_tensor.shape, dtype=torch.bool).to(self.device)

            # Extract features - shape: (batch, time, features)
            embedding = self.model.extract_features(audio_tensor, padding_mask=padding_mask)[0]
            
            F1 = F.normalize(embedding[0:1, :, :], p=2, dim=-1) # Unit length
            F2 = F.normalize(embedding[1:2, :, :], p=2, dim=-1) # Unit length

            delta = F1 - F2
            
            # Apply generalized mean pooling (p-norm) over the time dimension
            p = 3
            eps = 1e-6
            embedding = torch.mean(F1.clamp(min=eps).pow(p), dim=1).pow(1./p)
        
        # Move to CPU and return as a 1D NumPy array
        return embedding.squeeze(0).cpu().numpy()
    
    def run(self):
           
        # Process machines
        self.process_machines()
        
        # Show summary
        self.get_embeddings_summary()        
        print("\n" + "="*60)
        print("COMPLETED!")
        print("="*60)
        return self.embeddings


# --- MAIN FUNCTION ---
def main(config=None):
    extractor = BeatsEmbeddings(config=config)
    embeddings = extractor.run()
    return embeddings


if __name__ == "__main__":
    print(f"Module name: {__name__}")