

import os
from scipy.io import wavfile
import glob
import numpy as np
import torch
import librosa # Highly recommended for resampling

class DataUtils:
    def __init__(self, data_path=None):
        """Initialize DataUtils with optional data path.
        
        Args:
            data_path: Path to data directory. If None, defaults to '../data' relative to pipeline directory.
        """
        if data_path is None:
            # Default: go up from utils to pipeline, then to project root, then to data
            pipeline_dir = os.path.dirname(os.path.dirname(__file__))
            project_root = os.path.dirname(pipeline_dir)
            self.data_path = os.path.join(project_root, 'data')
        else:
            self.data_path = data_path
        
        self.train_dic = {}
        self.machine_types = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']

    def show_content_of_data_path(self):
        try:
            files = os.listdir(self.data_path)
            print(f"Contents of {self.data_path}:")
            for file in files:
                print(file)
        except FileNotFoundError:
            print(f"The directory {self.data_path} does not exist.")

    def discover_machines(self):
        """Discover available machine types in the data directory"""
        machines = []
        try:
            for item in os.listdir(self.data_path):
                item_path = os.path.join(self.data_path, item)
                if os.path.isdir(item_path):
                    machines.append(item)
        except FileNotFoundError:
            print(f"The directory {self.data_path} does not exist.")
        return machines

    def load_train_data(self, machine_types=None, domain=None):
        """
        Load all training wav files for specified machine types.

        Args:
            machine_types: List of machine types to load. If None, loads all available machines.
            domain: Filter by 'source', 'target', or None for both

        Returns:
            Dictionary with structure: {'machine1': [file_path1, file_path2, ...], 'machine2': [...]}
        """
        if machine_types is None:
            machine_types = self.discover_machines()
            print(f"Discovered machines: {machine_types}")
        
        self.train_dic = {}

        for machine in machine_types:
            train_path = os.path.join(self.data_path, machine, 'train')

            if not os.path.exists(train_path):
                print(f"Warning: Train directory for {machine} not found at {train_path}")
                continue

            # Get all .wav files from the train directory
            wav_files = glob.glob(os.path.join(train_path, '*.wav'))

            # Filter by domain if specified
            if domain == 'source':
                wav_files = [f for f in wav_files if '_source_' in os.path.basename(f)]
            elif domain == 'target':
                wav_files = [f for f in wav_files if '_target_' in os.path.basename(f)]

            if wav_files:
                self.train_dic[machine] = sorted(wav_files)
                domain_str = f" ({domain})" if domain else ""
                print(f"Loaded {len(wav_files)} files for {machine}{domain_str}")
            else:
                print(f"Warning: No wav files found for {machine}")

        return self.train_dic

    def load_train_data_with_labels(self, machine_types=None):
        """
        Load training data with source/target labels.

        Args:
            machine_types: List of machine types to load. If None, loads all available machines.

        Returns:
            Dictionary with structure:
            {'machine1': {'file_paths': [...], 'domains': [...]}, ...}
            where domains is a list of 'source' or 'target' labels
        """
        if machine_types is None:
            machine_types = self.discover_machines()

        train_data = {}

        for machine in machine_types:
            train_path = os.path.join(self.data_path, machine, 'train')
            train_path = os.path.normpath(train_path)

            if not os.path.exists(train_path):
                print(f"Warning: Train directory for {machine} not found at {train_path}")
                continue

            # Get all .wav files from the train directory
            wav_files = glob.glob(os.path.join(train_path, '*.wav'))

            if wav_files:
                file_paths = sorted(wav_files)
                # Extract domain labels from filenames
                domains = []
                for f in file_paths:
                    if '_source_' in os.path.basename(f):
                        domains.append('source')
                    elif '_target_' in os.path.basename(f):
                        domains.append('target')
                    else:
                        domains.append('unknown')

                train_data[machine] = {
                    'file_paths': file_paths,
                    'domains': domains
                }

                source_count = domains.count('source')
                target_count = domains.count('target')
                print(f"Loaded {machine}: {source_count} source, {target_count} target files")
            else:
                print(f"Warning: No wav files found for {machine}")

        return train_data

    def load_supplemental_data(self, machine_types=None, label_type=None):
        """
        Load supplemental data with clean/noise labels.

        Args:
            machine_types: List of machine types to load. If None, loads all available machines.
            label_type: Filter by 'clean' (machine_source), 'noise', or None for both

        Returns:
            Dictionary with structure:
            {'machine1': {'file_paths': [...], 'labels': [...]}, ...}
            where labels is a list of 'clean' or 'noise' labels
        """
        if machine_types is None:
            machine_types = self.discover_machines()

        supplemental_data = {}

        for machine in machine_types:
            supplemental_path = os.path.join(self.data_path, machine, 'supplemental')

            if not os.path.exists(supplemental_path):
                print(f"Warning: Supplemental directory for {machine} not found at {supplemental_path}")
                continue

            # Get all .wav files from the supplemental directory
            wav_files = glob.glob(os.path.join(supplemental_path, '*.wav'))

            if wav_files:
                file_paths = sorted(wav_files)
                # Extract labels from filenames
                labels = []
                filtered_paths = []
                
                for f in file_paths:
                    filename = os.path.basename(f)
                    if 'machine_source' in filename:
                        label = 'clean'
                    elif 'noise' in filename:
                        label = 'noise'
                    else:
                        label = 'unknown'
                    
                    # Filter by label_type if specified
                    if label_type is None or label == label_type:
                        filtered_paths.append(f)
                        labels.append(label)

                if filtered_paths:
                    supplemental_data[machine] = {
                        'file_paths': filtered_paths,
                        'labels': labels
                    }

                    clean_count = labels.count('clean')
                    noise_count = labels.count('noise')
                    label_str = f" ({label_type})" if label_type else ""
                    print(f"Loaded {machine} supplemental{label_str}: {clean_count} clean, {noise_count} noise files")
            else:
                print(f"Warning: No supplemental wav files found for {machine}")

        return supplemental_data

    def load_wav_file(self, file_path):
        """Load a single wav file"""
        try:
            sample_rate, data = wavfile.read(file_path)
            return sample_rate, data
        except FileNotFoundError:
            print(f"The file {file_path} does not exist.")
            return None, None
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def process_audio_for_clap(self, file_path):
        sample_rate, data = wavfile.read(file_path)
        
        # 1. Convert to Float32 and Normalize to [-1.0, 1.0]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        
        # 2. Handle Stereo (Shape is usually [samples, channels])
        if len(data.shape) > 1:
            # Mean across the channel axis (axis 1)
            data = np.mean(data, axis=1)
        
        # 3. Resample to 48kHz (CRITICAL for this CLAP model)
        target_sr = 48000 
        if sample_rate != target_sr:
            # Using librosa for high-quality resampling
            data_audio = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr

        return data_audio
   
    def get_from_train_dic(self, key):
        """Get file paths for a specific machine type"""
        return self.train_dic.get(key, None)
    
    def get_machine_count(self):
        """Get count of files per machine"""
        return {machine: len(files) for machine, files in self.train_dic.items()}

    def list_from_data_path(self):
        try:
            files = os.listdir(self.data_path)
            return files
        except FileNotFoundError:
            print(f"The directory {self.data_path} does not exist.")
            return []
    
    def preprocess_audio(self, audio_data, sample_rate, target_sr=16000):
        """
        Preprocess audio data for model input.
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: current sample rate
            target_sr: target sample rate (default 16kHz)
        
        Returns:
            torch tensor of preprocessed audio
        """
        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # Handle stereo - convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Warn if sample rate mismatch
        if sample_rate != target_sr:
            print(f"Warning: Sample rate is {sample_rate}Hz, expected {target_sr}Hz")
        
        # Convert to torch tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
        
        return audio_tensor
    
    def load_and_preprocess(self, file_path, target_sr=16000):
        """
        Load and preprocess a single audio file.
        
        Args:
            file_path: path to audio file
            target_sr: target sample rate
        
        Returns:
            preprocessed audio tensor, or None if error
        """
        sample_rate, audio_data = self.load_wav_file(file_path)
        if audio_data is None:
            return None
        return self.preprocess_audio(audio_data, sample_rate, target_sr)
    
    def iter_machine_audio(self, machine_type, target_sr=16000):
        """
        Iterator that yields preprocessed audio tensors for a machine type.
        
        Args:
            machine_type: machine type to iterate over
            target_sr: target sample rate
        
        Yields:
            (file_path, audio_tensor) tuples
        """
        if machine_type not in self.train_dic:
            print(f"Warning: {machine_type} not loaded. Call load_train_data() first.")
            return
        
        for file_path in self.train_dic[machine_type]:
            audio_tensor = self.load_and_preprocess(file_path, target_sr)
            if audio_tensor is not None:
                yield file_path, audio_tensor
    
    def load_audio_file(self, file_path):
        """Load a single audio file with metadata.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing sample rate and audio data, or None if error
        """
        try:
            sample_rate, audio_data = wavfile.read(file_path)
            return {
                'path': file_path,
                'sample_rate': sample_rate,
                'data': audio_data,
                'duration': len(audio_data) / sample_rate
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_audio_samples(self, num_samples=20, machine_types=None, domain=None, verbose=True):
        """Load audio samples from each machine type in the training set.
        
        Args:
            num_samples: Number of samples to load per machine (default: 20)
            machine_types: List of machine types to load. If None, loads all available machines.
            domain: Filter by 'source', 'target', or None for both
            verbose: Print progress information
        
        Returns:
            Dictionary with structure: 
            {
                'machine1': [
                    {'path': ..., 'sample_rate': ..., 'data': ..., 'duration': ...},
                    ...
                ],
                ...
            }
        """
        if verbose:
            print(f"\nLoading {num_samples} audio samples from each machine...")
            print(f"Data path: {self.data_path}")
        
        # Load training file paths for all machines
        train_files = self.load_train_data(machine_types=machine_types, domain=domain)
        
        audio_data = {}
        
        # Load first N audio samples from each machine
        for machine, file_paths in train_files.items():
            if verbose:
                print(f"\nProcessing {machine}...")
            
            # Take only the first N samples
            selected_files = file_paths[:num_samples]
            
            audio_data[machine] = []
            
            for i, file_path in enumerate(selected_files, 1):
                audio_info = self.load_audio_file(file_path)
                
                if audio_info:
                    audio_data[machine].append(audio_info)
                    
                    # Print progress every 5 files
                    if verbose and (i % 5 == 0 or i == len(selected_files)):
                        print(f"  Loaded {i}/{len(selected_files)} samples")
            
            # Print summary for this machine
            if verbose and audio_data[machine]:
                sample_rates = set(s['sample_rate'] for s in audio_data[machine])
                durations = [s['duration'] for s in audio_data[machine]]
                avg_duration = np.mean(durations)
                
                print(f"  ✓ Loaded {len(audio_data[machine])} samples")
                print(f"    Sample rate(s): {sample_rates}")
                print(f"    Average duration: {avg_duration:.2f}s")
        
        return audio_data
    
    def load_supplemental_audio_samples(self, num_samples=20, machine_types=None, label_type=None, verbose=True):
        """Load supplemental audio samples from each machine type.
        
        Args:
            num_samples: Number of samples to load per machine (default: 20)
            machine_types: List of machine types to load. If None, loads all available machines.
            label_type: Filter by 'clean', 'noise', or None for both
            verbose: Print progress information
        
        Returns:
            Dictionary with structure: 
            {
                'machine1': [
                    {'path': ..., 'sample_rate': ..., 'data': ..., 'duration': ..., 'label': 'clean'/'noise'},
                    ...
                ],
                ...
            }
        """
        if verbose:
            print(f"\nLoading {num_samples} supplemental audio samples from each machine...")
            print(f"Data path: {self.data_path}")
        
        # Load supplemental file paths for all machines
        supplemental_files = self.load_supplemental_data(machine_types=machine_types, label_type=label_type)
        
        audio_data = {}
        
        # Load first N audio samples from each machine
        for machine, data_dict in supplemental_files.items():
            if verbose:
                print(f"\nProcessing {machine}...")
            
            file_paths = data_dict['file_paths']
            labels = data_dict['labels']
            
            # Take only the first N samples
            selected_files = file_paths[:num_samples]
            selected_labels = labels[:num_samples]
            
            audio_data[machine] = []
            
            for i, (file_path, label) in enumerate(zip(selected_files, selected_labels), 1):
                audio_info = self.load_audio_file(file_path)
                
                if audio_info:
                    audio_info['label'] = label  # Add label to metadata
                    audio_data[machine].append(audio_info)
                    
                    # Print progress every 5 files
                    if verbose and (i % 5 == 0 or i == len(selected_files)):
                        print(f"  Loaded {i}/{len(selected_files)} samples")
            
            # Print summary for this machine
            if verbose and audio_data[machine]:
                sample_rates = set(s['sample_rate'] for s in audio_data[machine])
                durations = [s['duration'] for s in audio_data[machine]]
                avg_duration = np.mean(durations)
                clean_count = sum(1 for s in audio_data[machine] if s['label'] == 'clean')
                noise_count = sum(1 for s in audio_data[machine] if s['label'] == 'noise')
                
                print(f"  ✓ Loaded {len(audio_data[machine])} samples ({clean_count} clean, {noise_count} noise)")
                print(f"    Sample rate(s): {sample_rates}")
                print(f"    Average duration: {avg_duration:.2f}s")
        
        return audio_data
    
    def get_audio_summary(self, audio_data):
        """Get a summary of loaded audio data.
        
        Args:
            audio_data: Dictionary of loaded audio samples (from load_audio_samples or load_supplemental_audio_samples)
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_machines': len(audio_data),
            'machines': {}
        }
        
        for machine, samples in audio_data.items():
            machine_summary = {
                'num_samples': len(samples),
                'sample_rates': list(set(s['sample_rate'] for s in samples)),
                'total_duration': sum(s['duration'] for s in samples),
                'avg_duration': np.mean([s['duration'] for s in samples]) if samples else 0
            }
            
            # Check if labels exist (for supplemental data)
            if samples and 'label' in samples[0]:
                labels = [s['label'] for s in samples]
                machine_summary['clean_count'] = labels.count('clean')
                machine_summary['noise_count'] = labels.count('noise')
            
            summary['machines'][machine] = machine_summary
        
        return summary