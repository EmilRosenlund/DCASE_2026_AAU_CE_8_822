import os
import glob
import numpy as np
import torch
import sys
from utils.data_utils import DataUtils
import yaml

class BaseEmbedding:
    def __init__(self, config):
        if config is None:
            with open("pipeline/config.yaml") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        # Paths based on environment
        env_paths = self.config['paths']['local'] if self.config['environment'] == "local" else self.config['paths']['ailab']
        self.data_path = env_paths['data']
        self.embeddings_output_path = env_paths['embeddings']
        self.models_folder_path = env_paths.get('models')

        # Embedding parameters
        embedding_cfg = self.config['embeddings']['params']
        self.split = embedding_cfg.get('split', 'train')
        self.max_samples = embedding_cfg.get('max_samples', 0)
        machine_types = embedding_cfg.get('machine_types', "")
        self.machine_types = [m.strip() for m in machine_types.split(',')] if machine_types else None
        self.use_gpu = embedding_cfg.get('use_gpu', True)
        
        # Initialize DataUtils
        self.data_utils = DataUtils(data_path=self.data_path)

        # Store embeddings
        self.embeddings = {}

        # Model_name
        model_list = self.config['embeddings']['modules']
        model_suffix = ", ".join(model_list)
        self.model_name = model_suffix


    def preprocess_audio(self, audio, sr):
        """Override this in the child class."""
        raise NotImplementedError

    def extract_embedding(self, inputs):
        """Override this in the child class."""
        raise NotImplementedError

    # --- PROCESSING MACHINES ---
    def process_machines(self, clap=False):
        print("\n" + "="*70)
        print(f"{self.model_name.upper()} EMBEDDING EXTRACTION")
        print("="*70)
        
        if self.split in ['train', 'both']:
            self._process_split('train', clap=clap)
        if self.split in ['test', 'both']:
            self._process_split('test', clap=clap)
            
        return self.embeddings

    # --- PROCESSING SPLIT ---
    def _process_split(self, split, clap=False):
        """Process a specific data split (train/test) based on config."""
        max_samples = self.max_samples
        machine_types = self.machine_types or self.data_utils.discover_machines()

        # Load data
        if split == 'train':
            data = self.data_utils.load_train_data_with_labels(machine_types=machine_types)
            for machine in data:
                data[machine]['labels'] = [0] * len(data[machine]['file_paths'])
        else:
            data = {}
            for machine in machine_types:
                test_path = os.path.join(self.data_utils.data_path, machine, 'test')
                if not os.path.exists(test_path):
                    print(f"Warning: Test directory for {machine} not found at {test_path}")
                    continue
                wav_files = sorted(glob.glob(os.path.join(test_path, '*.wav')))
                if wav_files:
                    file_paths, domains, labels = [], [], []
                    for f in wav_files:
                        file_paths.append(f)
                        basename = os.path.basename(f)
                        domains.append('source' if '_source_' in basename else 'target' if '_target_' in basename else 'unknown')
                        labels.append(0 if '_normal_' in basename else 1 if '_anomaly_' in basename else -1)
                    data[machine] = {'file_paths': file_paths, 'domains': domains, 'labels': labels}

        # Apply max_samples limit
        if max_samples and max_samples > 0:
            for machine in data:
                for key in ['file_paths', 'domains', 'labels']:
                    if key in data[machine]:
                        data[machine][key] = data[machine][key][:max_samples]

        if not data:
            print(f"No data found for {split} split")
            return

        print(f"Loaded {split} data for {len(data)} machine type(s)")

        for machine, machine_data in data.items():
            print(f"\nProcessing {machine} ({split})...")
            file_paths = machine_data['file_paths']
            domains = machine_data.get('domains', ['unknown'] * len(file_paths))
            labels = machine_data.get('labels', [0] * len(file_paths))

            embeddings_list, valid_paths, valid_domains, valid_labels = [], [], [], []

            for i, (file_path, domain, label) in enumerate(zip(file_paths, domains, labels)):
                try:
                    if clap:
                        audio_data = self.data_utils.process_audio_for_clap(file_path)
                        sample_rate = 48000
                    else:
                        sample_rate, audio_data = self.data_utils.load_wav_file(file_path)

                    if audio_data is None:
                        continue
                    inputs = self.preprocess_audio(audio_data, sample_rate)
                    embedding = self.extract_embedding(inputs)
                    embeddings_list.append(embedding)
                    valid_paths.append(file_path)
                    valid_domains.append(domain)
                    valid_labels.append(label)
                    if (i+1) % 100 == 0:
                        print(f"  Processed {i+1}/{len(file_paths)} files")
                except Exception as e:
                    print(f"  Error processing {file_path}: {e}")
                    continue

            key = f"{machine}_{split}"
            self.embeddings[key] = {
                'embeddings': np.array(embeddings_list),
                'file_paths': valid_paths,
                'domains': valid_domains,
                'labels': valid_labels
            }
            print(f"  ✓ Completed {machine} ({split}): {len(embeddings_list)} embeddings")


    # --- SUMMARY AND SAVE ---
    def get_embeddings_summary(self):
        print("\n" + "="*60)
        print("EMBEDDINGS SUMMARY")
        print("="*60)
        for key, data in self.embeddings.items():
            num_embeddings = len(data['embeddings'])
            if num_embeddings > 0:
                embedding_shape = data['embeddings'][0].shape
                print(f"{key}: {num_embeddings} samples, embedding shape: {embedding_shape}")
            else:
                print(f"{key}: 0 samples")

    # --- RUN EMBEDDINGS ---
    def run(self):
        self.process_machines()
        return self.embeddings