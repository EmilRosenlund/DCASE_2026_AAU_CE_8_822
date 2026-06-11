import os
import re
import logging
import librosa
import numpy as np
import scipy.io.wavfile as wav
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


logger = logging.getLogger(__name__)

class DCASE_Dataset:
    def __init__(self, debug=False):
        self.debug = debug
        self.local_env = True  # Set to False when running on cluster with actual paths
        
        if self.local_env:
            self.data_path = r"C:\Users\emilr\Documents\GitHub\Semester8\DCASE\data"
            self.aug_base_dir = r"C:\Users\emilr\Documents\GitHub\Semester8\DCASE\augmentation\augmented"
        else:
            self.data_path = "/ceph/project/P8_DCASE/data_2026"
            self.aug_base_dir = "/ceph/project/P8_DCASE/data_2026_AUG"

        self.machines = ["fan", "valve", "slider", "ToyCar", "ToyTrain", "gearbox", "bearing"]
        #self.machines = ["fan", "valveEmu", "sliderEmu", "ToyCarEmu", "ToyCar", "gearboxEmu", "bearingEmu"]
        self.class_map = {}
        self.domain_map = {}

    def extract_machine_identity(self, machine, filename):
        """
        For POEM, we want the Class to be the broad machine type.
        This allows us to make specific unit/env features orthogonal to this label.
        """
        return machine.lower()

    def build_domain(self, filename, machine_name):
        """
        Captures all variation (Unit, Speed, Mic, Augmentation) 
        and prefixes with machine_name to ensure domains are unique per class.
        """
        name = filename.lower()
        # Start with the machine name to prevent mixing (e.g., 'fan_source_clean')
        domain_parts = [machine_name.lower()]

        # 1. Source vs Target
        domain_parts.append("source" if "source" in name else "target")

        # 2. Fan / Slider / Bearing: Unit A, B, C (Pattern: _n_A)
        unit_match = re.search(r"_n_([a-z])", name)
        if unit_match:
            domain_parts.append(f"unit{unit_match.group(1).upper()}")

        # 3. Gearbox: Process A, B (Pattern: _pro_A)
        pro_match = re.search(r"_pro_([a-z])", name)
        if pro_match:
            domain_parts.append(f"pro{pro_match.group(1).upper()}")

        # 4. Valve: Pattern v1pat_XX_v2pat_XX
        v_match = re.search(r"v1pat_(\d+)_v2pat_(\d+)", name)
        if v_match:
            domain_parts.append(f"v1p{v_match.group(1)}v2p{v_match.group(2)}")

        # 5. ToyCar/ToyTrain: car_a1, spd_11v, mic_1
        car = re.search(r"car_([a-z]\d)", name)
        if car: domain_parts.append(car.group(1).upper())
        
        spd = re.search(r"spd_(\d+)", name)
        if spd: domain_parts.append(f"spd{spd.group(1)}")
        
        mic = re.search(r"mic_(\d)", name)
        if mic: domain_parts.append(f"mic{mic.group(1)}")

        # 6. Augmentation tag
        aug = self.detect_augmentation(filename)
        domain_parts.append(aug)

        return "_".join(domain_parts)

    def detect_augmentation(self, filename):
        name = filename.lower()
        # Specific tags
        for i in range(1, 8):
            if f"rir{i}" in name or f"aug{i}" in name:
                return f"rir{i}"
        
        if "noise" in name or "noisy" in name:
            return "noise"

        if "clean" in name:
            return "clean"

        # If it's a standard training file, it's original.
        # Only warn if it's in the augmented folder but has no tag.
        return "original"

   

    def get_class_id(self, name):
        if name not in self.class_map: 
            self.class_map[name] = len(self.class_map)
        return self.class_map[name]

    def get_domain_id(self, name):
        if name not in self.domain_map: 
            self.domain_map[name] = len(self.domain_map)
        return self.domain_map[name]

    def discover_train(self):
        dataset = []
        # Temporary storage for debug view
        debug_info = []

        for machine in self.machines:
            # 1. ORIGINALS
            orig_machine_root = next((d for d in os.listdir(self.data_path) 
                                   if d.lower() == machine.lower()), None)
            
            if orig_machine_root:
                orig_dir = os.path.join(self.data_path, orig_machine_root, "train")
                if os.path.isdir(orig_dir):
                    for file in os.listdir(orig_dir):
                        if file.endswith(".wav") and self.detect_augmentation(file) == "original":
                            path = os.path.join(orig_dir, file)
                            c_name = self.extract_machine_identity(machine, file)
                            d_name = self.build_domain(file, machine)
                            
                            c_id = self.get_class_id(c_name)
                            d_id = self.get_domain_id(d_name)
                            
                            dataset.append((path, c_id, d_id))
                            if self.debug and len(debug_info) < 5000: # Limit samples to avoid lag
                                debug_info.append({"machine": machine, "file": file, "class": c_name, "domain": d_name})

            # 2. AUGMENTED
            if os.path.exists(self.aug_base_dir):
                aug_machine_root = next((d for d in os.listdir(self.aug_base_dir) 
                                      if d.lower() == machine.lower()), None)
                
                if aug_machine_root:
                    aug_dir = os.path.join(self.aug_base_dir, aug_machine_root, "train")
                    if os.path.isdir(aug_dir):
                        for file in os.listdir(aug_dir):
                            if file.endswith(".wav") and self.detect_augmentation(file) != "original":
                                path = os.path.join(aug_dir, file)
                                c_name = self.extract_machine_identity(machine, file)
                                d_name = self.build_domain(file, machine)
                                
                                c_id = self.get_class_id(c_name)
                                d_id = self.get_domain_id(d_name)
                                
                                dataset.append((path, c_id, d_id))
                                if self.debug:
                                    debug_info.append({"machine": machine, "file": file, "class": c_name, "domain": d_name})

        if self.debug:
            self.print_debug_summary(debug_info)

        return dataset

    def print_debug_summary(self, info):
        print("\n" + "="*80)
        print(f"{'DEBUG: DCASE DATASET DISCOVERY':^80}")
        print("="*80)
        
        # 1. Show a few mapping examples
        print(f"\n{'[ Sample Identity Mapping ]':<80}")
        print(f"{'Filename':<45} | {'Class (Identity)':<20} | {'Domain'}")
        print("-"*80)
        # Show first 2 samples of each machine found
        seen_machines = set()
        for item in info:
            if list(seen_machines).count(item['machine']) < 2:
                print(f"{item['file'][:42]:<45} | {item['class']:<20} | {item['domain']}")
                seen_machines.add(item['machine'])

        # 2. Show the Class Map
        print(f"\n{'[ Class ID Map ]':<80}")
        for name, idx in sorted(self.class_map.items()):
            print(f" ID {idx:02d} -> {name}")

        # 3. Show the Domain Map
        print(f"\n{'[ Domain ID Map ]':<80}")
        for name, idx in sorted(self.domain_map.items()):
            print(f" ID {idx:02d} -> {name}")
        
        print("="*80 + "\n")

    def load_audio(self, file_path):
        #print(f"[DCASE_Dataset] Attempting to load: {file_path}")
        try:
            audio, sample_rate = librosa.load(file_path, sr=None)
            return audio, sample_rate
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None


class DCASEAudioDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list, sample_rate: int = 16_000,
                 window_sec: float = 5.0, mode: str = "train",
                 crops_per_sample: int = 1,
                 preload_to_ram: bool = False,
                 preload_max_ram_bytes: Optional[int] = None,
                 preload_num_workers: int = 1):
        self.samples = samples
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_sec)
        self.mode = mode
        self.crops_per_sample = max(1, int(crops_per_sample)) if mode == "train" else 1
        self._dcase = DCASE_Dataset()
        self.preload_to_ram = bool(preload_to_ram)
        self.preload_num_workers = max(1, int(preload_num_workers))
        self.preload_max_ram_bytes = (
            int(preload_max_ram_bytes)
            if preload_max_ram_bytes is not None
            else None
        )
        self._audio_cache: Dict[str, np.ndarray] = {}
        self._cached_bytes = 0

        if self.preload_to_ram:
            self._preload_audio_cache()

    def __len__(self) -> int:
        return len(self.samples) * self.crops_per_sample

    def _crop(self, audio: np.ndarray) -> np.ndarray:
        win = self.window_samples
        if audio.ndim == 1:
            audio = audio[None, :]

        if audio.shape[-1] <= win:
            pad = win - audio.shape[-1]
            return np.pad(audio, ((0, 0), (0, pad)), mode="reflect")

        start = (
            np.random.randint(0, audio.shape[-1] - win)
            if self.mode == "train"
            else (audio.shape[-1] - win) // 2
        )
        return audio[:, start: start + win]

    def _load_stereo_audio(self, file_path: str):
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            return audio, sr
        except Exception:
            return None, None

    def _resample_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.sample_rate:
            return audio
        if audio.ndim == 1:
            return librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return np.stack([
            librosa.resample(audio[ch], orig_sr=sr, target_sr=self.sample_rate)
            for ch in range(audio.shape[0])
        ], axis=0)

    def _prepare_base_audio(self, file_path: str) -> np.ndarray:
        audio, sr = self._load_stereo_audio(file_path)
        if audio is None:
            return np.zeros((2, self.window_samples), dtype=np.float32)
        audio = self._resample_audio(audio, sr).astype(np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio

    def _preload_audio_cache(self) -> None:
        unique_paths = []
        seen = set()
        for path, _, _ in self.samples:
            if path in seen:
                continue
            seen.add(path)
            unique_paths.append(path)

        use_parallel = self.preload_num_workers > 1 and len(unique_paths) > 1
        loaded = 0
        if use_parallel:
            with ThreadPoolExecutor(max_workers=self.preload_num_workers) as executor:
                for path, audio in zip(unique_paths, executor.map(self._prepare_base_audio, unique_paths)):
                    n_bytes = int(audio.nbytes)

                    if (
                        self.preload_max_ram_bytes is not None
                        and self._cached_bytes + n_bytes > self.preload_max_ram_bytes
                    ):
                        break

                    self._audio_cache[path] = audio
                    self._cached_bytes += n_bytes
                    loaded += 1
        else:
            for path in unique_paths:
                audio = self._prepare_base_audio(path)
                n_bytes = int(audio.nbytes)

                if (
                    self.preload_max_ram_bytes is not None
                    and self._cached_bytes + n_bytes > self.preload_max_ram_bytes
                ):
                    break

                self._audio_cache[path] = audio
                self._cached_bytes += n_bytes
                loaded += 1

    def _get_base_audio(self, file_path: str) -> np.ndarray:
        if not self.preload_to_ram:
            return self._prepare_base_audio(file_path)
        audio = self._audio_cache.get(file_path)
        if audio is not None:
            return audio
        return self._prepare_base_audio(file_path)

    def _to_three_channel_input(self, audio: np.ndarray) -> np.ndarray:
        # Accept mono/stereo and always convert to [ch1, ch2, ch1-ch2].
        if audio.ndim == 1:
            ch1 = audio.astype(np.float32)
            ch2 = audio.astype(np.float32)
        else:
            if audio.shape[0] == 1:
                ch1 = audio[0].astype(np.float32)
                ch2 = audio[0].astype(np.float32)
            else:
                ch1 = audio[0].astype(np.float32)
                ch2 = audio[1].astype(np.float32)
        residual = ch1 - ch2
        return np.stack([ch1, ch2, residual], axis=0).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        base_idx = idx // self.crops_per_sample
        file_path, machine_label, domain_label = self.samples[base_idx]

        audio = self._get_base_audio(file_path)
        audio = self._crop(audio)
        audio = self._to_three_channel_input(audio)
        return torch.from_numpy(audio), int(machine_label), int(domain_label)


class AUCValidationDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list, sample_rate: int = 16_000,
                 window_sec: float = 5.0, use_sliding_windows: bool = False,
                 num_windows: int = 5):
        self.samples = samples
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_sec)
        self._dcase = DCASE_Dataset()
        self.use_sliding_windows = bool(use_sliding_windows)
        self.num_windows = max(1, int(num_windows))

        if self.use_sliding_windows:
            self.window_to_sample_idx = []
            for sample_idx in range(len(self.samples)):
                for window_idx in range(self.num_windows):
                    self.window_to_sample_idx.append((sample_idx, window_idx))
        else:
            self.window_to_sample_idx = [(i, 0) for i in range(len(self.samples))]

    def __len__(self) -> int:
        return len(self.window_to_sample_idx)

    def _sliding_window_crop(self, audio: np.ndarray, window_idx: int) -> np.ndarray:
        win = self.window_samples
        if audio.shape[-1] <= win:
            pad = win - audio.shape[-1]
            return np.pad(audio, ((0, 0), (0, pad)), mode="reflect")

        total_length = audio.shape[-1]
        if self.num_windows <= 1:
            start = (total_length - win) // 2
        else:
            stride = max(1, (total_length - win) // (self.num_windows - 1))
            start = min(window_idx * stride, total_length - win)
        return audio[:, start:start + win]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_idx, window_idx = self.window_to_sample_idx[idx]
        file_path, label = self.samples[sample_idx]

        try:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
        except Exception:
            audio, sr = None, None

        if audio is None:
            audio = np.zeros((2, self.window_samples), dtype=np.float32)
            sr = self.sample_rate

        if sr != self.sample_rate:
            if audio.ndim == 1:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            else:
                audio = np.stack([
                    librosa.resample(audio[ch], orig_sr=sr, target_sr=self.sample_rate)
                    for ch in range(audio.shape[0])
                ], axis=0)

        if audio.ndim == 1:
            audio = audio[None, :]
        if audio.shape[0] == 1:
            audio = np.concatenate([audio, audio], axis=0)

        audio = audio.astype(np.float32)
        if self.use_sliding_windows:
            audio = self._sliding_window_crop(audio, window_idx)
        else:
            if audio.shape[-1] > self.window_samples:
                start = (audio.shape[-1] - self.window_samples) // 2
                audio = audio[:, start:start + self.window_samples]
            elif audio.shape[-1] < self.window_samples:
                pad = self.window_samples - audio.shape[-1]
                audio = np.pad(audio, ((0, 0), (0, pad)), mode="reflect")

        ch1 = audio[0]
        ch2 = audio[1]
        residual = ch1 - ch2
        audio_3ch = np.stack([ch1, ch2, residual], axis=0).astype(np.float32)
        return torch.from_numpy(audio_3ch), int(label), int(sample_idx)


def _auc_window_settings(cfg: dict) -> Tuple[bool, int]:
    num_windows = max(5, int(cfg.get("auc_val_num_windows", 5)))
    return True, num_windows


def _is_anomaly(filename: str) -> int:
    name = filename.lower()
    if "anomaly" in name or "abnormal" in name:
        return 1
    if "normal" in name:
        return 0
    return -1


def _find_machine_root(data_root: str, machine: str) -> Optional[str]:
    if not os.path.exists(data_root):
        return None
    return next((d for d in os.listdir(data_root) if d.lower() == machine.lower()), None)


def _path_has_machine_segment(path: str, machine: str) -> bool:
    machine_l = machine.lower()
    parts = os.path.normpath(path).replace("\\", "/").split("/")
    return any(part.lower() == machine_l for part in parts)


def discover_target_test_samples(machine: str, data_root: str) -> list:
    samples = []
    dcase = DCASE_Dataset()
    machine_root = _find_machine_root(data_root, machine)
    if machine_root is None:
        return samples

    test_dir = None
    for candidate in ("test", "test_data"):
        path = os.path.join(data_root, machine_root, candidate)
        if os.path.isdir(path):
            test_dir = path
            break

    if test_dir is None:
        return samples

    for fname in os.listdir(test_dir):
        if not fname.endswith(".wav"):
            continue
        label = _is_anomaly(fname)
        if label < 0:
            continue
        path = os.path.join(test_dir, fname)
        domain_str = dcase.build_domain(fname, machine)
        samples.append((path, 1, domain_str))

    return samples


def discover_target_test_auc_samples(machine: str, data_root: str) -> list:
    samples = []
    machine_root = _find_machine_root(data_root, machine)
    if machine_root is None:
        return samples

    test_dir = None
    for candidate in ("test", "test_data"):
        path = os.path.join(data_root, machine_root, candidate)
        if os.path.isdir(path):
            test_dir = path
            break

    if test_dir is None:
        return samples

    for fname in os.listdir(test_dir):
        if not fname.endswith(".wav"):
            continue
        label = _is_anomaly(fname)
        if label < 0:
            continue
        samples.append((os.path.join(test_dir, fname), label))
    return samples


def aggregate_windowed_embeddings(model, dataloader, device, aggregation: str,
                                  return_labels: bool):
    model.eval()
    window_embeddings = {}
    with torch.no_grad():
        for audio, labels, sample_idxs in dataloader:
            audio = audio.to(device)
            z = model.module.get_embeddings(audio) if hasattr(model, "module") else model.get_embeddings(audio)
            emb_np = z.cpu().numpy()
            if isinstance(sample_idxs, torch.Tensor):
                sample_idxs = sample_idxs.tolist()
            for i, sample_idx in enumerate(sample_idxs):
                if sample_idx not in window_embeddings:
                    window_embeddings[sample_idx] = []
                window_embeddings[sample_idx].append(emb_np[i])

    if not window_embeddings:
        return (None, None) if return_labels else None

    samples = getattr(dataloader.dataset, "samples", None)
    emb_list = []
    lbl_list = []
    for sample_idx in sorted(window_embeddings.keys()):
        windows = np.array(window_embeddings[sample_idx])
        if aggregation == "max":
            agg_emb = np.max(windows, axis=0)
        else:
            agg_emb = np.mean(windows, axis=0)
        emb_list.append(agg_emb)
        if return_labels and samples is not None and sample_idx < len(samples):
            lbl_list.append(samples[sample_idx][1])

    emb = np.vstack(emb_list)
    if return_labels:
        labels = np.array(lbl_list, dtype=int)
        return emb, labels
    return emb


def generate_reference_embeddings(model, dataloader, device, cfg, limit: int = 40_000):
    use_windows, num_windows = _auc_window_settings(cfg)
    if use_windows:
        samples = getattr(dataloader.dataset, "samples", None)
        if not samples:
            return None
        ref_samples = [(path, 0) for path, *_ in samples][:limit]
        ref_ds = AUCValidationDataset(
            ref_samples,
            cfg["sample_rate"],
            cfg["max_audio_len_sec"],
            use_sliding_windows=True,
            num_windows=num_windows,
        )
        ref_loader = DataLoader(
            ref_ds,
            batch_size=cfg.get("auc_val_batch_size", 16),
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=True,
        )
        return aggregate_windowed_embeddings(
            model,
            ref_loader,
            device,
            aggregation=str(cfg.get("auc_val_aggregation", "mean")).lower(),
            return_labels=False,
        )

    model.eval()
    embeddings = []
    seen = 0
    with torch.no_grad():
        for audio, _, _ in dataloader:
            if seen >= limit:
                break
            audio = audio.to(device)
            z = model.module.get_embeddings(audio) if hasattr(model, "module") else model.get_embeddings(audio)
            embeddings.append(z.cpu().numpy())
            seen += len(audio)
    if not embeddings:
        return None
    return np.vstack(embeddings)[:limit]


def validate_auc(model, dataloader, reference_embeddings, device, cfg, rank: int = 0):
    if dataloader is None or reference_embeddings is None or len(reference_embeddings) == 0:
        return 0.0

    use_windows, _ = _auc_window_settings(cfg)
    if use_windows:
        q_emb, q_lbl = aggregate_windowed_embeddings(
            model,
            dataloader,
            device,
            aggregation=str(cfg.get("auc_val_aggregation", "mean")).lower(),
            return_labels=True,
        )
        if q_emb is None or q_lbl is None:
            return 0.0
    else:
        model.eval()
        query_embeddings, query_labels = [], []
        with torch.no_grad():
            for audio, labels in dataloader:
                audio = audio.to(device)
                z = model.module.get_embeddings(audio) if hasattr(model, "module") else model.get_embeddings(audio)
                query_embeddings.append(z.cpu().numpy())
                query_labels.append(labels.numpy())

        if not query_embeddings:
            return 0.0

        q_emb = np.vstack(query_embeddings)
        q_lbl = np.concatenate(query_labels)
    if len(np.unique(q_lbl)) < 2:
        return 0.5

    n_neighbors = min(cfg.get("auc_val_n_neighbors", 2), len(reference_embeddings))
    n_neighbors = max(1, n_neighbors)

    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(reference_embeddings)
    q_scaled = scaler.transform(q_emb)

    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(ref_scaled)
    distances, _ = nn_model.kneighbors(q_scaled)
    scores = distances[:, 0]

    try:
        auc = roc_auc_score(q_lbl, scores)
    except ValueError:
        return 0.5

    return auc


def build_datasets(cfg: dict, rank: int = 0):
    target = cfg["target_machine"].lower()
    dcase = DCASE_Dataset()
    all_samples = dcase.discover_train()
    target_test_samples_raw = discover_target_test_samples(target, dcase.data_path)

    inv_domain_map = {v: k for k, v in dcase.domain_map.items()}

    target_domain_strings = sorted({
        inv_domain_map[dom_id]
        for path, _, dom_id in all_samples
        if _path_has_machine_segment(path, target) and dom_id in inv_domain_map
    })

    domain_str_to_new_id = {s: i + 1 for i, s in enumerate(target_domain_strings)}
    domain_label_to_name = {0: "other"}
    domain_label_to_name.update({v: k for k, v in domain_str_to_new_id.items()})

    target_test_samples = []
    for path, machine_label, domain_str in target_test_samples_raw:
        new_dom = domain_str_to_new_id.get(domain_str, 0)
        target_test_samples.append((path, machine_label, new_dom))

    if rank == 0:
        logger.info(f"Target machine  : '{target}'")
        logger.info(f"Target domains  : {len(target_domain_strings)}")
        for new_id, name in sorted(domain_label_to_name.items()):
            logger.info(f"  label {new_id:3d} → {name}")

    target_samples, other_samples, aug_samples = [], [], []

    for path, class_id, dom_id in all_samples:
        filename = os.path.basename(path)
        is_target = _path_has_machine_segment(path, target)

        if is_target:
            dom_str = inv_domain_map.get(dom_id, "")
            new_dom = domain_str_to_new_id.get(dom_str, 0)
            machine_label = 1
        else:
            new_dom = 0
            machine_label = 0

        entry = (path, machine_label, new_dom)

        if dcase.detect_augmentation(filename) != "original":
            aug_samples.append(entry)
        elif is_target:
            target_samples.append(entry)
        else:
            other_samples.append(entry)

    oversample = cfg.get("target_oversample", 10)
    train_samples = target_samples * oversample + other_samples + aug_samples
    val_samples = target_test_samples

    if rank == 0:
        logger.info(
            f"Train  — target: {len(target_samples)} x{oversample}, "
            f"other: {len(other_samples)}, aug: {len(aug_samples)}"
        )
        logger.info(
            f"Train crops/sample: {cfg.get('train_crops_per_sample', 1)} "
            f"(effective train size: {len(train_samples) * cfg.get('train_crops_per_sample', 1)})"
        )
        logger.info(f"Val    — target test: {len(target_test_samples)}")

    preload_enabled = bool(cfg.get("preload_audio_to_ram", False))
    preload_train_only = bool(cfg.get("preload_train_only", True))
    preload_max_ram_gb = cfg.get("preload_max_ram_gb", None)
    preload_max_ram_bytes = None
    if preload_max_ram_gb is not None:
        preload_max_ram_bytes = int(float(preload_max_ram_gb) * (1024 ** 3))

    if rank == 0 and preload_enabled:
        cap_str = (
            f"{float(preload_max_ram_gb):.2f} GiB"
            if preload_max_ram_gb is not None
            else "unbounded"
        )
        logger.info(
            f"RAM preload enabled (train_only={preload_train_only}, cap={cap_str})."
        )

    window_sec = cfg["max_audio_len_sec"]
    train_ds = DCASEAudioDataset(
        train_samples,
        cfg["sample_rate"],
        window_sec,
        "train",
        crops_per_sample=cfg.get("train_crops_per_sample", 1),
        preload_to_ram=preload_enabled,
        preload_max_ram_bytes=preload_max_ram_bytes,
        preload_num_workers=cfg.get("preload_num_workers", 1),
    )
    preload_eval = preload_enabled and not preload_train_only
    val_ds = DCASEAudioDataset(
        val_samples,
        cfg["sample_rate"],
        window_sec,
        "val",
        preload_to_ram=preload_eval,
        preload_max_ram_bytes=preload_max_ram_bytes,
        preload_num_workers=cfg.get("preload_num_workers", 1),
    )
    reference_ds = DCASEAudioDataset(
        target_samples,
        cfg["sample_rate"],
        window_sec,
        "val",
        preload_to_ram=preload_eval,
        preload_max_ram_bytes=preload_max_ram_bytes,
        preload_num_workers=cfg.get("preload_num_workers", 1),
    )
    auc_val_ds = AUCValidationDataset(
        discover_target_test_auc_samples(target, dcase.data_path),
        cfg["sample_rate"],
        window_sec,
        use_sliding_windows=cfg.get("auc_val_sliding_window", False),
        num_windows=cfg.get("auc_val_num_windows", 5),
    )

    tsne_raw = target_test_samples + target_samples + other_samples[: len(target_test_samples) + len(target_samples)]
    tsne_ds = DCASEAudioDataset(
        tsne_raw,
        cfg["sample_rate"],
        window_sec,
        "val",
        preload_to_ram=preload_eval,
        preload_max_ram_bytes=preload_max_ram_bytes,
        preload_num_workers=cfg.get("preload_num_workers", 1),
    )

    return train_ds, val_ds, tsne_ds, reference_ds, auc_val_ds, domain_label_to_name


if __name__ == "__main__":
    data_root = r"C:\Users\emilr\Documents\GitHub\AAU_P8\data"

    print("=" * 100)
    print("DCASE Data Overview")
    print("=" * 100)
    print(f"Data root: {data_root}")

    ds = DCASE_Dataset(debug=False)
    ds.data_path = data_root

    # Use a local augmented directory if present; otherwise keep the default path.
    local_aug = os.path.join(os.path.dirname(data_root), "augmented2")
    if os.path.isdir(local_aug):
        ds.aug_base_dir = local_aug

    print(f"Augmented root: {ds.aug_base_dir}")
    print(f"Machines: {', '.join(ds.machines)}")

    print("\n" + "-" * 100)
    print("Per-machine folder overview")
    print("-" * 100)

    for machine in ds.machines:
        machine_root = next((d for d in os.listdir(ds.data_path) if d.lower() == machine.lower()), None)
        if machine_root is None:
            print(f"{machine:10s} | MISSING in data root")
            continue

        machine_path = os.path.join(ds.data_path, machine_root)
        train_dir = os.path.join(machine_path, "train")
        test_dir = os.path.join(machine_path, "test")
        supplemental_dir = os.path.join(machine_path, "supplemental")

        train_count = len([f for f in os.listdir(train_dir) if f.endswith(".wav")]) if os.path.isdir(train_dir) else 0
        test_count = len([f for f in os.listdir(test_dir) if f.endswith(".wav")]) if os.path.isdir(test_dir) else 0
        supplemental_count = (
            len([f for f in os.listdir(supplemental_dir) if f.endswith(".wav")])
            if os.path.isdir(supplemental_dir)
            else 0
        )

        print(
            f"{machine:10s} | train: {train_count:5d} | "
            f"test: {test_count:5d} | supplemental: {supplemental_count:5d}"
        )

    print("\n" + "-" * 100)
    print("discover_train() split/count overview")
    print("-" * 100)

    samples = ds.discover_train()
    print(f"Total discovered samples: {len(samples)}")

    original_count = 0
    augmented_count = 0
    machine_counts = {m.lower(): 0 for m in ds.machines}

    for path, class_id, domain_id in samples:
        filename = os.path.basename(path)
        aug_tag = ds.detect_augmentation(filename)
        if aug_tag == "original":
            original_count += 1
        else:
            augmented_count += 1

        for m in ds.machines:
            if m.lower() in path.lower():
                machine_counts[m.lower()] += 1
                break

    print(f"Original samples : {original_count}")
    print(f"Augmented samples: {augmented_count}")
    print(f"Class IDs        : {len(ds.class_map)}")
    print(f"Domain IDs       : {len(ds.domain_map)}")

    print("\nDomain ID mapping (domain -> id):")
    for domain_name, domain_idx in sorted(ds.domain_map.items(), key=lambda x: x[1]):
        print(f"  {domain_idx:4d} -> {domain_name}")

    print("\nSamples per machine in discover_train():")
    for m in ds.machines:
        print(f"  {m:10s}: {machine_counts[m.lower()]:6d}")

    print("\nDone.")