import os
import re
import librosa
import numpy as np
import scipy.io.wavfile as wav

class DCASE_Dataset:
    def __init__(self, debug=False):
        self.debug = debug
        self.local_env = False  # Set to False when running on cluster with actual paths
        
        if self.local_env:
            self.data_path = r"C:\Users\emilr\Documents\GitHub\Semester8\DCASE\data"
            self.aug_base_dir = r"C:\Users\emilr\Documents\GitHub\Semester8\DCASE\augmentation\augmented"
        else:
            self.data_path = "/ceph/project/P8_DCASE/data_2026"
            self.aug_base_dir = "/ceph/project/P8_DCASE/data_2026_AUG"

        #self.machines = ["fan", "valve", "slider", "ToyCar", "ToyTrain", "gearbox", "bearing"]
        self.machines = ["fan", "valveEmu", "sliderEmu", "ToyCarEmu", "ToyCar", "gearboxEmu", "bearingEmu"]
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