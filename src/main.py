import os
import sys
import importlib.util
from datetime import datetime
from multiprocessing import freeze_support
import yaml
import numpy as np

class PipelineOrchestrator:
    def __init__(self):
        with open("src/config.yaml") as f:
            self.config = yaml.safe_load(f)
        self.pipeline_dir = os.path.dirname(__file__)
        self.data = {}
        
        # Setup sys.path to include modules directory
        modules_dir = os.path.join(self.pipeline_dir, 'modules')
        if modules_dir not in sys.path:
            sys.path.insert(0, modules_dir)

    def _execute_module(self, stage_name, module_file):
        """Unified runner for Preprocessing, Embedding, and Classification."""
        if not module_file:
            print(f"No module specified for {stage_name}, skipping...")
            return None

        print(f"\n{'='*70}\nRUNNING: {stage_name.upper()}\n{'='*70}")
        
        # 1. Resolve Path & Load
        module_path = os.path.join(stage_name, module_file)
        full_path = os.path.join(self.pipeline_dir, 'modules', module_path)
        
        try:
            # Dynamic Import
            module_name = f"mod_{stage_name}_{module_file.replace('.', '_')}"
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 2. Execution - calling the main() function of the loaded module
            result = module.main(config=self.config)
            print(f"✓ {stage_name} completed")
            return result

        except Exception as e:
            print(f"Error in {stage_name} ({module_file}): {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_embeddings(self, embeddings_dict):
        # Look for models inside the embeddings dictionary
        model_list = self.config['embeddings']['modules']
        model_suffix = "_".join(model_list)
        
        path = os.path.join(self.config['paths'][self.config['environment']]['embeddings'], model_suffix)
        os.makedirs(path, exist_ok=True)

        for key, data in embeddings_dict.items():
            emb_array = data.get('embeddings')
            if emb_array is None or (isinstance(emb_array, np.ndarray) and emb_array.size == 0):
                continue
                
            np.save(os.path.join(path, f'{key}_embeddings.npy'), emb_array)
            
            for meta in ['file_paths', 'domains', 'labels']:
                if meta in data and data[meta]:
                    with open(os.path.join(path, f'{key}_{meta}.txt'), 'w') as f:
                        for line in data[meta]: 
                            f.write(f"{line}\n")
        print(f"✓ Saved embeddings to: {path}")


    def run(self):
        print(f"\nSTARTING PIPELINE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Stage 1: Preprocessing
        if self.config['preprocessing']['enabled']:
            self.data['preprocessed'] = self._execute_module('preprocessing', self.config['preprocessing']['module'])

        # Stage 2: Embeddings (with Fusion Logic)
        if self.config['embeddings']['enabled']:
            # UPDATED: Find models under config['embeddings']['models']
            models = self.config['embeddings']['modules'] 
            
            grouped_data = {}
            for model in models:
                # Generate filename dynamically: 'sslam_finetuned' -> 'sslam_finetuned.py'
                module_filename = f"{model}.py"
                
                # Execute the module found in modules/embeddings/
                emb_dict = self._execute_module('embeddings', module_filename)
                
                if not emb_dict:
                    print(f"Warning: No data returned from {module_filename}")
                    continue

                for machine, content in emb_dict.items():
                    if machine not in grouped_data:
                        grouped_data[machine] = {'embeddings': [], 'meta': content}
                    
                    grouped_data[machine]['embeddings'].append(content['embeddings'])

            # Flatten/Concatenate results
            final_embeddings = {}
            for machine, val in grouped_data.items():
                if len(val['embeddings']) > 1:
                    concatenated = np.concatenate(val['embeddings'], axis=1)
                else:
                    concatenated = val['embeddings'][0]

                final_embeddings[machine] = {
                    'embeddings': concatenated,
                    **{k: v for k, v in val['meta'].items() if k != 'embeddings'}
                }
            
            self.data['embeddings'] = final_embeddings
            self.save_embeddings(final_embeddings)

        # Stage 3: Classification
        if self.config['classification']['enabled']:
            module_filename = f"{self.config['classification']['module']}.py"
            self.data['classification'] = self._execute_module('classification', module_filename)

        return self.data

def main():
    results = PipelineOrchestrator().run()
    print("\nFinal Status:", {k: ("Success" if v else "Failed/Skipped") for k, v in results.items()})

if __name__ == "__main__":
    freeze_support()
    main()