from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np
import warnings
import matplotlib.pyplot as plt
from wfl.configset import ConfigSet, OutputSpec
from wfl.descriptors import quippy
from wfl.select.by_descriptor import CUR_conf_global, prep_descs_and_exclude, write_selected_and_clean
from wfl.select.flat_histogram import biased_select_conf
from wfl.map import map as wfl_map

class ConfigurationSelector:
    def __init__(self, inputs: Union[ConfigSet, List[Any]], output_prefix: str, seed: int = 10):
        """
        Initializes the ConfigurationSelector.

        Args:
            inputs: Input configurations (ConfigSet or list of Atoms).
            output_prefix: Prefix for output files.
            seed: Random seed for reproducibility.
        """
        self.inputs = ConfigSet(inputs)
        self.output_prefix = output_prefix
        self.rng = np.random.default_rng(seed)
        self.global_desc = None
        self.desc_key = "SOAP"

    def _get_average_desc(self, at: Any, descriptor_key: str) -> Any:
        at_desc = at.arrays.pop(descriptor_key)
        at_desc = np.sum(at_desc, axis=0)
        at_desc /= np.linalg.norm(at_desc)
        at.info[descriptor_key] = at_desc
        return at

    def calculate_global_descriptors(self, descs: List[str], key: str = "SOAP", write_xyz: bool = False) -> ConfigSet:
        """
        Calculates global descriptors (e.g., SOAP) for the input configurations.

        Args:
            descs: List of descriptor strings (e.g., used by quippy).
            key: Key to store the descriptors in atoms.info/arrays.
            write_xyz: Whether to write the descriptors to an XYZ file.

        Returns:
            ConfigSet containing the configurations with calculated global descriptors.
        """
        self.desc_key = key
        
        # Local Descriptors
        mols_desc_local = quippy.calculate(
            self.inputs,
            OutputSpec(),
            descs=descs,
            key=self.desc_key,
            per_atom=True
        )

        # Global Average Descriptors
        if write_xyz:
            out = OutputSpec(f'{self.output_prefix}_global_desc.xyz')
        else:
            out = OutputSpec()

        self.global_desc = wfl_map(
            inputs=mols_desc_local,
            outputs=out,
            map_func=self._get_average_desc,
            args=[self.desc_key]
        )
        return self.global_desc

    def _ensure_descriptors_calculated(self):
        if self.global_desc is None:
            raise RuntimeError("Global descriptors have not been calculated. Call 'calculate_global_descriptors' first.")

    def greedy_fps_with_tracking(self, inputs: ConfigSet, outputs: OutputSpec, num: int, 
                                 at_descs: Optional[np.ndarray] = None, 
                                 at_descs_info_key: Optional[str] = None,
                                 keep_descriptor_info: bool = True, 
                                 exclude_list: Optional[List[int]] = None,
                                 prev_selected_descs: Optional[Union[List[Any], np.ndarray]] = None, 
                                 O_N_sq: bool = False, 
                                 verbose: bool = False) -> Tuple[ConfigSet, List[float]]:
        """
        Full-feature Farthest Point Sampling (FPS) that captures distances while supporting all original WFL variables.
        Refactored as a method of the class.

        Args:
            inputs: Input configurations.
            outputs: Output specification.
            num: Number of configurations to select.
            at_descs: Pre-calculated descriptors (optional).
            at_descs_info_key: Key in atoms.info where descriptors are stored.
            keep_descriptor_info: Whether to keep descriptor info in selected configs.
            exclude_list: List of indices to exclude from selection.
            prev_selected_descs:  Descriptors of previously selected configurations (to seed FPS).
            O_N_sq: Use O(N^2) algorithm (default False).
            verbose: Verbosity flag.

        Returns:
            Tuple containing the selected configurations (ConfigSet) and a list of minimum distances.
        """
        if outputs.all_written():
            return outputs.to_ConfigSet(), []

        if prev_selected_descs is not None and not isinstance(prev_selected_descs, np.ndarray):
            prev_selected_descs = np.asarray(prev_selected_descs)
            
        # Default key if not provided
        if at_descs_info_key is None:
            at_descs_info_key = self.desc_key

        # Use the original WFL helper to handle exclusion and descriptor extraction
        at_descs, exclude_ind_list = prep_descs_and_exclude(inputs, at_descs, at_descs_info_key, exclude_list)
        
        n_avail = at_descs.shape[1] - len(exclude_ind_list)
        if n_avail < num:
            raise RuntimeError(f'Asked for {num} configs but only {n_avail} are available')

        min_distances = [] 
        max_similarity = 2.0 

        # --- BLOCK: O(N^2) Path ---
        if O_N_sq:
            if prev_selected_descs is not None and len(prev_selected_descs) > 0:
                lhs = np.vstack([prev_selected_descs, at_descs.T])
                prev_selected = list(range(len(prev_selected_descs)))
            else:
                lhs = at_descs.T
                prev_selected = []
            
            similarities = np.matmul(lhs, at_descs)
            similarities[:, exclude_ind_list] = max_similarity + 1.0

            if len(prev_selected) == 0:
                p = np.ones(similarities.shape[1]); p[exclude_ind_list] = 0.0; p /= np.sum(p)
                selected_indices = [self.rng.choice(range(similarities.shape[1]), p=p)]
                similarities[:, selected_indices[-1]] = max_similarity + 1.0
                min_distances.append(0.0)
            else:
                selected_indices = []

            while len(selected_indices) < num:
                sims_to_nearest = np.max(similarities[prev_selected + [s + len(prev_selected) for s in selected_indices]], axis=0)
                farthest_available = np.argmin(sims_to_nearest)
                
                # Distance capture
                val = 1.0 - sims_to_nearest[farthest_available]
                min_distances.append(np.sqrt(2.0 * val) if val > 0 else 0.0)

                selected_indices.append(farthest_available)
                similarities[:, selected_indices[-1]] = max_similarity + 1.0

        # --- BLOCK: Memory-Efficient Path ---
        else:
            if prev_selected_descs is not None and len(prev_selected_descs) > 0:
                selected_indices = []
                similarities_arr = prev_selected_descs @ at_descs
            else:
                p = np.ones(at_descs.shape[1]); p[exclude_ind_list] = 0.0; p /= np.sum(p)
                selected_indices = [self.rng.choice(range(at_descs.shape[1]), p=p)]
                similarities_arr = np.asarray([at_descs[:, selected_indices[-1]].T @ at_descs])
                similarities_arr[:, selected_indices[-1]] = max_similarity
                min_distances.append(0.0)

            similarities_arr[:, exclude_ind_list] = max_similarity

            while len(selected_indices) < num:
                sims_to_nearest = np.max(similarities_arr, axis=0)
                farthest_available = np.argmin(sims_to_nearest)
                
                # Distance capture
                val = 1.0 - sims_to_nearest[farthest_available]
                min_distances.append(np.sqrt(2.0 * val) if val > 0 else 0.0)

                selected_indices.append(farthest_available)
                similarity_row = at_descs[:, selected_indices[-1]].T @ at_descs
                similarity_row[exclude_ind_list] = max_similarity
                similarities_arr = np.vstack([similarities_arr, similarity_row])

        write_selected_and_clean(inputs, outputs, selected_indices, at_descs_info_key, keep_descriptor_info)

        return outputs.to_ConfigSet(), min_distances


    def find_robust_elbow(self, distances: List[float], start_idx: int = 20) -> int:
        """
        Identifies the 'Elbow' or 'Knee' of the distance curve to determine the optimal number of samples.

        Args:
            distances: List of distances from FPS.
            start_idx: Index to start looking for the elbow (to avoid initial sharp drop).

        Returns:
            The optimal number of samples (index + 1).
        """
        # 1. Convert to numpy array for vector math
        y_full = np.array(distances)
        
        # 2. Safety Check
        if len(y_full) <= start_idx + 2:
            return len(y_full)

        # 3. Truncate the "Cliff"
        y_subset = y_full[start_idx:]
        x_subset = np.arange(start_idx, len(y_full))
        
        # 4. Normalize Data
        x_norm = (x_subset - x_subset.min()) / (x_subset.max() - x_subset.min())
        y_norm = (y_subset - y_subset.min()) / (y_subset.max() - y_subset.min())
        
        # 5. Kneedle Algorithm
        line_vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
        vec_from_start = np.stack([x_norm - x_norm[0], y_norm - y_norm[0]], axis=1)
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return len(distances)

        vec_cross = np.cross(line_vec, vec_from_start)
        dist_to_line = np.abs(vec_cross) / line_len
        
        # 6. Find the index with the maximum distance
        elbow_idx_local = np.argmax(dist_to_line)
        
        # 7. Convert back to the original index
        n_optimal = start_idx + elbow_idx_local
        
        return n_optimal + 1


    def select_by_cur(self, num: int, inputs: Optional[ConfigSet] = None, **kwargs) -> Any:
        """
        Selects configurations using CUR decomposition.

        Args:
            num: Number of configurations to select.
            inputs: Input configurations (defaults to self.global_desc).
            **kwargs: Additional arguments passed to `wfl.select.by_descriptor.CUR_conf_global`.
                      Supported args: at_descs, at_descs_info_key, kernel_exp, stochastic, rng,
                      keep_descriptor_info, exclude_list, center, leverage_score_key.

        Returns:
            Selected configurations.
        """
        self._ensure_descriptors_calculated()
        inputs = inputs or self.global_desc
        return CUR_conf_global(
            inputs,
            OutputSpec(),
            at_descs_info_key=self.desc_key,
            num=num,
            rng=self.rng,
            **kwargs
        )


    def select_by_histogram(self, num: int, info_field: str, inputs: Optional[ConfigSet] = None, **kwargs) -> Any:
        """
        Selects configurations based on a flat histogram of a specific info field.

        Args:
            num: Number of configurations to select.
            info_field: The key in atoms.info to use for the histogram.
            inputs: Input configurations (defaults to self.global_desc).
            **kwargs: Additional arguments passed to `wfl.select.flat_histogram.biased_select_conf`.
                      Supported args: kT, bins, by_bin, replace, verbose.

        Returns:
            Selected configurations.
        """
        self._ensure_descriptors_calculated()
        inputs = inputs or self.global_desc
        return biased_select_conf(
            inputs,
            OutputSpec(),
            num=num,
            info_field=info_field,
            rng=self.rng,
            **kwargs
        )

    def select_optimal_n(self, max_n: int = 1000) -> Tuple[int, List[float]]:
        """
        Determines the optimal number of configurations using FPS and the Elbow method.

        Args:
            max_n: Maximum number of configurations to check.

        Returns:
            Tuple containing the optimal number of configurations and the list of distances.
        """
        self._ensure_descriptors_calculated()
        print(f"Running FPS to find optimal N (max {max_n})...")
        fps_out, distances = self.greedy_fps_with_tracking(
            inputs=self.global_desc,
            outputs=OutputSpec(),
            num=max_n,
            at_descs_info_key=self.desc_key
        )
        
        n_optimal = self.find_robust_elbow(distances)

        # Round up to nearest multiple of 20
        n_optimal = round((1.1 * n_optimal) / 20) * 20
        
        print(f"Optimal N determined: {n_optimal}")

        return n_optimal, distances


    def get_descriptors_from_configset(self, confs: Any) -> np.ndarray:
        """Helper to extract descriptors from a ConfigSet for passing to FPS"""
        descs = []
        for at in confs:
            if self.desc_key in at.info:
                descs.append(at.info[self.desc_key])
        return np.array(descs)


    def select_final(self, n_optimal: int, info_field: str, **kwargs) -> Tuple[Any, Any, Any]:
        """
        Performs the final selection using a mix of CUR, Histogram, and FPS methods.
        
        Args:
            n_optimal: Total number of configurations to select.
            info_field: Info field for histogram selection.
            **kwargs: Additional arguments passed to underlying selection methods.
                      Separated into `cur_kwargs` and `hist_kwargs`.

        Returns:
            Tuple of selected configurations (cur_selected, hist_selected, fps_selected).
        """
        self._ensure_descriptors_calculated()
        print(f"Performing final selection for N={n_optimal}...")
        n_cur = int(0.4 * n_optimal)
        n_hist = int(0.2 * n_optimal)
        n_fps = n_optimal - n_cur - n_hist # Remainder to FPS

        # Distribute kwargs
        cur_keys = ['at_descs', 'at_descs_info_key', 'kernel_exp', 'stochastic', 'rng', 
                    'keep_descriptor_info', 'exclude_list', 'center', 'leverage_score_key']
        hist_keys = ['kT', 'bins', 'by_bin', 'replace', 'verbose'] # info_field, num, rng are handled explicitly or via self

        cur_kwargs = {k: v for k, v in kwargs.items() if k in cur_keys}
        hist_kwargs = {k: v for k, v in kwargs.items() if k in hist_keys}

        # 1. CUR
        print(f"Selecting {n_cur} via CUR...")
        cur_selected = self.select_by_cur(n_cur, **cur_kwargs)

        # 2. Hist
        print(f"Selecting {n_hist} via Histogram...")
        hist_selected = self.select_by_histogram(n_hist, info_field=info_field, **hist_kwargs)

        # Combine selected descriptors for FPS exclusion/initialization
        prev_descs = []
        
        # Cur
        for at in cur_selected:
            prev_descs.append(at.info[self.desc_key])
        # Hist
        for at in hist_selected:
             prev_descs.append(at.info[self.desc_key])
        
        prev_descs = np.array(prev_descs)

        # 3. FPS
        print(f"Selecting {n_fps} via FPS (with prior knowledge)...")
        fps_selected, _ = self.greedy_fps_with_tracking(
            inputs=self.global_desc,
            outputs=OutputSpec(),
            num=n_fps,
            at_descs_info_key=self.desc_key,
            prev_selected_descs=prev_descs
        )
        OutputSpec(f'{self.output_prefix}_final_selection.xyz').write(ConfigSet([cur_selected, hist_selected, fps_selected]))
        return cur_selected, hist_selected, fps_selected

    def plot_elbow(self, distances, n_optimal):
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(distances) + 1), distances, label='FPS Distance')
        plt.axvline(x=n_optimal, color='r', linestyle='--', label=f'Optimal N = {n_optimal}')
        plt.xlabel('Number of Configurations')
        plt.ylabel('Distance')
        plt.title('Elbow Test for Optimal Selection Number')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_prefix}_elbow_plot.png')
        plt.close()


    def run_two_stage_selection(self, n_optimal: Optional[int] = None, n_max: int = 1000, info_field: Optional[str] = None, **kwargs) -> Tuple[Any, Any, Any]:
        """
        Orchestrates the selection.
        If n_optimal is provided, skips stage 1 unless descriptors are needed.
        Requires calculate_global_descriptors to have been called.

        Args:
            n_optimal: Optimal number of configurations (optional).
            n_max: Maximum number of configurations for optimal N search.
            info_field: Info field for histogram selection (required).
            **kwargs: Additional arguments passed to `select_final`.

        Returns:
            Tuple of selected configurations.
        """
        self._ensure_descriptors_calculated()
        
        if n_optimal is None:
            n_optimal, distances = self.select_optimal_n(max_n=n_max)
            #self.plot_elbow(distances, n_optimal)
        
        if info_field is None:
            raise ValueError("info_field is required for the histogram selection part of the two-stage strategy.")
            
        return self.select_final(n_optimal, info_field=info_field, **kwargs)