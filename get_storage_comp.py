"""
Script to generate line chart showing storage and performance metrics
for different rotation representations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import timeit
import os
import json

class StoragePerformanceGenerator:
    def __init__(self):
        self.representations = ['euler', 'axis_angle', 'quaternion', 'rotation_matrix', 'exponential_map', '6d_continuous']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        self.config = self._load_config()
        
        # Create fig directory if it doesn't exist
        self.fig_dir = 'fig'
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)

    def _load_config(self):
        try:
            with open('rotation_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: 'rotation_config.json' not found. Using default values.")
            return {
                "storage_bytes": {
                    "euler": 24, "axis_angle": 24, "quaternion": 32, 
                    "rotation_matrix": 72, "exponential_map": 24, 
                    "6d_continuous": 48
                },
                "evaluation_params": {
                    "num_trials": 1000
                }
            }

    def storage_bytes(self):
        """Storage requirements in bytes (assuming float64 = 8 bytes per scalar)."""
        return self.config['storage_bytes']

    def generate_random_rotations(self, n=2, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return [R.random() for _ in range(n)]

    def to_representation(self, rot, rep_type):
        try:
            if rep_type == 'euler':
                return rot.as_euler('xyz', degrees=False)
            elif rep_type == 'axis_angle':
                return rot.as_rotvec()
            elif rep_type == 'quaternion':
                return rot.as_quat()
            elif rep_type == 'rotation_matrix':
                return rot.as_matrix().flatten()
            elif rep_type == 'exponential_map':
                return rot.as_rotvec()
            elif rep_type == '6d_continuous':
                mat = rot.as_matrix()
                return mat[:, :2].flatten()
            else:
                raise ValueError(f"Unknown representation: {rep_type}")
        except Exception:
            return None

    def from_representation(self, params, rep_type):
        try:
            if rep_type == 'euler':
                return R.from_euler('xyz', params)
            elif rep_type == 'axis_angle':
                return R.from_rotvec(params)
            elif rep_type == 'quaternion':
                # Normalize quaternion
                params = params / np.linalg.norm(params)
                return R.from_quat(params)
            elif rep_type == 'rotation_matrix':
                mat = params.reshape(3, 3)
                U, _, Vt = np.linalg.svd(mat)
                mat_clean = U @ Vt
                if np.linalg.det(mat_clean) < 0:
                    mat_clean[:, -1] *= -1
                return R.from_matrix(mat_clean)
            elif rep_type == 'exponential_map':
                return R.from_rotvec(params)
            elif rep_type == '6d_continuous':
                a1, a2 = params[:3], params[3:]
                if np.linalg.norm(a1) < 1e-8:
                    a1 = np.array([1, 0, 0])
                if np.linalg.norm(a2) < 1e-8:
                    a2 = np.array([0, 1, 0])
                b1 = a1 / np.linalg.norm(a1)
                dot_product = np.dot(b1, a2)
                if abs(abs(dot_product) - 1.0) < 1e-6:
                    if abs(b1[0]) < 0.9:
                        a2 = np.array([1, 0, 0])
                    else:
                        a2 = np.array([0, 1, 0])
                b2 = a2 - np.dot(b1, a2) * b1
                b2_norm = np.linalg.norm(b2)
                if b2_norm < 1e-8:
                    if abs(b1[0]) < 0.9:
                        temp = np.array([1, 0, 0])
                    else:
                        temp = np.array([0, 1, 0])
                    b2 = temp - np.dot(b1, temp) * b1
                    b2_norm = np.linalg.norm(b2)
                b2 = b2 / b2_norm
                b3 = np.cross(b1, b2)
                mat = np.column_stack([b1, b2, b3])
                if np.linalg.det(mat) < 0:
                    b3 = -b3
                    mat = np.column_stack([b1, b2, b3])
                return R.from_matrix(mat)
            else:
                raise ValueError(f"Unknown representation: {rep_type}")
        except Exception:
            return None

    def slerp_quat(self, q1, q2, t):
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        omega = np.arccos(dot)
        if np.abs(omega) < 1e-6:
            return q1
        sin_omega = np.sin(omega)
        if np.abs(sin_omega) < 1e-6:
            return q1
        return (np.sin((1 - t) * omega) / sin_omega) * q1 + (np.sin(t * omega) / sin_omega) * q2

    def compose(self, params1, params2, rep_type):
        rot1 = self.from_representation(params1, rep_type)
        rot2 = self.from_representation(params2, rep_type)
        if rot1 is None or rot2 is None:
            return None
        composed = rot1 * rot2
        return self.to_representation(composed, rep_type)

    def interpolate(self, params1, params2, t=0.5, rep_type='quaternion'):
        rot1 = self.from_representation(params1, rep_type)
        rot2 = self.from_representation(params2, rep_type)
        if rot1 is None or rot2 is None:
            return None
        if rep_type == 'quaternion':
            q1 = rot1.as_quat()
            q2 = rot2.as_quat()
            qi = self.slerp_quat(q1, q2, t)
            return qi
        elif rep_type in ['rotation_matrix', '6d_continuous']:
            q1 = rot1.as_quat()
            q2 = rot2.as_quat()
            qi = self.slerp_quat(q1, q2, t)
            rot_i = R.from_quat(qi)
            return self.to_representation(rot_i, rep_type)
        else:
            v1 = rot1.as_rotvec()
            v2 = rot2.as_rotvec()
            vi = (1 - t) * v1 + t * v2
            rot_i = R.from_rotvec(vi)
            return self.to_representation(rot_i, rep_type)

    def efficiency_analysis(self, rep_type, num_trials=None):
        if num_trials is None:
            num_trials = self.config['evaluation_params']['num_trials']
            
        rots = self.generate_random_rotations(2, seed=42)
        params1 = self.to_representation(rots[0], rep_type)
        params2 = self.to_representation(rots[1], rep_type)
        if params1 is None or params2 is None:
            return {'composition_time': float('inf'), 'interpolation_time': float('inf')}
        
        def compose_func():
            self.compose(params1, params2, rep_type)
        comp_time = timeit.timeit(compose_func, number=num_trials) / num_trials
        
        def interp_func():
            self.interpolate(params1, params2, 0.5, rep_type)
        interp_time = timeit.timeit(interp_func, number=num_trials) / num_trials
        
        return {'composition_time': comp_time, 'interpolation_time': interp_time}

    def collect_data(self):
        """Collect performance data for all representations."""
        results = []
        storage = self.storage_bytes()
        
        for rep in self.representations:
            row = {'Representation': rep.title().replace('_', ' ')}
            row['Storage (bytes)'] = storage[rep]
            efficiency = self.efficiency_analysis(rep)
            row['Composition Time (μs)'] = efficiency['composition_time'] * 1e6
            row['Interpolation Time (μs)'] = efficiency['interpolation_time'] * 1e6
            results.append(row)
            
        return pd.DataFrame(results)

    def plot_storage_performance(self, df):
        """Plot storage and performance metrics on dual-axis chart."""
        fig, ax1 = plt.subplots(figsize=(16, 12))
        
        x = range(len(df))
        representations = [name.replace('_', ' ').title() for name in self.representations]
        
        # Plot Storage (primary axis)
        line1 = ax1.plot(
            x, df['Storage (bytes)'], color=self.colors[0], linewidth=4,
            marker='o', markersize=12, label='Storage (bytes)', linestyle='-'
        )
        ax1.set_ylabel('Storage (bytes)', color=self.colors[0], fontsize=26, fontweight='regular')
        ax1.tick_params(axis='y', labelcolor=self.colors[0], labelsize=18)
        ax1.set_ylim(0, max(df['Storage (bytes)']) * 1.25)
        
        # Plot Timing (secondary axis, log scale)
        ax2 = ax1.twinx()
        line2 = ax2.plot(
            x, df['Composition Time (μs)'], color=self.colors[1], linewidth=4,
            marker='s', markersize=12, label='Composition Time (μs)', linestyle='--'
        )
        line3 = ax2.plot(
            x, df['Interpolation Time (μs)'], color=self.colors[2], linewidth=4,
            marker='^', markersize=12, label='Interpolation Time (μs)', linestyle='-.'
        )
        ax2.set_ylabel('Time (μs, log scale)', color='black', fontsize=26, fontweight='regular')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelsize=18)
        
        # X-axis
        ax1.set_xticks(x)
        ax1.set_xticklabels(representations, rotation=40, ha='right', fontsize=16, fontweight='regular')
        ax1.set_xlabel('Rotation Representation', fontsize=24, fontweight='regular', labelpad=2)
        
        # Title
        ax1.set_title(
            'Rotation Representation Performance Overview',
            fontsize=26, fontweight='bold', pad=5
        )
        
        # Grid
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2, labels1 + labels2,
            loc='lower center', bbox_to_anchor=(0.5, -0.31), ncol=3, fontsize=18,
            frameon=True, fancybox=True, shadow=False
        )
        
        # Annotate Data Points
        for i, (_, row) in enumerate(df.iterrows()):
            storage_offset = 0.05
            if i == 0:
                storage_offset = 0.01
            if i == 1:
                storage_offset = 0.04
            if i == 2:
                storage_offset = 0.03
            if i == 3:
                storage_offset = -0.02
            if i == 4:
                storage_offset = 0.02
            if i == 5:
                storage_offset = 0.02
            
            # Storage annotation
            ax1.text(
                i, row['Storage (bytes)'] + max(df['Storage (bytes)']) * storage_offset,
                f'{int(row["Storage (bytes)"])}',
                ha='center', va='bottom', fontsize=10, fontweight='regular',
                color=self.colors[0],
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='gray', alpha=0.7, linewidth=0.6)
            )
            
            # Composition Time annotation
            comp_offset = 1.03
            if i == 0:
                comp_offset = 1.02
            if i == 1:
                comp_offset = 1.04
            if i == 2:
                comp_offset = 1.01
            if i == 3:
                comp_offset = 1.04
            if i == 4:
                comp_offset = 1.02
            if i == 5:
                comp_offset = 1.01
            
            ax2.text(
                i, row['Composition Time (μs)'] * comp_offset,
                f'{row["Composition Time (μs)"]:.1f}',
                ha='center', va='center', fontsize=10, fontweight='regular',
                color=self.colors[1],
                bbox=dict(boxstyle='round,pad=0.95', facecolor='white', edgecolor='gray', alpha=0.7, linewidth=0.6)
            )
            
            # Interpolation Time annotation
            interp_offset = 1.03
            if i == 0:
                interp_offset = 1.02
            if i == 1:
                interp_offset = 1.04
            if i == 2:
                interp_offset = 1.01
            if i == 3:
                interp_offset = 1.04
            if i == 4:
                interp_offset = 1.02
            if i == 5:
                interp_offset = 1.05
            
            ax2.text(
                i, row['Interpolation Time (μs)'] * interp_offset,
                f'{row["Interpolation Time (μs)"]:.1f}',
                ha='center', va='center', fontsize=10, fontweight='regular',
                color=self.colors[2],
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='gray', alpha=0.7, linewidth=0.6)
            )
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18, top=0.82, right=0.88, left=0.12)
        plt.savefig(f'{self.fig_dir}/storage_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """Run analysis and generate storage performance chart."""
        print("Collecting performance data...")
        df = self.collect_data()
        
        print("Generating storage performance chart...")
        self.plot_storage_performance(df)
        
        print(f"\nChart saved to '{self.fig_dir}/storage_performance.png'")
        
        # Display the data
        print("\nPerformance Data:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    generator = StoragePerformanceGenerator()
    generator.run_analysis()