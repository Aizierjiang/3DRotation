import numpy as np
from scipy.spatial.transform import Rotation as R
import timeit
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

class RotationEvaluator:
    def __init__(self):
        self.representations = ['euler', 'axis_angle', 'quaternion', 'rotation_matrix', 'exponential_map', '6d_continuous']
        self.config = self._load_config()
        
    def _load_config(self):
        try:
            with open('rotation_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: 'rotation_config.json' not found. Using default values.")
            return {
                "storage_bytes": {
                    "euler": 3 * 8,  # 3 params
                    "axis_angle": 3 * 8, 
                    "quaternion": 4 * 8, 
                    "rotation_matrix": 9 * 8, 
                    "exponential_map": 3 * 8, 
                    "6d_continuous": 6 * 8, 
                    "matrix_fisher": 9 * 8
                },
                "evaluation_params": {
                    "num_trials": 1000,
                    "interpolation_points": 100,
                    "derivative_samples": 50,
                    "batch_size": 100
                }
            }
        
    def storage_bytes(self):
        """Storage requirements in bytes (assuming float64)."""
        return self.config['storage_bytes']

    def generate_random_rotations(self, n=2, seed=None):
        """Generate n random rotations as scipy Rotation objects."""
        if seed is not None:
            np.random.seed(seed)
        return [R.random() for _ in range(n)]

    def to_representation(self, rot, rep_type):
        """Convert scipy Rotation to specific representation."""
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
        except Exception as e:
            print(f"Error converting to {rep_type}: {e}")
            return None

    def from_representation(self, params, rep_type):
        """Convert params to scipy Rotation object."""
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
                # Use SVD but measure the deviation
                U, s, Vt = np.linalg.svd(mat)
                mat_clean = U @ Vt
                if np.linalg.det(mat_clean) < 0:
                    mat_clean[:, -1] *= -1
                return R.from_matrix(mat_clean)
            elif rep_type == 'exponential_map':
                return R.from_rotvec(params)
            elif rep_type == '6d_continuous':
                a1, a2 = params[:3], params[3:]
                
                # Ensure non-zero vectors
                if np.linalg.norm(a1) < 1e-8:
                    a1 = np.array([1, 0, 0])
                if np.linalg.norm(a2) < 1e-8:
                    a2 = np.array([0, 1, 0])
                
                # Gram-Schmidt orthogonalization
                b1 = a1 / np.linalg.norm(a1)
                
                # Make sure a2 is not parallel to b1
                dot_product = np.dot(b1, a2)
                if abs(abs(dot_product) - 1.0) < 1e-6:  # Nearly parallel
                    # Find a perpendicular vector
                    if abs(b1[0]) < 0.9:
                        a2 = np.array([1, 0, 0])
                    else:
                        a2 = np.array([0, 1, 0])
                
                b2 = a2 - np.dot(b1, a2) * b1
                b2_norm = np.linalg.norm(b2)
                if b2_norm < 1e-8:
                    # Find another perpendicular vector
                    if abs(b1[0]) < 0.9:
                        temp = np.array([1, 0, 0])
                    else:
                        temp = np.array([0, 1, 0])
                    b2 = temp - np.dot(b1, temp) * b1
                    b2_norm = np.linalg.norm(b2)
                
                b2 = b2 / b2_norm
                b3 = np.cross(b1, b2)

                # Ensure right-handed coordinate system
                mat = np.column_stack([b1, b2, b3])
                if np.linalg.det(mat) < 0:
                    b3 = -b3
                    mat = np.column_stack([b1, b2, b3])
                
                return R.from_matrix(mat)
            else:
                raise ValueError(f"Unknown representation: {rep_type}")
        except Exception as e:
            print(f"Error converting from {rep_type}: {e}")
            return None

    def slerp_quat(self, q1, q2, t):
        """Spherical linear interpolation for quaternions."""
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        omega = np.arccos(dot)
        if np.abs(omega) < 1e-6:
            return q1  # Very close rotations
        sin_omega = np.sin(omega)
        if np.abs(sin_omega) < 1e-6:
            return q1  # Avoid division by zero
        return (np.sin((1-t)*omega)/sin_omega) * q1 + (np.sin(t*omega)/sin_omega) * q2

    def compose(self, params1, params2, rep_type):
        """Compose two rotations in the given representation."""
        rot1 = self.from_representation(params1, rep_type)
        rot2 = self.from_representation(params2, rep_type)
        if rot1 is None or rot2 is None:
            return None
        composed = rot1 * rot2
        return self.to_representation(composed, rep_type)

    def interpolate_native(self, params1, params2, t, rep_type):
        """
        Interpolate in native parameter space for each representation.
        """
        if rep_type == 'euler':
            # Linear interpolation in Euler angle space (will show gimbal lock issues)
            return (1-t) * params1 + t * params2
            
        elif rep_type == 'axis_angle' or rep_type == 'exponential_map':
            # Linear interpolation in rotation vector space
            return (1-t) * params1 + t * params2
            
        elif rep_type == 'quaternion':
            # SLERP for quaternions
            return self.slerp_quat(params1, params2, t)
            
        elif rep_type == 'rotation_matrix':
            # Matrix interpolation via geodesic
            mat1 = params1.reshape(3, 3)
            mat2 = params2.reshape(3, 3)
            rot1 = R.from_matrix(mat1)
            rot2 = R.from_matrix(mat2)
            
            # Geodesic interpolation
            q1 = rot1.as_quat()
            q2 = rot2.as_quat()
            qi = self.slerp_quat(q1, q2, t)
            rot_i = R.from_quat(qi)
            return rot_i.as_matrix().flatten()
            
        elif rep_type == '6d_continuous':
            # Linear interpolation in 6D space then project
            interp_6d = (1-t) * params1 + t * params2
            # The projection happens in from_representation
            rot_i = self.from_representation(interp_6d, rep_type)
            if rot_i is None:
                return None
            return self.to_representation(rot_i, rep_type)
        
        return None

    def interpolation_path_analysis(self, rep_type, num_points=None):
        """Analyze interpolation path properties."""
        if num_points is None:
            num_points = self.config['evaluation_params']['interpolation_points']
            
        rot1, rot2 = self.generate_random_rotations(2, seed=42)
        params1 = self.to_representation(rot1, rep_type)
        params2 = self.to_representation(rot2, rep_type)
        
        if params1 is None or params2 is None:
            return {'path_length': float('inf'), 'geodesic_error': float('inf')}
        
        # Generate interpolation path using NATIVE interpolation
        t_values = np.linspace(0, 1, num_points)
        path_rotations = []
        
        for t in t_values:
            interp_params = self.interpolate_native(params1, params2, t, rep_type)
            if interp_params is None:
                continue
            interp_rot = self.from_representation(interp_params, rep_type)
            if interp_rot is not None:
                path_rotations.append(interp_rot)
        
        if len(path_rotations) < 2:
            return {'path_length': float('inf'), 'geodesic_error': float('inf')}
        
        # Calculate path length
        path_length = 0
        for i in range(len(path_rotations) - 1):
            diff_rot = R.inv(path_rotations[i]) * path_rotations[i + 1]
            path_length += np.linalg.norm(diff_rot.as_rotvec())
        
        # Compare to shortest path
        geodesic_rot = R.inv(rot1) * rot2
        geodesic_length = np.linalg.norm(geodesic_rot.as_rotvec())
        geodesic_error = abs(path_length - geodesic_length) / (geodesic_length + 1e-8)
        
        return {'path_length': path_length, 'geodesic_error': geodesic_error}

    def derivative_continuity_test(self, rep_type, num_samples=None):
        """Test derivative continuity during interpolation."""
        if num_samples is None:
            num_samples = self.config['evaluation_params']['derivative_samples']
            
        rot1, rot2 = self.generate_random_rotations(2, seed=42)
        params1 = self.to_representation(rot1, rep_type)
        params2 = self.to_representation(rot2, rep_type)
        
        if params1 is None or params2 is None:
            return float('inf')
        
        # Sample points for derivative estimation
        t_values = np.linspace(0.01, 0.99, num_samples)
        derivatives = []
        
        dt = 0.001
        for t in t_values:
            # Use NATIVE interpolation
            interp1 = self.interpolate_native(params1, params2, t - dt/2, rep_type)
            interp2 = self.interpolate_native(params1, params2, t + dt/2, rep_type)
            
            if interp1 is None or interp2 is None:
                continue
                
            rot1_t = self.from_representation(interp1, rep_type)
            rot2_t = self.from_representation(interp2, rep_type)
            
            if rot1_t is None or rot2_t is None:
                continue
            
            # Angular velocity approximation
            diff_rot = R.inv(rot1_t) * rot2_t
            angular_vel = np.linalg.norm(diff_rot.as_rotvec()) / dt
            derivatives.append(angular_vel)
        
        if len(derivatives) < 2:
            return float('inf')
        
        # Measure derivative smoothness (lower variance = smoother)
        return np.std(derivatives) / (np.mean(derivatives) + 1e-8)

    def parallel_efficiency_test(self, rep_type, batch_size=None):
        """Test batch operation efficiency."""
        if batch_size is None:
            batch_size = self.config['evaluation_params']['batch_size']
            
        # Generate batch of random rotations
        rotations = self.generate_random_rotations(batch_size, seed=42)
        
        start_time = timeit.default_timer()
        params_batch = []
        for rot in rotations:
            params = self.to_representation(rot, rep_type)
            if params is not None:
                params_batch.append(params)
        batch_convert_time = timeit.default_timer() - start_time
        
        if len(params_batch) < 2:
            return float('inf')
        
        # Test batch composition (simulate parallel operations)
        start_time = timeit.default_timer()
        for i in range(0, len(params_batch) - 1, 2):
            self.compose(params_batch[i], params_batch[i + 1], rep_type)
        batch_compose_time = timeit.default_timer() - start_time
        
        # Lower is better (faster batch processing)
        return (batch_convert_time + batch_compose_time) / batch_size

    def efficiency_analysis(self, rep_type, num_trials=None):
        """Run efficiency analysis for composition and interpolation."""
        if num_trials is None:
            num_trials = self.config['evaluation_params']['num_trials']
            
        rots = self.generate_random_rotations(2, seed=42)
        params1 = self.to_representation(rots[0], rep_type)
        params2 = self.to_representation(rots[1], rep_type)
        
        if params1 is None or params2 is None:
            return {'composition_time': float('inf'), 'interpolation_time': float('inf')}
        
        # Composition timing
        def compose_func():
            self.compose(params1, params2, rep_type)
        
        comp_time = timeit.timeit(compose_func, number=num_trials) / num_trials
        
        # Interpolation timing - using NATIVE interpolation
        def interp_func():
            self.interpolate_native(params1, params2, 0.5, rep_type)
        
        interp_time = timeit.timeit(interp_func, number=num_trials) / num_trials
        
        return {'composition_time': comp_time, 'interpolation_time': interp_time}

    def evaluate_all(self, verbose=True):
        """Run complete evaluation of all representations."""
        results = []
        storage = self.storage_bytes()
        
        if verbose:
            print("Running comprehensive rotation representation evaluation...")
            print("=" * 60)
        
        for i, rep in enumerate(self.representations):
            if verbose:
                print(f"Evaluating {rep.title()}... ({i+1}/{len(self.representations)})")
            
            row = {'Representation': rep.title().replace('_', ' ')}
            
            # Basic properties
            row['Storage (bytes)'] = storage[rep]
            row['Parameters'] = storage[rep] // 8
            
            # Efficiency analysis
            efficiency = self.efficiency_analysis(rep)
            row['Comp. Time (μs)'] = efficiency['composition_time'] * 1e6
            row['Interp. Time (μs)'] = efficiency['interpolation_time'] * 1e6
            
            # Interpolation quality
            path_analysis = self.interpolation_path_analysis(rep)
            row['Path Length'] = path_analysis['path_length']
            
            # Advanced analysis
            row['Deriv. Continuity'] = self.derivative_continuity_test(rep)
            row['Batch Efficiency (μs)'] = self.parallel_efficiency_test(rep) * 1e6
            
            results.append(row)
        
        # Add Matrix Fisher for reference
        mf_row = {
            'Representation': 'Matrix Fisher',
            'Storage (bytes)': storage['matrix_fisher'],
            'Parameters': 9,
            'Comp. Time (μs)': 'N/A',
            'Interp. Time (μs)': 'N/A',
            'Path Length': 'N/A',
            'Deriv. Continuity': 'N/A',
            'Batch Efficiency (μs)': 'N/A'
        }
        results.append(mf_row)
        
        return pd.DataFrame(results)

    def format_dataframe_for_display(self, df):
        """Format DataFrame for display with mixed data types."""
        display_df = df.copy()
        
        # Define numeric columns that should be formatted
        numeric_columns = ['Storage (bytes)', 'Parameters', 'Comp. Time (μs)', 
                          'Interp. Time (μs)', 'Path Length', 'Deriv. Continuity',
                          'Batch Efficiency (μs)']
        
        for col in numeric_columns:
            if col in display_df.columns:
                # Only format numeric values, leave strings as-is
                display_df[col] = display_df[col].apply(
                    lambda x: f'{x:.6f}' if isinstance(x, (int, float)) and not pd.isna(x) and x != float('inf') else str(x)
                )
        
        return display_df


# Run the evaluation
if __name__ == "__main__":
    evaluator = RotationEvaluator()
    
    # Run evaluation
    print("Starting comprehensive rotation representation evaluation...")
    results_df = evaluator.evaluate_all(verbose=True)
    
    # Format and display results table
    print("\n" + "="*120)
    print("Empirical Evaluation Results on Different Metrics")
    print("="*120)
    
    formatted_df = evaluator.format_dataframe_for_display(results_df)
    print(formatted_df.to_string(index=False))
    
    # Save results
    # results_df.to_csv('rotation_evaluation_results.csv', index=False)
    # print(f"\nResults saved to 'rotation_evaluation_results.csv'")
    print("Evaluation complete!")