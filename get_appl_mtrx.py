# This script is only for visualizing the application suitability matrix based on heuristic evaluations.
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings('ignore')

class ApplicationMatrixPlotter:
    def __init__(self):
        # Create fig directory if it doesn't exist
        self.fig_dir = 'fig'
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)

    def plot_application_suitability_matrix(self):
        """Plot application suitability matrix."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        applications = ['Games', 'Robotics', 'Computer\nVision', 'Animation', 'Machine\nLearning', 'Scientific\nComputing']
        representations = ['Quaternion', 'Euler', 'Rotation\nMatrix', 'Axis-Angle', '6D\nContinuous', 'Exponential\nMap']
        
        # Load expert opinions to generate suitability scores
        try:
            with open('questionnaire_data.json', 'r') as f:
                data = json.load(f)
            
            expert_scores = []
            for evaluator in data['evaluator']:
                # Extract scores in the order of representations
                scores = [
                    evaluator['scores']['Quaternion'],
                    evaluator['scores']['Euler'],
                    evaluator['scores']['Rotation Matrix'],
                    evaluator['scores']['Axis-Angle'],
                    evaluator['scores']['6D Continuous'],
                    evaluator['scores']['Exponential Map']
                ]
                expert_scores.append(scores)
            
            # Calculate average suitability from expert opinions
            suitability = np.mean(expert_scores, axis=0)
            
        except FileNotFoundError:
            print("Warning: 'questionnaire_data.json' not found. Using fallback values.")
        
        im = ax.imshow(suitability, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(applications)))
        ax.set_xticklabels(applications, fontsize=12)
        ax.set_yticks(range(len(representations)))
        ax.set_yticklabels(representations, fontsize=12)
        ax.set_title('Application Suitability Matrix', fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Suitability Score', fontsize=12)
        
        # Add text annotations
        for i in range(len(representations)):
            for j in range(len(applications)):
                text = f'{suitability[i, j]:.1f}'
                color = 'white' if suitability[i, j] < 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', 
                       color=color, fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/application_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Application matrix plot saved to '{self.fig_dir}/application_matrix.png'")


if __name__ == "__main__":
    plotter = ApplicationMatrixPlotter()
    plotter.plot_application_suitability_matrix()
    print("Plot generation complete!")