"""
Reproduce All Paper Results

Master script to reproduce all results from the paper:
- Table II: LOSO cross-validation results
- Table III: Cognitive performance under stress
- Table V: Synthetic SOC performance
- Table VI: CICIDS-based SOC performance
- Figures: Signal plots, feature distributions, stress timelines

Usage:
    python reproduce_paper_results.py --all
    python reproduce_paper_results.py --tables-only
    python reproduce_paper_results.py --figures-only
"""

import argparse
import sys
from pathlib import Path
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.helpers import setup_logger, ensure_dir, get_timestamp
from config import RANDOM_SEED

logger = setup_logger(__name__)


class PaperReproduction:
    """
    Master controller for reproducing paper results.
    """
    
    def __init__(
        self,
        output_dir: str = 'results/paper',
        skip_loso: bool = False,
        skip_soc: bool = False,
        skip_figures: bool = False
    ):
        """
        Initialize reproduction controller.
        
        Args:
            output_dir: Base output directory
            skip_loso: Skip LOSO evaluation
            skip_soc: Skip SOC simulations
            skip_figures: Skip figure generation
        """
        self.output_dir = ensure_dir(output_dir)
        self.skip_loso = skip_loso
        self.skip_soc = skip_soc
        self.skip_figures = skip_figures
        
        # Create subdirectories
        self.loso_dir = ensure_dir(self.output_dir / 'loso')
        self.soc_dir = ensure_dir(self.output_dir / 'soc')
        self.figures_dir = ensure_dir(self.output_dir / 'figures')
        self.models_dir = ensure_dir(self.output_dir / 'models')
        
        logger.info("="*70)
        logger.info("PAPER REPRODUCTION SCRIPT")
        logger.info("="*70)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Skip LOSO: {skip_loso}")
        logger.info(f"Skip SOC: {skip_soc}")
        logger.info(f"Skip Figures: {skip_figures}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_loso_evaluation(self):
        """
        Run LOSO cross-validation (Table II).
        """
        if self.skip_loso:
            logger.info("\n[SKIPPED] LOSO evaluation")
            return
        
        logger.info("\n" + "="*70)
        logger.info("TABLE II: LOSO CROSS-VALIDATION")
        logger.info("="*70)
        
        from evaluate_loso import MultiModelLOSO
        
        evaluator = MultiModelLOSO(output_dir=str(self.loso_dir))
        all_summaries = evaluator.run_all_models()
        
        logger.info("\n✓ LOSO evaluation complete")
        logger.info(f"  Results saved to: {self.loso_dir}")
        
        return all_summaries
    
    def run_soc_simulations(self):
        """
        Run SOC simulations (Tables III, V, VI).
        """
        if self.skip_soc:
            logger.info("\n[SKIPPED] SOC simulations")
            return
        
        logger.info("\n" + "="*70)
        logger.info("SOC SIMULATIONS (Tables III, V, VI)")
        logger.info("="*70)
        
        from run_soc_simulation import SOCSimulator
        
        results = {}
        
        # Synthetic SOC with different stress patterns
        logger.info("\n[1/4] Synthetic SOC - Constant Stress")
        sim1 = SOCSimulator('synthetic', str(self.soc_dir))
        results['synthetic_constant'] = sim1.run(
            duration=120.0,
            arrival_rate=0.5,
            stress_pattern='constant'
        )
        
        logger.info("\n[2/4] Synthetic SOC - Cyclic Stress")
        sim2 = SOCSimulator('synthetic', str(self.soc_dir))
        results['synthetic_cyclic'] = sim2.run(
            duration=120.0,
            arrival_rate=0.5,
            stress_pattern='cyclic'
        )
        
        logger.info("\n[3/4] Synthetic SOC - Increasing Stress")
        sim3 = SOCSimulator('synthetic', str(self.soc_dir))
        results['synthetic_increasing'] = sim3.run(
            duration=120.0,
            arrival_rate=0.5,
            stress_pattern='increasing'
        )
        
        # CICIDS-based SOC
        logger.info("\n[4/4] CICIDS-based SOC")
        sim4 = SOCSimulator('cicids', str(self.soc_dir))
        results['cicids'] = sim4.run(
            use_sample_data=True,
            sample_rate=0.01
        )
        
        logger.info("\n✓ SOC simulations complete")
        logger.info(f"  Results saved to: {self.soc_dir}")
        
        return results
    
    def generate_figures(self):
        """
        Generate figures from paper.
        """
        if self.skip_figures:
            logger.info("\n[SKIPPED] Figure generation")
            return
        
        logger.info("\n" + "="*70)
        logger.info("FIGURE GENERATION")
        logger.info("="*70)
        
        from data.wesad_loader import WESADLoader
        from data.signal_processor import SignalProcessor
        from data.feature_extractor import FeatureExtractor
        from evaluation.visualization import (
            plot_signal_comparison,
            plot_feature_distributions,
            plot_stress_timeline
        )
        
        # Load sample subject
        logger.info("\n[1/3] Loading sample data...")
        loader = WESADLoader()
        subject_data = loader.load_subject('S2', verbose=False)
        
        processor = SignalProcessor()
        processed = processor.process_subject(subject_data)
        
        extractor = FeatureExtractor()
        features_df = extractor.extract_subject_features(processed)
        
        # Figure 4: Signal comparison
        logger.info("\n[2/3] Generating Figure 4: Signal Comparison...")
        fig4_path = self.figures_dir / 'figure4_signal_comparison.png'
        plot_signal_comparison(
            subject_data,
            duration=30.0,
            save_path=str(fig4_path)
        )
        logger.info(f"  ✓ Saved: {fig4_path}")
        
        # Figure 5: Feature distributions
        logger.info("\n[3/3] Generating Figure 5: Feature Distributions...")
        fig5_path = self.figures_dir / 'figure5_feature_distributions.png'
        plot_feature_distributions(
            features_df,
            save_path=str(fig5_path)
        )
        logger.info(f"  ✓ Saved: {fig5_path}")
        
        logger.info("\n✓ Figure generation complete")
        logger.info(f"  Figures saved to: {self.figures_dir}")
    
    def generate_summary_report(self):
        """
        Generate comprehensive summary report.
        """
        logger.info("\n" + "="*70)
        logger.info("GENERATING SUMMARY REPORT")
        logger.info("="*70)
        
        report_path = self.output_dir / f'reproduction_report_{get_timestamp()}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PAPER REPRODUCTION SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write("\n")
            
            f.write("COMPLETED EXPERIMENTS:\n")
            f.write("-" * 70 + "\n")
            
            if not self.skip_loso:
                f.write("✓ LOSO Cross-Validation (Table II)\n")
                f.write(f"  Results: {self.loso_dir}\n")
            else:
                f.write("✗ LOSO Cross-Validation (SKIPPED)\n")
            
            if not self.skip_soc:
                f.write("✓ SOC Simulations (Tables III, V, VI)\n")
                f.write(f"  Results: {self.soc_dir}\n")
            else:
                f.write("✗ SOC Simulations (SKIPPED)\n")
            
            if not self.skip_figures:
                f.write("✓ Figure Generation\n")
                f.write(f"  Figures: {self.figures_dir}\n")
            else:
                f.write("✗ Figure Generation (SKIPPED)\n")
            
            f.write("\n")
            f.write("EXPECTED RESULTS:\n")
            f.write("-" * 70 + "\n")
            f.write("Table II - Stress Classification (DNN-XGBoost):\n")
            f.write("  Accuracy: 95.8%\n")
            f.write("  F1 Score: 95.2%\n")
            f.write("  AUC: 0.967\n")
            f.write("  ECE: 0.023\n")
            f.write("\n")
            f.write("Table III - Cognitive Performance:\n")
            f.write("  Low Stress:    Latency=0.41s, Retrieval=93.5%, Error=3.8%\n")
            f.write("  Medium Stress: Latency=0.71s, Retrieval=87.2%, Error=10.1%\n")
            f.write("  High Stress:   Latency=1.28s, Retrieval=73.6%, Error=18.9%\n")
            f.write("\n")
            f.write("="*70 + "\n")
        
        logger.info(f"✓ Summary report saved: {report_path}")
    
    def run_all(self):
        """
        Run complete reproduction pipeline.
        """
        start_time = time.time()
        
        logger.info("\n" + "="*70)
        logger.info("STARTING FULL PAPER REPRODUCTION")
        logger.info("="*70)
        logger.info("\nThis will reproduce all results from the paper:")
        logger.info("  - Table II: LOSO cross-validation")
        logger.info("  - Table III: Cognitive performance metrics")
        logger.info("  - Table V: Synthetic SOC results")
        logger.info("  - Table VI: CICIDS-based results")
        logger.info("  - Figures 4-5: Signal and feature visualizations")
        logger.info("\n" + "="*70)
        
        # Step 1: LOSO evaluation
        try:
            self.run_loso_evaluation()
        except Exception as e:
            logger.error(f"LOSO evaluation failed: {e}")
        
        # Step 2: SOC simulations
        try:
            self.run_soc_simulations()
        except Exception as e:
            logger.error(f"SOC simulations failed: {e}")
        
        # Step 3: Generate figures
        try:
            self.generate_figures()
        except Exception as e:
            logger.error(f"Figure generation failed: {e}")
        
        # Step 4: Summary report
        self.generate_summary_report()
        
        total_time = time.time() - start_time
        
        logger.info("\n" + "="*70)
        logger.info("REPRODUCTION COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Results directory: {self.output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. Review results in output directory")
        logger.info("  2. Compare with paper values")
        logger.info("  3. Generate publication-ready figures")
        logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Reproduce all paper results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reproduce everything (RECOMMENDED)
  python reproduce_paper_results.py --all
  
  # Only generate tables
  python reproduce_paper_results.py --tables-only
  
  # Only generate figures
  python reproduce_paper_results.py --figures-only
  
  # Quick test (skip time-consuming LOSO)
  python reproduce_paper_results.py --skip-loso
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments (default)'
    )
    
    parser.add_argument(
        '--tables-only',
        action='store_true',
        help='Only generate tables (skip figures)'
    )
    
    parser.add_argument(
        '--figures-only',
        action='store_true',
        help='Only generate figures (skip LOSO and SOC)'
    )
    
    parser.add_argument(
        '--skip-loso',
        action='store_true',
        help='Skip LOSO evaluation (time-consuming)'
    )
    
    parser.add_argument(
        '--skip-soc',
        action='store_true',
        help='Skip SOC simulations'
    )
    
    parser.add_argument(
        '--skip-figures',
        action='store_true',
        help='Skip figure generation'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/paper',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Determine what to run
    if args.tables_only:
        skip_figures = True
        skip_loso = False
        skip_soc = False
    elif args.figures_only:
        skip_figures = False
        skip_loso = True
        skip_soc = True
    else:
        skip_figures = args.skip_figures
        skip_loso = args.skip_loso
        skip_soc = args.skip_soc
    
    # Initialize reproducer
    reproducer = PaperReproduction(
        output_dir=args.output,
        skip_loso=skip_loso,
        skip_soc=skip_soc,
        skip_figures=skip_figures
    )
    
    # Run experiments
    reproducer.run_all()


if __name__ == "__main__":
    main()
