"""
Setup configuration for Stress-Adaptive Cyber Defense Framework

Installation:
    pip install -e .                    # Development mode
    pip install -e ".[dev]"             # With dev tools
    pip install -e ".[all]"             # With all optional dependencies
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="stress-adaptive-defense",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Cognitive-Physiological Synchronization Framework for Stress-Adaptive Cyber Defense in IoT Environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/stress-adaptive-defense",
    project_urls={
        "Bug Tracker": "https://github.com/yourorg/stress-adaptive-defense/issues",
        "Documentation": "https://github.com/yourorg/stress-adaptive-defense/wiki",
        "Source Code": "https://github.com/yourorg/stress-adaptive-defense",
    },
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Core dependencies (required)
    install_requires=[
        # Core scientific computing
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
        "scipy>=1.7.0,<2.0.0",
        
        # Machine learning
        "scikit-learn>=1.0.0,<2.0.0",
        "xgboost>=1.5.0,<3.0.0",
        "tensorflow>=2.8.0,<3.0.0",
        
        # Signal processing
        "neurokit2>=0.1.7,<1.0.0",
        
        # Visualization
        "matplotlib>=3.4.0,<4.0.0",
        "seaborn>=0.11.0,<1.0.0",
        
        # Utilities
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
    ],
    
    # Optional dependencies
    extras_require={
        # Development tools
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
        
        # Advanced signal processing
        "signal": [
            "pywavelets>=1.3.0",
            "emd>=0.5.0",  # Empirical Mode Decomposition
        ],
        
        # HMM and time series
        "temporal": [
            "hmmlearn>=0.2.7",
            "statsmodels>=0.13.0",
        ],
        
        # Deep learning extras
        "dl": [
            "tensorflow-addons>=0.17.0",
            "keras-tuner>=1.1.0",
        ],
        
        # Documentation
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-bibtex>=2.4.0",
        ],
        
        # All optional dependencies
        "all": [
            # Dev tools
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "jupyter>=1.0.0",
            # Signal processing
            "pywavelets>=1.3.0",
            "hmmlearn>=0.2.7",
            "statsmodels>=0.13.0",
            # Deep learning
            "tensorflow-addons>=0.17.0",
            "keras-tuner>=1.1.0",
        ],
    },
    
    # Entry points (command-line scripts)
    entry_points={
        "console_scripts": [
            "train-stress-model=experiments.train_stress_model:main",
            "evaluate-loso=experiments.evaluate_loso:main",
            "run-soc-sim=experiments.run_soc_simulation:main",
            "reproduce-paper=experiments.reproduce_paper_results:main",
        ],
    },
    
    # Package metadata
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        
        # Intended audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Security",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # OS
        "Operating System :: OS Independent",
    ],
    
    keywords=[
        "stress",
        "cognitive computing",
        "cybersecurity",
        "physiological signals",
        "machine learning",
        "IoT security",
        "SOC",
        "decision making",
        "instance-based learning",
    ],
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": [
            "config/*.yaml",
            "config/*.json",
            "skills/*.md",
        ],
    },
    
    # Zip safe
    zip_safe=False,
)
