# setup.py
from setuptools import setup, find_packages

setup(
    name="weather-forecast-app",
    version="1.0.0",
    author="Kelompok 3 Big Data",
    description="Weather Temperature Forecasting App",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.28.0',
        'pandas>=1.5.3',
        'numpy>=1.23.5',
        'scikit-learn>=1.2.2',
        'matplotlib>=3.6.3',
        'seaborn>=0.12.2',
        'plotly>=5.13.0',
    ],
    python_requires='>=3.8',
)