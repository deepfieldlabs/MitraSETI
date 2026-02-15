"""astroSETI — Intelligent SETI Signal Analysis."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="astroseti",
    version="0.1.0",
    author="Saman Tabatabaeian",
    author_email="saman@astroseti.dev",
    description="Intelligent SETI Signal Analysis — Decode the Cosmos with Machine Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamanTabworlds/astroSETI",
    project_urls={
        "Bug Tracker": "https://github.com/SamanTabworlds/astroSETI/issues",
        "Documentation": "https://github.com/SamanTabworlds/astroSETI#readme",
        "Source Code": "https://github.com/SamanTabworlds/astroSETI",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "httpx>=0.24.0",
        "aiosqlite>=0.19.0",
        "jinja2>=3.1.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0",
        "PyQt5>=5.15.0",
        "astroquery>=0.4.6",
        "websockets>=11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "ruff>=0.1.0",
            "maturin>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "astroseti=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    keywords="seti, astronomy, signal-processing, machine-learning, radio-astronomy, rust",
    license="MIT",
)
