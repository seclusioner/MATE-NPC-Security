from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


setup(
    name="mate-npc-security",
    version="0.1.0",
    description=(
        "Layered runtime security for "
        "LLM-powered NPC agents."
    ),
    long_description=(
        ROOT / "README.md"
    ).read_text(
        encoding="utf-8"
    ),
    long_description_content_type="text/markdown",
    author="Jing Lu and Jing-ming Guo",
    author_email="M11307506@mail.ntust.edu.tw",
    license="MIT",
    packages=find_packages(
        exclude=(
            "tests",
            "tests.*",
        )
    ),
    install_requires=[
        "numpy",
        "outlines",
        "pydantic",
        "PyYAML",
        "torch",
        "transformers",
    ],
    extras_require={
        "train": [
            "datasets",
            "scikit-learn",
            "tqdm",
        ],
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "mate-npc=examples.demo:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    zip_safe=False,
)
