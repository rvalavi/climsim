# climsim: climate dissimilarity and uniqueness
[![CI](https://github.com/rvalavi/climsim/actions/workflows/CI.yml/badge.svg)](https://github.com/rvalavi/climsim/actions/workflows/CI.yml)

This library measures climate dissimilarity and uniqueness for gridded spatial data. For each cell, it calculates how different local climate conditions are from other cells using kernel-weighted multivariate distance metrics, optionally within a radius.

## Installation

This package includes a Rust extension, so you will need to have the
[Rust](https://www.rust-lang.org/) programming language installed on your system
before proceeding. If Rust is not already installed, follow the instructions on
the Rust website to set it up.

First, create and activate a Python virtual environment of your choice.
Then install `maturin`, which is used to build and install the Rust-based
components:

```bash
pip install maturin
```

Next, clone the repository and move into the project directory:
```bash
git clone git@github.com:rvalavi/climsim.git
cd climsim
```

Finally, compile and install the library into your active Python environment:
```bash
maturin develop --release
```

## Usage

Impelmeted method:
* `dissim`: Calculates a kernel-weighted multivariate dissimilarity between climate samples. It sums distances across variables (optionally within a spatial neighborhood) to quantify how different each sample’s climate is compared to others.
* `climdist`: Computes the minimum climate/environmental distance of a cell from a sample point.

## Unit testing

Testing core module (without Python binding):

```bash
cargo test core
```
Testing main Rust fucntion (same as above, now):

```bash
cargo test dissimrs
```