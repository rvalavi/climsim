# climsim
[![CI](https://github.com/rvalavi/climsim/actions/workflows/CI.yml/badge.svg)](https://github.com/rvalavi/climsim/actions/workflows/CI.yml)

`climsim` is a Python package for measuring climate similarity, dissimilarity, and environmental uniqueness in gridded spatial data. It computes kernel-weighted multivariate distances that quantify how distinct local climate conditions are compared with other locations, optionally within a spatial radius.

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

The currently impelmeted methods:
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