## climsim: climate similarity and uniqueness

This library calculates cell-bassed climate similarity and uniqueness using multi-varaite ecological distance.

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
