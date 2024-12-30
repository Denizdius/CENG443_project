# CENG443 Project

## Project Overview

This project is part of the CENG443 course. The goal of the project is to implement basic GEMM (General Matrix Multiply) applications for different floating-point precisions and different architectures.

## Building the Project

To build the project, follow these steps:

1. Navigate to the `src` directory:
    ```sh
    cd src
    ```

2. Use `make` to build the project. Inside the Makefile, you can find application-specific targets for different architectures and precisions:
    ```sh
    make
    ```

### Available Targets

- `hgemm`: Builds the half-precision GEMM application for the default architecture.
- `sgemm`: Builds the single-precision GEMM application for the default architecture.
- `hgemm_sm80`: Builds the half-precision GEMM application for the `sm_80` architecture.
- `sgemm_sm80`: Builds the single-precision GEMM application for the `sm_80` architecture.
- `hgemm_sm75`: Builds the half-precision GEMM application for the `sm_75` architecture.
- `sgemm_sm75`: Builds the single-precision GEMM application for the `sm_75` architecture.

To build a specific target, use the following command:
```sh
make <target>

