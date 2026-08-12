# kodama split repositories

This directory contains split-ready repositories:

- `KODAMA`: canonical R package with a vendored portable CPU core
- `kodama-python`: thin Python wrapper for `kodama-cpp`

They are staged inside this checkout because the current sandbox can only write
inside the `kodama-cpp` workspace. They are intended to be moved or pushed as:

- `tkcaccia/KODAMA`
- `tkcaccia/kodama-python`

The R source package vendors the portable CPU implementation so it can be
installed on CRAN/Bioconductor builders without an external system library.
Optional developer builds link CUDA or Metal `kodama-cpp`. The Python wrapper
links the standalone core and exposes its broader numerical API.
