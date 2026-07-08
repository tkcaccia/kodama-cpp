# kodama split repositories

This directory contains split-ready repositories:

- `kodama-r`: thin R wrapper for `kodama-cpp`
- `kodama-python`: thin Python wrapper for `kodama-cpp`

They are staged inside this checkout because the current sandbox can only write
inside the `kodama-cpp` workspace. They are intended to be moved or pushed as:

- `tkcaccia/kodama-r`
- `tkcaccia/kodama-python`

The wrappers do not contain algorithm implementations. They link to the
standalone `kodama-cpp` library and expose KODAMA matrix optimization with KNN
and PLS-LDA classifiers.
