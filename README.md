# Primtag

## Setting Up the Python Environment

First, download OpenSCAD from [here](https://openscad.org/).

For Mac or Linux users, add the following line to your `.zshrc` or `.bashrc`, depending on the shell you're using:
```bash
export OPENSCAD_EXEC=<full_path_to_openscad_executable>
```

For Windows users, add an environment variable named `OPENSCAD_EXEC` with the path to the OpenSCAD executable.

Ensure the `modules` variable is updated to the correct absolute path for your system.

Then, install the necessary packages in your Python environment:
```bash
pip install solidpython
pip install libigl
```

A working Python version is 3.8, but the latest version should also be compatible.

## Using fabhacks_min

Activate your environment and run:
```bash
python viewer
```

## Accessing the libigl Repository

Clone the libigl repository to your local machine:
```bash
git clone https://github.com/libigl/libigl.git
```

Build libigl with the following commands:
```bash
cd libigl
mkdir build
cd build
cmake ..
make
```

To run the tutorial files, for example, `106_ViewerMenu.cpp`, navigate to the `bin` directory and run:
```bash
./106_ViewerMenu
```

For more detailed instructions on using the libigl repository, visit their [tutorial page](https://libigl.github.io/).

## Building UI for Mesh Segmentation

In the `step2` directory, `106_ViewerMenu.cpp` has been enhanced with additional functionalities, such as tracking mouse coordinates, plotting screen points, and segmenting the mesh for user interaction with the uploaded mesh. This directory also contains the original and reduced meshes.

We use the libigl repository to segment a primitive mesh (reduced mesh) from the original mesh and save it as an OBJ file. Follow these steps:

1. Upload a mesh.
2. Click the `track mouse` button to start tracking the mouse's screen coordinates.
3. Use the `plot points` button to plot intersection points and the `color faces` button to visualize the hit faces.
4. Click the `segmentation` button to segment and save the mesh from the original mesh based on the selected region.

Ensure that the selected region is precise and forms a tight boundary for the desired primitive type.

## Parameters Optimization

In the `step3_parameters_learning` directory, we include seven different types of primitives along with corresponding parameter learning `.ipynb` files.

The Python environment for these Jupyter Notebook files requires Python 3.10, along with PyTorch and PyTorch3D dependencies.

## Limitations

Due to the instability of randomly sampling points on the surface of each reduced mesh, the learning part for Clip, Hook, and Hole might not be accurate at all times. If the results are incorrect, rerun the file.

There is a slight discrepancy between our current techniques and the ground truth for some meshes with small sizes. However, as the size of the mesh increases, our techniques become more accurate and have a smaller margin of error.

## Future Work

We have uploaded the volumetric mesh, which includes interior faces, for the reduced mesh in the `step2` directory. In future work, we aim to reconstruct the parametrization function for each primitive to sample points both on the surface and interior, improving the accuracy and stability of the learning part.
