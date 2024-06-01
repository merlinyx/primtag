# primtag

## Setting up python environment

Download openscad from [here](https://openscad.org/).

Add the line below to your `.zshrc` or `.bashrc` if you are on Mac / Linux depending on the shell you're using
`export OPENSCAD_EXEC=<full_path_to_openscad_executable>`

Add an environment variable named `OPENSCAD_EXEC` if you are on Windows.

You can update the `modules` variable to be the correct absolute path for you.

Then use `pip install solidpython` and `pip install libigl` to install the packages in your python environment. A working python version is 3.8 but you can probably use the latest python version.

## Using fabhacks_min

activate the environment and run `python viewer`

## Access libigl repository
Clone the libigl repository into your local computer by the command: `git clone https://github.com/libigl/libigl.git`

Build libgil as followings:

`cd libigl`

`mkdir build`

`cd build`

`cmake ..`

`make`

To run the tutorial files, assume you want to run the executable for 106_ViewerMenu.cpp, go to the directory of bin, and run `./106_ViewerMenu`

For more instructions of how to use libigl repository, visit their tutorial page: https://libigl.github.io/

## Build UI For Mesh Segmentation
Under `step2` directory, `106_ViewerMenu.cpp` adds more functionalities, such as keeping track of the mouse coordinates, plotting the screen points, and segmenting the mesh, etc, for users to interact with the uploaded mesh. 
We also include the original and reduced mesh in this directory

We use libigl repository to segment primitive mesh, called reduced mesh, from the original mesh and save it as an obj file. 
First, upload a mesh. Second, click the `track mouse` button to keep tracking the screen coordinates of the mouse. Third, use the `plot points` button to plot the intersection points and use 'color faces' button to visualize the hit faces. Lastly, click the `segmentation` button to segment and save the mesh from the original mesh based on the selected region. Make sure the selected region to be perfect and to be the tight boundary for the primitive type you want.

## Parameters Optimization
Under the `step3_parameters_learning` directory, we include 7 different types of primitives and corresponding parameter learning `.ipynb` files. 

The Python environment for these Jupiter Notebook files is 3.10
You also need to download pytorch and pytorch3d dependencies. 

## Limitations
Due to the instability of randomly sampling the points on the surface of each reduced mesh (primitive mesh), the learning part for Clip, Hook, and Hole might not be accurate at all times. When the results are wrong, rerun the file. 
