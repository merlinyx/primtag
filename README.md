# primtag

## setting up python environment

Download openscad from [here](https://openscad.org/).

Add the line below to your `.zshrc` or `.bashrc` if you are on Mac / Linux depending on the shell you're using
`export OPENSCAD_EXEC=<full_path_to_openscad_executable>`

Add an environment variable named `OPENSCAD_EXEC` if you are on Windows.

You can update the `modules` variable to be the correct absolute path for you.

Then use `pip install solidpython` and `pip install libigl` to install the packages in your python environment. A working python version is 3.8 but you can probably use the latest python version.

## Using fabhacks_min

activate the environment and run `python viewer`