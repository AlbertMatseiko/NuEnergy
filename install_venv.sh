if [ -d "$(pwd)/venvNE" ]; then
    echo "venv exists"
else
    create_here venvNE
fi

activate_here venvNE

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
conda install -c conda-forge tensorflow-gpu=2.15.0 -y
conda install jupyter numpy matplotlib scipy h5py pydantic -y
pip install show_h5