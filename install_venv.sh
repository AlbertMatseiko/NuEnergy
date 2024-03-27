if [ -d "$(pwd)/venvNE" ]; then
    echo "venv exists"
else
    create_here venvNE
fi

activate_here venvNE

# Make sure youre python, tensorflow and cuda versions are compatible!
# The build works for CUDA Version = 11.1 (GPU GeForce GTX 1080 and 2080)
conda install python=3.10
conda install -c conda-forge tensorflow-gpu=2.11 -y
conda install jupyter numpy matplotlib scipy h5py pydantic -y

pip3 install show_h5