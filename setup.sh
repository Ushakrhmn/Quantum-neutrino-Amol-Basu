echo "Running setup up script for environment ..."

echo "Creating virtual environment ..."
virtualenv --no-download ENV
pip install --no-index --upgrade pip

echo "Activating system modules and virtual environment ..."

source env.sh

echo "Installing packages for the first time ..."
echo "numpy, matplotlib, qiskit, qiskit_aer, scipy, setuptool, tqdm from local index"
pip install numpy matplotlib  qiskit qiskit_aer quimb scipy setuptools tqdm --no-index