echo "*************** conda create center track env *************************"
conda create --name CenterTrack_MiD_TTC python=3.10
conda init bash
conda activate CenterTrack_MiD_TTC
pip install --upgrade pip setuptools wheel
echo "***************intall torch and torchvision *************************"
#pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
# conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch
echo "****************install cython and cocodataset pythonAPI************************"
pip install cython
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
echo "************clone CenterTrack ****************************"
git clone --recursive https://github.com/dheerajvarma24/CenterTrack_MiD_TTC.git
cd CenterTrack_MiD_TTC
echo "************clone nuscenes devkit ****************************"
cd src/tools
git clone https://github.com/nutonomy/nuscenes-devkit
echo "************clone DCNv2 latest ****************************"
cd  ../src/lib/model/networks
git clone https://github.com/jinfagang/DCNv2_latest.git
mv DCNv2_latest DCNv2

echo "************ install requirements  ****************************"
cd CenterTrack_MiD_TTC/
pip install -r requirements.txt

echo "************ compile DCNv2  ****************************"
cd src/lib/model/networks/DCNv2
sh ./make.sh
echo "************ done  ****************************"