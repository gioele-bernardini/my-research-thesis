python3 -m venv thesis_env
source thesis_env/bin/activate

pip install torch torchaudio numpy soundfile

db_download.py
K*
generate



# per pulizia cmake
rm -rf CMakeCache.txt CMakeFiles build
mkdir build
cd build
cmake ..
# per compilare
make -j$(nproc)

IMPORTANTISSIMO -> DEVO PRIMA LANCIARE IL DEBUG SE POI VOGLIO USARE LA SCORCIATOIA!!!!!
