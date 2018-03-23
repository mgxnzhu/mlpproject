SCRIPTDIR=/afs/inf.ed.ac.uk/user/s17/s1749267/mlpproject/scripts

cd ${SCRIPTDIR}

source activate mlp

python Plot_datafeatures.py
python Data_Preprocession.py
bash compare_model.sh
python Plot.py
bash test.sh
python Plot_test.py
