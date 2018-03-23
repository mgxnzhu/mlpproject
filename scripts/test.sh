SCRIPTDIR=/afs/inf.ed.ac.uk/user/s17/s1749267/mlpproject/scripts
${SCRIPTDIR}/get_model.sh 5 150 test_L5U150 > ${SCRIPTDIR}/My_model.py
python ${SCRIPTDIR}/Neural_Network.py --evaluation test --steps 100

${SCRIPTDIR}/get_model.sh 4 200 test_L4U200 > ${SCRIPTDIR}/My_model.py
python ${SCRIPTDIR}/Neural_Network.py --evaluation test --steps 100

${SCRIPTDIR}/get_model.sh 3 150 test_L3U150 > ${SCRIPTDIR}/My_model.py
python ${SCRIPTDIR}/Neural_Network.py --evaluation test --steps 100
