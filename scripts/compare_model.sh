SCRIPTDIR=/afs/inf.ed.ac.uk/user/s17/s1749267/mlpproject/scripts
for n_layer in {2..5}
do
	for n_unit in {50..200..50}
	do
		${SCRIPTDIR}/get_model.sh ${n_layer} ${n_unit} "l${n_layer}_u${n_unit}" > ${SCRIPTDIR}/My_model.py
		python ${SCRIPTDIR}/Neural_Network.py
	done
done
mv l*.npy ${SCRIPTDIR}/../stats
