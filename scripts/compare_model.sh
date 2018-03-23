for n_layer in {2..5}
do
	for n_unit in {50..200..50}
	do
		./get_model.sh ${n_layer} ${n_unit} "l${n_layer}_u${n_unit}" > My_model.py
		python Neural_Network.py
	done
done
mv *.npy ../stats
