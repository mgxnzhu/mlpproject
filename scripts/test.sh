./get_model.sh 5 150 test_L5U150 > My_model.py
python Neural_Network.py --evaluation test

./get_model.sh 4 200 test_L4U200 > My_model.py
python Neural_Network.py --evaluation test

./get_model.sh 3 150 test_L3U150 > My_model.py
python Neural_Network.py --evaluation test
