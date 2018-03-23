n_layer=$1
n_unit=$2
i_layer=1

echo "import tensorflow as tf"
echo ""
echo "modelname = \"$3\""
echo "def Baseline_model(x, n_classes, reuse, is_training):"
echo "    with tf.variable_scope('Baseline', reuse=reuse):"
echo "        layer1 = tf.layers.dense(inputs=x, units=${n_unit}, activation=tf.nn.leaky_relu)"
while(( i_layer < n_layer))
do
	((ii_layer=i_layer+1));
	echo "        layer${ii_layer} = tf.layers.dense(inputs=layer${i_layer}, units=${n_unit}, activation=tf.nn.leaky_relu)"
	((i_layer=i_layer+1));
done
echo "        out = tf.layers.dense(inputs=layer${ii_layer}, units=n_classes)"
echo "        out = tf.nn.softmax(out) if not is_training else out"
echo "    return out"
