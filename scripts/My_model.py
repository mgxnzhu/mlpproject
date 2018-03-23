import tensorflow as tf

modelname = "l5_u{50..200..50}"
def Baseline_model(x, n_classes, reuse, is_training):
    with tf.variable_scope('Baseline', reuse=reuse):
        layer1 = tf.layers.dense(inputs=x, units={50..200..50}, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(inputs=layer1, units={50..200..50}, activation=tf.nn.leaky_relu)
        layer3 = tf.layers.dense(inputs=layer2, units={50..200..50}, activation=tf.nn.leaky_relu)
        layer4 = tf.layers.dense(inputs=layer3, units={50..200..50}, activation=tf.nn.leaky_relu)
        layer5 = tf.layers.dense(inputs=layer4, units={50..200..50}, activation=tf.nn.leaky_relu)
        out = tf.layers.dense(inputs=layer5, units=n_classes)
        out = tf.nn.softmax(out) if not is_training else out
    return out
