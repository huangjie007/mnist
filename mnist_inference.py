#前向神经网络过程及参数定义


import tensorflow as tf

INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10

def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses",regularizer(weights))
    return weights
#定义前向神经网络传播过程
def inference(input_tensor,regularizer):
    #声明第一层神经网络并前向传播
    with tf.variable_scope("layer1"):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    #声明第二层神经网络并前向传播
    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights) + biases
    return layer2
