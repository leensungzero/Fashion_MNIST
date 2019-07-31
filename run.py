import tensorflow as tf
import numpy as np


fashion = np.loadtxt('./fashionmnist/fashion-mnist_test.csv', delimiter=',', dtype=np.float32, skiprows=1)
fashion /= 255.0
x_data = fashion[:, 1:]
y_data = fashion[:, [0]] * 255

print("x_data:", x_data.shape)
print("y_data:", y_data.shape)

training_x_data = x_data[0:int(x_data.shape[0] * 0.7), :]
training_y_data = y_data[0:int(y_data.shape[0] * 0.7), :]

print("training_x_data:", training_x_data.shape)
print("training_y_data:", training_y_data.shape)
print(training_y_data)

test_x_data = x_data[int(x_data.shape[0] * 0.7):, :]
test_y_data = y_data[int(y_data.shape[0] * 0.7):, :]

print("test_x_data:", test_x_data.shape)
print("test_y_data:", test_y_data.shape)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
# Y = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.int32, [None, 1])


Y_one_hot =tf.one_hot(Y, nb_classes)
sess = tf.Session()
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])


W = tf.Variable(tf.random_normal([784, nb_classes], 3), name='weight')
b = tf.Variable(tf.random_normal([nb_classes], 3), name='bias')
# W = tf.Variable(tf.zeros([784, nb_classes]), name='weight')
# b = tf.Variable(tf.zeros([nb_classes]), name='bias')


# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for step in range(13001):
        loss, acc, _ = sess.run([cost, accuracy, optimizer], feed_dict={X: training_x_data, Y: training_y_data})

        if step % 50 == 0:
            print("Step: {:5}\tLoss: {:3f}\tAcc: {:2%}".format(
                step, loss, acc
            ))

    print("Accuracy: {:2%}".format(accuracy.eval(session=sess, feed_dict={X: test_x_data, Y: test_y_data})))