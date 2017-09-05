import tensorflow as tf

from random import randint

random_values = [None] * 10000
odd_even = [None] * 10000

for i in range(0, 10000):
	random_values[i] = [randint(1, 10)]

for i in range(0, 10000):
	odd = random_values[i][0] & 1
	odd_even[i] = [odd, 1-odd]

def get_randoms():
	return random_values

def get_odd_even():
	return odd_even

x = tf.placeholder(tf.float32, [None, 1]) # 1 feature, which is our random number

#W = tf.Variable(tf.zeros([1, 2])) # 1 feature, 2 classes (odd, even)
W = tf.Variable(tf.random_normal([1, 2], stddev=0.5), name="weights")


b = tf.Variable(tf.zeros([2])) # bias

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 2]) # 2 classes
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# need to generate inputs for our random numbers and the actual answers
# x: feature value
# y_: classification

sess.run(train_step, feed_dict={x: get_randoms(), y_: get_odd_even()})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

validation_random_values =  [[1],  [2],  [3],  [4]]
validation_odd_even =       [[1,0],[0,1],[1,0],[0,1]] # 0 = odd, 1 = even

validation_random_values =  [[3]]
validation_odd_even =       [[1,0]] # 0 = odd, 1 = even

print("----------------------")
print(sess.run(accuracy, feed_dict={x: validation_random_values, y_: validation_odd_even}))

prediction = tf.argmax(y,1)
print(y.eval(feed_dict={x: validation_random_values}))
