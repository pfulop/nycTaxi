import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
import tensorflow as tf
import numpy as np

predict = pd.read_csv('./inputs/test.csv')
train = pd.read_csv('./inputs/train.csv').sample(n=1000)


# train = pd.read_csv('./inputs/train.csv')

def strip_time(data):
    t = data['pickup_datetime'].apply(
        lambda d: pd.Series([datetime.strptime(d, "%Y-%m-%d %H:%M:%S").strftime("%H"),
                             datetime.strptime(d, "%Y-%m-%d %H:%M:%S").strftime("%d%m")], index=['hour', 'date']))
    return pd.concat([data, t], axis=1)


train = strip_time(train)
predict = strip_time(predict)


def get_time_period(i):
    return train[train['hour'] == str(i).zfill(2)]


def animate_in_time():
    fig, (ax_pup, ax_doff) = plt.subplots(1, 2)
    plt.tight_layout()

    ax_pup.set_xlim(-74.1, -73.7)
    ax_pup.set_ylim(40.60, 40.90)
    ax_pup.set_title("Pickup")

    ax_doff.set_xlim(-74.1, -73.7)
    ax_doff.set_ylim(40.60, 40.90)
    ax_doff.set_title("Dropoff")

    markers_pup, = ax_pup.plot([], [], '.', markersize=0.05)
    markers_doff, = ax_doff.plot([], [], 'r.', markersize=0.05)

    time_text = ax_pup.text(0.05, 0.05, '', transform=ax_pup.transAxes)

    def animate(i):
        anim_data = get_time_period(i)
        markers_doff.set_data(anim_data.dropoff_longitude, anim_data.dropoff_latitude)
        markers_pup.set_data(anim_data.pickup_longitude, anim_data.pickup_latitude)
        time_text.set_text('%02d H' % i)
        return markers_doff, markers_pup, time_text,

    def init():
        markers_doff.set_data([], [])
        markers_pup.set_data([], [])
        return markers_doff, markers_pup,

    animation.FuncAnimation(fig, animate, init_func=init,
                            frames=24, interval=40, blit=True)
    plt.show()


def show_hour_mean():
    train.groupby(['hour'])['trip_duration'].mean().plot(kind="bar")
    plt.show()


def create_placeholders(n_x, n_y):
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name="Y")
    X = tf.placeholder(tf.float32, shape=(n_x, None), name="X")
    return X, Y


def initialize_parameters():
    W1 = tf.get_variable("W1", [25, 8], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 12], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    print(parameters)

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    return A3


def compute_cost(A3, Y):

    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf.log1p(A3), tf.log1p(Y)))))
    return cost


def random_mini_batches(X, Y, size):
    folds = np.maximum(np.floor(X.shape[1] / size), 2).astype(int)
    X_batches = np.array_split(X, folds, axis=1)
    Y_batches = np.array_split(Y, folds, axis=1)
    batches = []
    for idx, x in enumerate(X_batches):
        batches.append((x, Y_batches[idx]))
    return batches


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=128, print_cost=True, predict = []):
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    A3 = forward_propagation(X, parameters)

    cost = compute_cost(A3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    mbatch = 0

    tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            mbatch = mbatch + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(A3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))


        pred = sess.run(A3, feed_dict={X: predict})
        saver.save(sess, './logs/nyc-model')
        t = pd.read_csv('./inputs/test.csv')
        output = pd.DataFrame()
        output['id'] = t['id']
        print(pred[0])
        print(pred.shape)
        print(t['id'].values.shape)
        output['trip_duration'] = pred[0]
        output[['id','trip_duration']].to_csv('./submit.csv', index = False)
        sess.close()
        return parameters


Y = train['trip_duration'].values
train.drop(['trip_duration', 'store_and_fwd_flag', 'pickup_datetime', 'dropoff_datetime', 'id'], axis=1, inplace=True)
predict.drop(['store_and_fwd_flag', 'pickup_datetime' , 'id'], axis=1, inplace=True)

msk = np.random.rand(len(train)) < 0.8
test = train[~msk]
train = train[msk]
Y = np.reshape(Y, (1, Y.shape[0]))
Ytest = Y[:, ~msk]
Y = Y[:, msk]
tf.reset_default_graph()
model(train.values.T, Y, test.values.T, Ytest, predict=predict.values.T)

