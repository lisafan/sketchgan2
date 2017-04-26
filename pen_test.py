import tensorflow as tf

def main():
    with tf.name_scope("discriminator_loss"):
        shape = classes_real.get_shape().dims
        real_penalty = tf.constant(1, shape=shape)
        fake_penalty = tf.constant(1, shape=shape)
        num_classes_const = tf.constant(NUM_CLASSES, shape=shape)
        A_const = tf.constant(2, shape=shape) 
        B_const = tf.constant(3, shape=shape) 
        C_const = tf.constant(4, shape=shape)
        classes_real_64 = classes_real
    
        # predictions
        real_softmax = tf.nn.softmax(real_outputs, dim=-1)	
        fake_softmax = tf.nn.softmax(fake_outputs, dim=-1)
        real_class_pred = tf.cast(tf.argmax(real_softmax, axis=1), tf.int32)
        fake_class_pred = tf.cast(tf.argmax(fake_softmax, axis=1), tf.int32)

        # determine if real / fake prediction was correct
        real_binary = tf.greater(real_class_pred, num_classes_const)
        fake_binary = tf.less(fake_class_pred, num_classes_const)

        # determine if class prediction was correct
        real_class_binary = tf.not_equal(tf.mod(real_class_pred, num_classes_const), classes_real_64)
        fake_class_binary = tf.not_equal(tf.mod(fake_class_pred, num_classes_const), classes_real_64)

        # get wrong wrong instances
        real_ = tf.logical_and(real_binary, real_class_binary)
        fake_ = tf.logical_and(fake_binary, fake_class_binary)

        # penalty calc
        real_penalty = tf.where(real_binary, A_const, real_penalty)
        real_penalty = tf.where(real_class_binary, B_const, real_penalty)
        real_penalty = tf.where(real_, C_const, real_penalty)
        fake_penalty = tf.where(fake_binary, A_const, fake_penalty)
        fake_penalty = tf.where(fake_class_binary, B_const, fake_penalty)
        fake_penalty = tf.where(fake_, C_const, fake_penalty)
        
        real_penalty = tf.cast(real_penalty, tf.float32)
        fake_penalty = tf.cast(fake_penalty, tf.float32)

        # calculate loss
        predict_real = tf.reduce_sum(real_softmax[:, :NUM_CLASSES], axis=1)
        predict_fake = tf.reduce_sum(fake_softmax[:, NUM_CLASSES:], axis=1)
        discrim_unsupervised_loss = tf.reduce_sum(-(tf.log(predict_real + EPS) + tf.log(predict_fake + EPS)))
        real_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes_real, logits=real_outputs)
        fake_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes_fake, logits=fake_outputs)

        discrim_real_supervised_loss = tf.reduce_sum(tf.multiply(real_cross_entropy, real_penalty))
        discrim_fake_supervised_loss = tf.reduce_sum(tf.multiply(fake_cross_entropy, fake_penalty))
        discrim_loss = tf.add_n([discrim_unsupervised_loss, discrim_real_supervised_loss, discrim_fake_supervised_loss])


main()
