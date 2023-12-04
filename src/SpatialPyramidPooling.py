from keras.layers import Layer
import tensorflow as tf


class SpatialPyramidPooling(Layer):
    def __init__(self, pool_list, **kwargs):
        self.num_channels = None
        self.pool_list = pool_list
        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_channels = input_shape[3]

    def call(self, inputs, **kwargs):
        output = []
        for size in self.pool_list:
            h = tf.shape(inputs)[1]
            w = tf.shape(inputs)[2]

            # 将 h 和 w 转换为浮点数
            h_float = tf.cast(h, tf.float32)
            w_float = tf.cast(w, tf.float32)
            size_float = tf.cast(size, tf.float32)

            new_h = tf.cast(tf.math.ceil(h_float / size_float), tf.int32)
            new_w = tf.cast(tf.math.ceil(w_float / size_float), tf.int32)

            for i in range(size):
                for j in range(size):
                    start_h = i * new_h
                    end_h = tf.minimum((i + 1) * new_h, h)
                    start_w = j * new_w
                    end_w = tf.minimum((j + 1) * new_w, w)

                    # Crop and resize
                    inputs_crop = inputs[:, start_h:end_h, start_w:end_w, :]
                    resized = tf.image.resize(inputs_crop, (new_h, new_w))
                    pooled = tf.reduce_mean(resized, axis=[1, 2])

                    # Add an extra dimension to match the 4D requirement
                    pooled = tf.expand_dims(pooled, 1)
                    pooled = tf.expand_dims(pooled, 2)
                    output.append(pooled)

        # Concatenate along the channels axis
        output = tf.concat(output, axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        pool_areas = sum([i * i for i in self.pool_list])
        return input_shape[0], 1, 1, self.num_channels * pool_areas
