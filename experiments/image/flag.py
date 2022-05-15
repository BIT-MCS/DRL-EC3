import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# map info
flags.DEFINE_integer('image_size', 80, 'the size of image')
flags.DEFINE_integer('image_deepth', 2, 'the deepth of image')
flags.DEFINE_integer('wall_value', -1, 'the value of wall')
flags.DEFINE_integer('wall_width', 4, 'the width of wall')
flags.DEFINE_integer('fill_value', -1, 'the value of FillStation')

flags.DEFINE_integer('map_x', 16, 'the length of x-axis')
flags.DEFINE_integer('map_y', 16, 'the length of y-axis')
