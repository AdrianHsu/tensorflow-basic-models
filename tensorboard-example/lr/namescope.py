import tensorflow as tf

# tf.get_variable(“vname”)方法，在创建变量时，如果这个变量vname已经存在，则，直接使用这个变量
# 如果不存在，则重新创建；而tf.Variable()在创建变量时，一律创建新的变量，
# 如果这个变量已存在，则后缀会增加0、1、2等数字编号予以区别。
# 把 name = '' 拿掉試試看

with tf.name_scope("my_scope"):
	v1 = tf.get_variable("var1", [1], dtype=tf.float32)
	v2 = tf.Variable(1, name="var2", dtype=tf.float32)
	a = tf.add(v1, v2)

print(v1.name)  # var1:0
print(v2.name)  # my_scope/var2:0
print(a.name)   # my_scope/Add:0
# 可以发现，name_scope不会作为tf.get_variable变量的前缀，
# 但是会作为tf.Variable的前缀。

# 要想在get_variable变量名称前面加上scope前缀，
# 需要使用variable_scope，例如：
with tf.variable_scope("my_scope"):
	v3 = tf.get_variable("var3", [1], dtype=tf.float32)
	v4 = tf.Variable(1, name="var4", dtype=tf.float32)
	a2 = tf.add(v3, v4)

print(v3.name)  # my_scope/var3:0
print(v4.name)  # my_scope/var4:0
print(a2.name)   # my_scope/Add:0