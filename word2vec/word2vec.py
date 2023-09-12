"""

3 3 3
3 3 3
3 3 3


[
[[1,1,1],[1,1,1]]
[[1,1,1],[1,1,1]]
[[1,1,1],[1,1,1]]
]
"""

import tensorflow as tf

tensor1 = tf.convert_to_tensor([
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3]
])

tensor2 = tf.convert_to_tensor(
    [
        [[1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1]]
    ]
)


print(tf.keras.layers.Dot(axes=[1,2])([tensor1,tensor2]))