import jax.numpy as jnp
from jax import lax
import jax

operand = jnp.expand_dims(jnp.zeros((2, 5)), axis=0)
# operand = jnp.zeros((2, 5))

# 定义scatter的维度对应关系
dimension_numbers = lax.ScatterDimensionNumbers(
  update_window_dims=(),
  inserted_window_dims=(0,1,2,),
  scatter_dims_to_operand_dims=(1,),
)

# 初始化数组B和C
scatter_indices = jnp.expand_dims(jnp.array([[3], [1]]), axis=0)
# scatter_indices = jnp.array([[3], [1]])
updates = jnp.expand_dims(jnp.array([[2.], [4.]]), axis=0).squeeze(-1)
# updates = jnp.array([[2.], [4.]])

print('operand.shape: %s' % str(operand.shape))
print('scatter_indices.shape: %s' % str(scatter_indices.shape))
print('updates.shape: %s' % str(updates.shape))

operand_shape = operand.shape
offset = jnp.arange(0, operand.shape[0]*operand.shape[1]) * operand.shape[2]
operand = jnp.ravel(operand)
scatter_indices = jnp.ravel(scatter_indices) + offset
updates = jnp.ravel(updates)
operand = jax.ops.index_update(operand, scatter_indices, updates)
# operand[scatter_indices] = updates
operand = operand.reshape(operand_shape)

# 对A进行scatter操作
# operand = lax.scatter(operand, scatter_indices, updates, dimension_numbers)

# 输出结果
print(operand)
# [[0 0 0 2 0]
#  [0 4 0 0 0]]
