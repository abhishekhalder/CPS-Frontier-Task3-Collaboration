#
# Train and test an ICNN which learns a function 
#   f: (Cache,MemBW,t) \mapsto retired instructions
#
# Specifically, this file focuses on a single path,cache,membw config and attempts to learn the 'average'
# execution profile. Since our ICNN is fully input-convex, it doesn't do a very good job.
#
# Author: Georgiy Antonovich Bondar
# Date  : 07-09-2023
#
"""Implementation of Amos+(2017) input convex neural networks (ICNN)."""
import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Any, Callable, Sequence, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
from absl.testing import absltest
from absl.testing import parameterized
import jax.test_util
from jax import grad, jit, vmap, random
import pandas as pd
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import sklearn.model_selection
import sklearn.preprocessing
from jax.scipy.special import logsumexp
from jax.example_libraries import optimizers
from flax.core.frozen_dict import unfreeze, freeze

import pdb

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  
Array = Any

# dx0=2
dx0=1
# dim_hidden = [2,50,100,100,50,5] # dimensions to be used by ICNN_Jax (see below)
dim_hidden = [1,50,100,100,50,5] # dimensions to be used by ICNN_Jax (see below)
# num_epochs = 150
num_epochs = 10
batch_size = 1
SHUFFLE_BUFFER_SIZE = 100



## BEGIN: Load train and test data - learning a function h: (z1,z2) \mapsto supFun
data_dir = '../../../Data/kbm_sim_profile/'
# data=pd.read_csv('Data_supFunElipsUnion.csv')
data=pd.read_csv(data_dir + 'kbm_sim_31_360_perf_0_345_clean.txt', names=['time','totret','ret','loads','stores','misses'], delim_whitespace=True)
Y=data.pop('ret') 
data.pop('totret')
data.pop('loads')
data.pop('stores')
data.pop('misses')
# X=data.pop('time')
X=data
## pdb.set_trace()
# X=sklearn.preprocessing.normalize(X)
########################################
# data_test=pd.read_csv('Data_supFunElipsTestUnion.csv')
data_test=pd.read_csv(data_dir + 'kbm_sim_31_360_perf_0_346_clean.txt', names=['time','totret','ret','loads','stores','misses'], delim_whitespace=True)
Ytest=data_test.pop('ret') 
data_test.pop('totret')
data_test.pop('loads')
data_test.pop('stores')
data_test.pop('misses')
Xtest = data_test
# Xtest=data_test.pop('time')
# Xtest=sklearn.preprocessing.normalize(Xtest)

validate_fraction=0

tf.config.set_visible_devices([], device_type='GPU')
def split_data(validate_fraction,X,Y): 

    if validate_fraction ==0:

        return tf.convert_to_tensor(X), tf.convert_to_tensor(Y), None, None # changing to a kind of tensor that is understandable for tensor flow

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=validate_fraction, random_state=0) # deterministic split 0
    return(X_train, y_train, X_test, y_test)

my_data_split = split_data(validate_fraction,X,Y)
xtrain=my_data_split[0]
ytrain=my_data_split[1]
train_dataset = tf.data.Dataset.from_tensor_slices((xtrain,ytrain))

my_data_split_test = split_data(validate_fraction,Xtest,Ytest)
xtest=my_data_split_test[0]
ytest=my_data_split_test[1]
test_dataset = tf.data.Dataset.from_tensor_slices((xtest, ytest))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

train_dataset = tf.data.experimental.make_csv_dataset(data_dir + "kbm_sim_31_360_perf_0_[0-9]_clean.txt", batch_size, field_delim=' ', column_names=['time','totret','ret','loads','stores','misses'], select_columns=['time','ret'], num_epochs=num_epochs)
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
train_iter = train_dataset.as_numpy_iterator()
# pdb.set_trace()



## END: Load train and test data - learning a function h: (z1,z2) \mapsto supFun

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

class PositiveDense(nn.Module):
  """A linear transformation using a weight matrix with all entries positive.
  Args:
    dim_hidden: the number of output dim_hidden.
    beta: inverse temperature parameter of the softplus function (default: 1).
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """

  dim_hidden: int
  beta: float = 1.0 
  use_bias: bool = True
  dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Callable[
    [PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

  @nn.compact 
  def __call__(self, inputs): #this is method for the class PositiveDense

    """Applies a linear transformation to inputs along the last dimension.
    Args:
      inputs: The nd-array to be transformed.
    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype) # This function will create arrays on JAX’s default device 
    ##dtype (data-type, optional),  a (array_like) – Input data, in any form that can be converted to an array.
     #This includes lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    kernel = self.param(
      'kernel', self.kernel_init, (inputs.shape[-1], self.dim_hidden))

    scaled_kernel = self.beta * kernel

    kernel = jnp.asarray(
      1 / self.beta * nn.softplus(scaled_kernel), self.dtype)

    y = jax.lax.dot_general(
      inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision)
    
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.dim_hidden,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y
##########################################################################################################
##########################################################################################################
##########################################################################################################
class ICNN_Jax(nn.Module):
  """Input convex neural network (ICNN) architeture.
  Args:
    dim_hidden: sequence specifying size of hidden dimensions. The
      output dimension of the last layer is 1 by default.
    init_std: value of standard deviation of weight initialization method.
    init_fn: choice of initialization method for weight matrices (default:
      `jax.nn.initializers.normal`).
    act_fn: choice of activation function used in network architecture
      (needs to be convex, default: `nn.leaky_relu`).
  """

  dim_hidden: Sequence[int]
  init_std: float = 0.01
  init_fn: Callable = jax.nn.initializers.normal
  act_fn: Callable = nn.leaky_relu

  def setup(self):
    num_hidden = len(self.dim_hidden)

    w_zs = list()

    for i in range(1, num_hidden):
      w_zs.append(PositiveDense(
        self.dim_hidden[i], kernel_init=self.init_fn(self.init_std),
        use_bias=True))

    w_zs.append(PositiveDense(
      1, kernel_init=self.init_fn(self.init_std), use_bias=True))
    self.w_zs = w_zs

    w_xs = list()
    for i in range(num_hidden):
      w_xs.append(nn.Dense(
        self.dim_hidden[i], kernel_init=self.init_fn(self.init_std),
        use_bias=True))
    w_xs.append(nn.Dense(
      1, kernel_init=self.init_fn(self.init_std), use_bias=True))
    self.w_xs = w_xs

  @nn.compact
  def __call__(self, x):
    """Applies ICNN module.
    Args:
      x: jnp.ndarray<float>[batch_size, n_features]: input to the ICNN.
    Returns:
      jnp.ndarray<float>[1]: output of ICNN.
    """
    # print('x#################', x.shape)
    z = self.act_fn(self.w_xs[0](x))
    # z = jnp.multiply(z, z)

    for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
      z = self.act_fn(jnp.add(Wz(z), Wx(x)))
    y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x))

    return jnp.squeeze(y)


## BEGIN: Create and initialize ICNN instance 
icnn=ICNN_Jax(dim_hidden)
Seed=jax.random.PRNGKey(0)
params = icnn.init(Seed, jnp.ones([batch_size, dx0]))['params']
## END: Create and initialize ICNN instance 


# ############################## Auto-batching predictions
batched_predict = vmap(icnn.apply, in_axes=(None, 0))

# ######################################################## Utility and loss function
##########################################################################################################
##########################################################################################################
###################################################
  
def accuracy(params, x, y):

  predicted_class = batched_predict({'params': params}, x)

  y=jnp.asarray(y )
  acuu=jnp.abs(predicted_class - y)
  # accuracy=jnp.linalg.norm(jnp.divide(acuu,y))

  # accuracy=jnp.linalg.norm(acuu)/jnp.linalg.norm(y)
  acuu2=jnp.divide(acuu,jnp.abs(y))
  
  accuracy=jnp.average(acuu2)

  return accuracy

def loss(params, x, y):

  preds = batched_predict({'params': params}, x)
  Lost=jnp.linalg.norm(preds-y)
  
  return Lost


@jit
def update(params, x, y, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    grads = grad(loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    params=get_params(opt_state)

########################################## Project the the Params to positive orthant
    cc=flax.core.frozen_dict.unfreeze(params)

    for i in range(1,len(dim_hidden)+1):
      
      cc['w_xs_'+str(i)]['kernel']=jnp.maximum(cc['w_xs_'+str(i)]['kernel'], 0)

    cc=flax.core.frozen_dict.freeze(cc)
###############################################Finding the gradient of the projection 

    Updated_grads = {}
    for outer_key, outer_value in params.items():
      inner_dict = {}

      for inner_key, inner_value in outer_value.items():
        inner_dict[inner_key] = (params[outer_key][inner_key] - cc[outer_key][inner_key]) /step_size

      Updated_grads[outer_key] = inner_dict

################################################### Using the gradient of the projection to find params
    Updated_grads=flax.core.frozen_dict.freeze(Updated_grads)  
################################################### Projecting the params to the positive orthant again
    opt_state = opt_update(0, Updated_grads, opt_state)
    params=get_params(opt_state)
   
    params=flax.core.frozen_dict.unfreeze(params)
#################################################
    for i in range(1,len(dim_hidden)+1):
      # print('Here',i)
      params['w_xs_'+str(i)]['kernel']=jnp.maximum(params['w_xs_'+str(i)]['kernel'], 0)

    params=flax.core.frozen_dict.freeze(params)

    return params, opt_state

step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size) ## Using ADAM optimizer
opt_state = opt_init(params)
params = get_params(opt_state)
##############################################
xtest=jnp.asarray(xtest.numpy())
ytest=jnp.asarray(ytest.numpy())
xtrain=jnp.asarray(xtrain.numpy())
ytrain=jnp.asarray(ytrain.numpy())

for epoch in range(num_epochs):
  
#  for x, y in train_dataset:
  # for x, y in iterator:
  i = 1;
  for xy in train_iter:
    print(i)
    i = i + 1
    # print(next(train_iter))
    # continue

    # Converting tf array to numpy and then deviceArray (jax)
    x = jnp.asarray(xy['time'])
    y = jnp.asarray(xy['ret'])
    # x=jnp.asarray(x.numpy())
    # y=jnp.asarray(y.numpy())
    # pdb.set_trace()
    params, opt_state= update(params, x, y, opt_state)  
    # params = update(params, x, y)

  # xtest=split_data(validate_fraction)[2]
  # ytest=split_data(validate_fraction)[3]

  train_acc = accuracy(params,xtest ,ytest )
  print("Epoch {}".format(epoch))
  print("Training set accuracy {}".format(train_acc))
  if train_acc < 0.05:
    break


########################################### 
print(xtest.shape)
print(batched_predict({'params': params}, xtrain).shape)
################    
## theta_test=jnp.arctan2(xtrain[:,1],xtrain[:,0])       
# theta_test=jnp.arctan2(xtest[:,1],xtest[:,0])
theta_test= xtest
#################################
fig, ax = plt.subplots()

ax.plot(theta_test, batched_predict({'params': params}, xtest),marker='o',linestyle='None',markersize = 3.0, c='xkcd:red',label=r" NN ")

ax.plot(theta_test, ytest ,marker='o',lw=3, c='xkcd:cobalt',linestyle='None',markersize = 2,label=r"Exact")
ax.legend(loc='upper left', ncol=1, frameon=False,fontsize=16)

ax.set_xlabel(r"$\theta$ ")

ax.set_ylabel(r"$h(\theta)$")
plt.savefig('ICNNEllipsUnions.pdf')

