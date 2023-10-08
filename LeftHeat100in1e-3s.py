# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Set constants
pi = tf.constant(np.pi, dtype=DTYPE)
#viscosity = .01/pi

# Define initial condition
def fun_u_0(x):
    return 0.0*x[:,0] #tf.sin(pi * x[:,0]) * tf.sin(pi * x[:,1])

# Define boundary condition
def fun_u_b(t, x):
    n = x.shape[0]
    return tf.zeros((n,1), dtype=DTYPE)

# Define residual of the PDE
def fun_r(t, x1, u, u_t, u_x1,u_x1x1):
    return u_t - u_x1x1

def fun_nb1(t, u_x1):

    D = 1.0
    timethreshold = 0.5
    flux = 100

#    bccenter1 = (t>=0.0)&(t<timethreshold)
#    bccenter2 = (t>=timethreshold)
    
    #loss_nb = tf.reduce_sum(tf.square(u_x1[bccenter1] + t[bccenter1] * flux * timethreshold))
    #loss_nb += tf.reduce_sum(tf.square(u_x1[bccenter2] + flux))
            
    return u_x1 + t * flux

def fun_nb2(u_x1):
        
    return u_x1
    
def init_model(num_hidden_layers=8, num_neurons_per_layer=40):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    model.add(tf.keras.Input(2))

    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = tf.keras.layers.Lambda(
                #lambda x: 2.0*(x[0] - lb[1])/(ub[1] - lb[1]) - 1.0,2.0*(x[1] - lb[2])/(ub[2] - lb[2]) - 1.0)
                lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
    model.add(scaling_layer)

    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get('tanh'),
            kernel_initializer='glorot_normal'))

    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))
    
    return model
    
def get_r(model, X_r, N_r, N_nb1):
    
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        t, x1 = X_r[:, 0], X_r[:,1]

        # Variables t and x are watched during tape
        # to compute derivatives u_t and u_x
        tape.watch(t)
        tape.watch(x1)

        # Determine residual 
        u = model(tf.stack([t, x1], axis=1))

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        u_x1 = tape.gradient(u, x1)
    
    u_t = tape.gradient(u, t)
    u_x1x1 = tape.gradient(u_x1, x1)
    
    del tape

    return fun_r(t, x1, u, u_t, u_x1, u_x1x1), \
            fun_nb1(t[N_r:N_r+N_nb1], u_x1[N_r:N_r+N_nb1]), fun_nb2(u_x1[N_r+N_nb1:])
            
def compute_loss(model, X_r, X_data, u_data, lossratio, N_r, N_nb1):
    
    # Compute phi^r
    r,r_nb1,r_nb2 = get_r(model, X_r, N_r, N_nb1)
    phi_r = tf.reduce_mean(tf.square(r))
    phi_rnb1 = tf.reduce_mean(tf.square(r_nb1))
    phi_rnb2 = tf.reduce_mean(tf.square(r_nb2))
    
    # Initialize loss
    loss = phi_r
    loss_r = phi_r
    loss += phi_rnb1
    loss_rnb1 = phi_rnb1
    loss += phi_rnb2
    loss_rnb2 = phi_rnb2    
    
    # Add phi^0 & phi^b to the loss
    u_pred = model(X_data[0])
    loss += tf.reduce_mean(tf.square(u_data - u_pred))
    loss_0 = tf.reduce_mean(tf.square(u_data - u_pred))
    
    #u_pred = model(X_data[1])
    #loss += tf.reduce_mean(tf.square(u_data[1] - u_pred))
    #loss_db = tf.reduce_mean(tf.square(u_data[1] - u_pred))
    loss_db = 0.0
    
    return loss,loss_r,loss_rnb1,loss_rnb2,loss_0,loss_db
    
def get_grad(model, X_r, X_data, u_data, lossratio, N_r, N_nb1):
    
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        tape.watch(model.trainable_variables)
        loss,loss_r,loss_rnb1,loss_rnb2,loss_0,loss_db = compute_loss(model, X_r, X_data, u_data, lossratio, N_r, N_nb1)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss,g,loss_r,loss_rnb1,loss_rnb2,loss_0,loss_db

# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step():
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta,loss_r,loss_rnb1,loss_rnb2,loss_0,loss_db = get_grad(model, X_r, X_data, u_data, lossratio, N_r, N_nb1)
    
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))
    
    return loss,loss_r,loss_rnb1,loss_rnb2,loss_0,loss_db
 
# Set number of data points
N_0 = 10000
N_db = 100
N_nb1 = 100
N_nb2 = 100
N_r = 10000

# Set boundary
tmin = 0.
tmax = 1e-3

x1min = 0.
x1max = 1.

x2min = 0.
x2max = 1.

# Lower bounds
lb = tf.constant([tmin, x1min], dtype=DTYPE)
# Upper bounds
ub = tf.constant([tmax, x1max], dtype=DTYPE)

# Set random seed for reproducible results
tf.random.set_seed(0)

# Draw uniform sample points for initial boundary data
t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
x_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype=DTYPE)
X_0 = tf.concat([t_0, x_0], axis=1)

# Evaluate intitial condition at x_0
u_0 = fun_u_0(x_0)

#t_db = tf.random.uniform((N_db,1), lb[0], ub[0], dtype=DTYPE)
#x_db = tf.ones((N_db,1), dtype=DTYPE)*ub[1]
#X_db = tf.concat([t_db, x_db], axis=1)
#u_db = fun_u_b(t_db, x_db)

t_nb1 = tf.random.uniform((N_nb1,1), lb[0], ub[0], dtype=DTYPE)
x_nb1 = tf.ones((N_nb1,1), dtype=DTYPE)*lb[1]
X_nb1 = tf.concat([t_nb1, x_nb1], axis=1)

t_nb2 = tf.random.uniform((N_nb2,1), lb[0], ub[0], dtype=DTYPE)
x_nb2 = tf.ones((N_nb2,1), dtype=DTYPE)*ub[1]
X_nb2 = tf.concat([t_nb2, x_nb2], axis=1)

t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
X_r = tf.concat([tf.concat([t_r, x_r], axis=1),X_nb1, X_nb2],axis=0)

# Collect boundary and inital data in lists
#X_data = [X_0, X_db, X_nb]
#u_data = [u_0, u_db]

X_data = [X_0, X_nb1, X_nb2]
u_data = u_0

# Initialize model aka u_\theta
model = init_model()

# We choose a piecewise decay of the learning rate, i.e., the
# step size in the gradient descent type algorithm
# the first 1000 steps use a learning rate of 0.01
# from 1000 - 3000: learning rate = 0.001
# from 3000 onwards: learning rate = 0.0005

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])

# Choose the optimizer, LBFGS
optim = tf.keras.optimizers.Adam(learning_rate=lr)


# Number of training epochs
N = 3000
lossratio = 1
hist = []
#nbcini = tf.Variable(tf.zeros((N_nb,1), dtype=DTYPE))

# Start timer
t0 = time()

for i in range(N+1):
    
    loss,loss_r,loss_rnb1,loss_rnb2,loss_0,loss_db = train_step()
    
    # Append current loss to hist
    hist.append(loss.numpy())
    
    # Output current loss after 50 iterates
    if i%50 == 0:
        print('It {:05d}: total loss = {:10.3e}'.format(i,loss))
        print('domain loss = {:10.3e}, nbc1 loss = {:10.3e}, nbc2 loss = {:10.3e}, initial loss = {:10.3e}, dbc loss = {:10.3e}'\
              .format(loss_r,loss_rnb1,loss_rnb2,loss_0,loss_db))
        
# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))

model.save('saved_model/1DHeatTransferNaturalBC-flux100t')

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
ax.semilogy(range(len(hist)), hist,'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi_{n_{epoch}}$');

N = 100
time = 1.0
x1space = np.linspace(lb[1], ub[1], N + 1)
X1 = np.meshgrid(x1space)
Xgrid = np.vstack([np.ones(((N+1),), dtype=DTYPE) * time, X1]).T
upred = model(tf.cast(Xgrid,DTYPE))
# Reshape upred
U = upred.numpy().reshape(N+1)

# Surface plot of solution u(t,x)
fig = plt.figure(figsize=(9,4))
ax = fig.add_subplot(111)
ax.plot(X1[0], U);
ax.set_xlabel('$x$')
ax.set_xlabel('$T$')
ax.set_title('Solution of T_t = T_xx @ t = 1')

model.save('saved_model/1DHeatTransferNaturalBC-flux100t')