import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp


rs = np.random.RandomState(seed=0)
m = 2000  # Number of data points.
n = 1000  # Number of variables.

A = rs.randn(m, n)
b = rs.randn(m, 1)

xstar = np.linalg.lstsq(A, b, rcond=None)[0]
f_star = (0.5/float(m)) * np.linalg.norm(np.dot(A, xstar) - b) ** 2

A = tf.convert_to_tensor(A, dtype=tf.float32)
b = tf.convert_to_tensor(b, dtype=tf.float32)

# This is a bias vector that will be added to the gradient
grad_bias = 1.0 * tf.nn.l2_normalize(tf.convert_to_tensor(rs.randn(n, 1), dtype=tf.float32))

@tf.function
def loss_and_grad_fun(x):
    residual = tf.matmul(A, x) - b
    loss = 0.5 * tf.norm(residual) ** 2 / float(m)
    
    # The 'gradient' that we observe is a noisy, biased version of the true gradient.
    # This is meant to mimic scenarios where we only have access to biased gradients.
    err = tf.matmul(tf.transpose(A), residual) / float(m)
    grad_noise = 1.5 * tf.nn.l2_normalize(tf.random.normal(shape=(n, 1)))
    gradient = err + (grad_bias + grad_noise) * tf.norm(err)
    
    return loss, gradient

opt = tf.keras.optimizers.SGD(5e-3)

@tf.function
def step_fun(x):
    loss, gradient = loss_and_grad_fun(x)
    opt.apply_gradients([(gradient, x)])
    return loss


x = tf.Variable(tf.zeros((n, 1)), dtype=tf.float32)

# fobj = []
# for _ in range(10000):
#     fobj.append(step_fun(x))
    
# # Store training curve for plotting later.
# f_gd = tf.stack(fobj).numpy().copy()


# Hyperparameters for Vanilla ES
sigma = 0.1
beta = 1.0
learning_rate = 0.2

# Defines the distribution for sampling parameter perturbations.
scale = sigma / np.sqrt(n)
def sample():
    return scale * tf.random.normal(shape=(n, 1), dtype=tf.float32)

opt = tf.keras.optimizers.SGD(learning_rate)

@tf.function
def step_fun(x):
    epsilon = sample()
    
    # We utilize antithetic (positive and negative) samples.
    f_pos, _ = loss_and_grad_fun(x + epsilon)
    f_neg, _ = loss_and_grad_fun(x - epsilon)
    
    # This update is a stochastic finite difference estimate of the true gradient.
    update = (beta / (2 * sigma ** 2)) * (f_pos - f_neg) * epsilon
    opt.apply_gradients([(update, x)])
    
    return loss_and_grad_fun(x)[0]


x = tf.Variable(tf.zeros((n, 1)), dtype=tf.float32)

# Run the optmizer.
# fobj = []
# for _ in range(10000):
#     fobj.append(step_fun(x))

# # Store training curve for plotting later.
# f_ves = tf.stack(fobj).numpy().copy()

mu = 0.01
phi_d = 1
b = 60

opt = tf.keras.optimizers.SGD(0.0002)

@tf.function
def step_fun(x):
    total = tf.zeros(shape=(n, 1))
    for i in range(b):

        u = tf.random.normal(shape=(n, 1), mean=0, stddev=1, dtype=tf.float32)
        # u = scale * tf.random.normal(shape=(n, 1), mean=0, stddev=1, dtype=tf.float32)
        
        # We utilize antithetic (positive and negative) samples.
        f_x, _ = loss_and_grad_fun(x)
        f_pos, _ = loss_and_grad_fun(x + mu * u)

        total += phi_d / mu * (f_pos - f_x) * u
        # total += (beta / (2 * sigma ** 2)) * (f_pos - f_x) * u

    zo_grad = total / b

    opt.apply_gradients([(zo_grad, x)])
    
    return loss_and_grad_fun(x)[0]


x = tf.Variable(tf.zeros((n, 1)), dtype=tf.float32)

# Run the optmizer.
fobj = []
for _ in range(10000):
    fobj.append(step_fun(x))

# Store training curve for plotting later.
zo = tf.stack(fobj).numpy().copy()

class GradientSubspace:
    def __init__(self, k=1, n=1000, alpha=0.5, sigma=0.1):
        self.repl_idx = 0
        self.size = 0
        self.k = k
        self.n = n
        self.alpha = alpha
        self.sigma = sigma

    def store_grad(self, grad):
        if self.size == 0:
            self.grads = grad
        elif self.size < k:
            tf.concat([self.grads, grad], 1)
        else:
            self.grads[:, self.repl_idx] = grad
            self.repl_idx += 1

        return
    
    def sample(self):
        u_full = tf.random.normal(shape=(n,1), dtype=tf.float32)
        if self.size == 0:
            return u_full
        
        u_subspace = tf.random.normal(shape=(k, 1), dtype=tf.float32)
        q, _ = tf.linalg.qr(self.grads)

        # cov = self.sigma ** 2 * (self.alpha / float(self.n)  * tf.eye(self.n) + (1-self.alpha)/self.k * tf.matmul(q, q.T))
        
        a = self.sigma * np.sqrt(self.alpha / float(self.n))
        c = self.sigma * np.sqrt((1. - self.alpha) / float(self.k))

        # mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(shape=(n,1)), covariance_matrix=cov)
        # return mvn.sample()
        return a * u_full + c * tf.matmul(q, u_subspace)

        
# Guided ZO
mu = 0.01
phi_d = 1
b = 60
grad_sub = GradientSubspace(k=3, n=n, alpha=0.5, sigma=0.1)

opt = tf.keras.optimizers.SGD(0.001)

@tf.function
def step_fun(x):
    total = tf.zeros(shape=(n, 1))
    for i in range(b):

        u = grad_sub.sample()
        # u = scale * tf.random.normal(shape=(n, 1), mean=0, stddev=1, dtype=tf.float32)
        
        # We utilize antithetic (positive and negative) samples.
        f_x, _ = loss_and_grad_fun(x)
        f_pos, _ = loss_and_grad_fun(x + mu * u)

        grad = phi_d / mu * (f_pos - f_x) * u
        total += grad
        grad_sub.store_grad(grad)
        # total += (beta / (2 * sigma ** 2)) * (f_pos - f_x) * u

    zo_grad = total / b

    opt.apply_gradients([(zo_grad, x)])
    
    return loss_and_grad_fun(x)[0]


x = tf.Variable(tf.zeros((n, 1)), dtype=tf.float32)

# Run the optmizer.
fobj = []
for _ in range(10000):
    fobj.append(step_fun(x))

# Store training curve for plotting later.
gzo = tf.stack(fobj).numpy().copy()



# Hyperparameters for Guided ES
sigma = 0.1
alpha = 0.5
beta = 1.0
k = 1        # Defines the dimensionality of the low-rank subspace.

# Defines parameters of the distribution for sampling perturbations.
a = sigma * np.sqrt(alpha / float(n))
c = sigma * np.sqrt((1. - alpha) / float(k))



def sample(gradient_subspace):
    epsilon_full = tf.random.normal(shape=(n, 1), dtype=tf.float32)
    epsilon_subspace = tf.random.normal(shape=(k, 1), dtype=tf.float32)

    Q, _ = tf.linalg.qr(gradient_subspace)
    epsilon = a * epsilon_full + c * tf.matmul(Q, epsilon_subspace)
    
    return epsilon

opt = tf.keras.optimizers.SGD(0.2)

@tf.function
def step_fun(x):
    
    # We pass the gradient to our sampling function.
    loss, gradient = loss_and_grad_fun(x)
    epsilon = sample(gradient)
    
    # We utilize antithetic (positive and negative) samples.
    f_pos, _ = loss_and_grad_fun(x + epsilon)
    f_neg, _ = loss_and_grad_fun(x - epsilon)
    
    # This update is a stochastic finite difference estimate of the true gradient.
    update = (beta / (2 * sigma ** 2)) * (f_pos - f_neg) * epsilon
    opt.apply_gradients([(update, x)])

    return loss_and_grad_fun(x)[0]
    # return loss_and_grad_fun


x = tf.Variable(tf.zeros((n, 1)), dtype=tf.float32)

# Run the optmizer.
# fobj = []
# for _ in range(10000):
#     fobj.append(step_fun(x))

# # Store training curve for plotting later.
# f_ges = tf.stack(fobj).numpy().copy()


COLORS = {'ges': '#7570b3', 'ves': '#1b9e77', 'sgdm': '#d95f02', 'zo': "#000000", 'gzo': '#FF2222'}
plt.figure(figsize=(8, 6))
# plt.plot(f_ves - f_star, color=COLORS['ves'], label='Vanilla ES')
# plt.plot(f_gd - f_star, color=COLORS['sgdm'], label='Grad. Descent')
# plt.plot(f_ges - f_star, color=COLORS['ges'], label='Guided ES')
plt.plot(zo - f_star, color=COLORS['zo'], label='ZO')
plt.plot(gzo - f_star, color=COLORS['gzo'], label='Guided ZO')
plt.legend(fontsize=16, loc=0)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Evolutionary Strategies', fontsize=16);
plt.show()