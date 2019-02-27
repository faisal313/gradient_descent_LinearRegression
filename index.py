import numpy as np # numpy is for math operations

def step_gradient(b_current, m_current, points, learningRate):
    # starting points for our gradient
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        # find lowest error by calculating partial derivatives w.r.t --> b and m
        b_gradient += -(2/N) * (y - ((m_current*x) + b_current))
        m_gradient += -(2/N) *x* (y - ((m_current*x) + b_current))
    # update b & m values
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m,learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b,m = step_gradient(b, m, np.array(points), learning_rate)
    return [b,m]

def compute_error(b,m,points):
    totalError = 0
    for i in range(0, len(points)):
        # get the x-value and y-value
        x = points[i,0]
        y = points[i,1]
        # calculate error--> Mean Sqaured Error
        totalError += (y - (m*x+b)) ** 2
    return totalError / float(len(points))

def run():
    points = np.genfromtxt('Data.csv', delimiter=',')
    # Slope of Line --> y = mx + b
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    # Hyperparmeter --> How fast our Model should converge & # of Iterations
    learning_rate = 0.0001
    num_iterations = 1000

    # Train our Model
    print (f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error(initial_b, initial_m, points)}")
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print (f"After {num_iterations} iterations, b = {b}, m = {m}, error = {compute_error(b, m, points)}")

# for executing run function automatically
if __name__ == '__main__':
    run()