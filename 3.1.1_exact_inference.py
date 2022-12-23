import numpy as np

def generate_potential(B):
    """Given Beta B, return a 2x2 array containing possible potential values

    Args:
        B (float): Beta constant used in potential function

    Returns:
        2x2 array: Contains all values of potential. Element (i,j) is phi(x1=i, x2=j)
    """
    potentials = np.ones((2,2))
    potentials[0,0] = np.exp(B)
    potentials[1,1] = potentials[0, 0]
    return potentials


def generate_column_states(n):
    """given the dimension n, return an array of all possible states for a binary column of length n

    Args:
        n (int): dimension of column
        
    Return:
        y : binary array of shape 2^n x n
    """
    
    y = np.zeros((pow(2, n), n)).astype(int)
    
    for state_index in range(pow(2, n)):
        state_binary = bin(state_index)[2:]
    
        for bit_index, bit_value in enumerate(reversed(state_binary)):
            y[state_index, bit_index] = bit_value
    
    return y


def f_0(y1, potential):
    """given binary column y of dim n, return the potential of y under f_0

    Args:
        y1 (n-array): n dimensional array (this is the first column in the nxn array)
        potential : 2x2 array of potential
    
    Return:
        message: return the message from this factor
    """
    
    factor = 1
    for row in range(y1.size - 1):
        factor *= potential[y1[row], y1[row+1]]
    
    return factor


def f_0_vectorize(column_states, potential):
    """given an array of all possible column states, return an array of all potentials under f_0
    This is vectorized version of previously defined function f_0

    Args:
        column_states (nxm array): array of m different states of binary column of length n
        potential (2x2 array): all possible values of potential
    """
    m = column_states.shape[0]
    f0 = np.zeros(m)
    
    for y_i in range(m):
        f0[y_i] = f_0(column_states[y_i], potential)
    
    return f0


def f_i(y1, y2, potential):
    """given column 1 and 2 vectors, calculate the shared potential between these two columns

    Args:
        y1 (n-array): column y1, binary array
        y2 (n-array): column y2, binary array
        potential (2x2 array): potential to be applied to all neighbours cells
    
    Return:
        message: return the message from y1 and y2
    """    
    factor = 1
    
    # calculate potentials going down the 2nd column and shared edges between y1 and y2
    for row in range(y1.size-1):
        factor *= potential[y2[row], y2[row+1]]
        factor *= potential[y1[row], y2[row]]
    # above for loop misses out last row - calculate that outside loop
    factor *= potential[y1[-1], y2[-1]]
    
    return factor


def f_i_vectorize(column_states, potential):
    """given an array of all possible column states, return an array of all potentials
    This is a vectorized version of function f_i previously defined

    Args:
        column_states (nxm array): array of m different binary states of length n to calculate the potential of
        potential (2x2 array): 2x2 array containing the possible potential values
    """
    m = column_states.shape[0]
    fi = np.zeros((m,m))
    
    for y_i in range(m):
        for y_j in range(m):
            fi[y_i, y_j] = f_i(column_states[y_i], column_states[y_j], potential)
    
    return fi


def caclulate_desired_prob_dist(n=10, B=1):
    """Given size of lattice (n) and Beta value (B), print the probability
    distribution of the top right and bottom right binary nodes

    Args:
        n (int, optional): Size of nxn lattice. Defaults to 10.
        B (int, optional): Beta value used in Potential. Defaults to 1.
    """
    column_states = generate_column_states(n)
    potential = generate_potential(B)

    # calculate transition potential matrixes of all possible states
    f0 = f_0_vectorize(column_states, potential)
    fi = f_i_vectorize(column_states, potential)

    # calculate log_messages passing through lattice columns
    log_message = np.log(f0)
    log_message_max = log_message.max()
    for col in range(1, n):
        log_message = log_message_max + np.log(np.exp(log_message - log_message_max) @ fi)
        log_message_max = log_message.max()

    # calculate log_Z (total potential of model)
    log_Z = log_message_max + np.log(np.exp(log_message - log_message_max).sum())
    
    # marginalise over all unwanted variables in final column
    log_message_x1_xn = np.zeros((2,2))
    n = column_states.shape[1]
    for state_index, state in enumerate(column_states):
        i = state[0]
        j = state[n-1]
        log_message_x1_xn[i,j] += np.exp(log_message[state_index] - log_message_max)
    log_message_x1_xn = np.log(log_message_x1_xn)
    log_message_x1_xn += log_message_max

    # calculate the final join distribution
    joint_dist = np.exp(log_message_x1_xn - log_Z)

    # print results
    print(f"\nn: {n}, Beta: {B}")
    print(f"log(Z): {log_Z}")
    print(f"joint_dist of top right and bottom left nodes: \n{joint_dist}")
    


# Pull it all together to calculate the desired solution
inputs = np.array([[10, 0.01], [10, 1], [10, 4]])
for input in inputs:
    caclulate_desired_prob_dist(n=int(input[0]), B=input[1])
    print("---------------")
