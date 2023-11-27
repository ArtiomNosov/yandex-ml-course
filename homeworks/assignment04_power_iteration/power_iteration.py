import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    n = len(data)
    # Initialize a random vector as an estimate for the dominant eigenvector
    eigenvector = np.random.rand(n)
    
    for _ in range(num_steps):
        # Perform power iteration
        eigenvector = np.dot(data, eigenvector)
        eigenvector /= np.linalg.norm(eigenvector)
    
    # Estimate the dominant eigenvalue
    eigenvalue = np.dot(np.dot(eigenvector, data), eigenvector)
    
    return float(eigenvalue), eigenvector