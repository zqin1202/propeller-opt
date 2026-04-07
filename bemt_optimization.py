import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Input parameters
V = 2.0  # Forward speed (m/s)
T_target = 6.86  # Target thrust (N)
R = 0.12  # Propeller radius (m)
R_h = 0.012  # Hub radius (m)
rpm = 7200  # Rotational speed (rpm)
N_b = 3  # Number of blades
rho = 1.225  # Air density (kg/m^3)
c_m = 0.21  # Max dimensionless chord
beta = 50.0  # Distribution coefficient
m = 0.5  # Max chord location
alpha_max_deg = 5.0  # Optimal angle of attack (degrees)

# Convert units
Omega = rpm * 2 * np.pi / 60  # Rotational speed (rad/s)
alpha_max_rad = np.deg2rad(alpha_max_deg)  # Optimal angle of attack (radians)

# 2. Core mathematical formulas
def prandtl_tip_loss(r, R, N_b, Omega, V):
    """
    Calculates the Prandtl's tip-loss function.
    """
    f = (N_b / 2) * (1 - r / R) * np.sqrt(1 + (Omega * R / V)**2)
    # Clamp the argument of arccos to avoid numerical errors
    exp_f = np.exp(-f)
    # Ensure the argument is within the valid range [-1, 1]
    arg = np.clip(exp_f, -1.0, 1.0)
    return (2 / np.pi) * np.arccos(arg)

def thrust_integrand(r, K, V, Omega, R, N_b):
    """
    The integrand for the thrust equation.
    """
    k_p = prandtl_tip_loss(r, R, N_b, Omega, V)
    # Avoid division by zero at r=0, although integration starts from R_h
    if r == 0:
        return 0
    k_1 = K / (1 + (V / (Omega * r))**2 * (1 + K)**2)
    return (k_1 + k_1**2) * k_p * r

def thrust_equation(K, T_target, rho, V, Omega, R, R_h, N_b):
    """
    The full thrust equation that we need to find the root for.
    The function returns the difference between the calculated thrust integral
    and the target thrust value.
    """
    target_integral_val = T_target / (4 * np.pi * rho * V**2)
    
    # Perform the integration
    calculated_integral, _ = quad(
        thrust_integrand, R_h, R, args=(K, V, Omega, R, N_b)
    )
    
    return calculated_integral - target_integral_val

# Solve for the Lagrange multiplier K
# We provide a bracket [0.001, 1.0] for the root finding algorithm.
try:
    sol = root_scalar(
        thrust_equation, 
        args=(T_target, rho, V, Omega, R, R_h, N_b),
        bracket=[0.001, 1.0], 
        method='brentq'
    )
    K = sol.root
    print(f"Successfully solved for K: {K:.6f}")
except (ValueError, RuntimeError) as e:
    print(f"Error solving for K: {e}")
    K = None

if K is not None:
    # 3. Geometric parameter calculation
    # Discretize the blade into 20 nodes
    r_nodes = np.linspace(R_h, R, 20)
    
    # Dimensionless coordinate
    xi = r_nodes / R
    
    # Calculate chord distribution
    b_xi = c_m * beta**(-(xi - m)**2)
    c = b_xi * R
    
    # Calculate airflow deflection angle (delta)
    delta = np.arctan((V / (Omega * r_nodes)) * (1 + K))
    
    # Calculate final pitch angle (theta)
    theta_rad = delta + alpha_max_rad
    theta_deg = np.rad2deg(theta_rad)
    
    # 4. Output: Plotting the results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Chord vs Radius
    ax1.plot(r_nodes, c, 'o-', label='Chord Distribution')
    ax1.set_xlabel('Radius r (m)')
    ax1.set_ylabel('Chord c (m)')
    ax1.set_title('Chord Distribution along the Blade')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Pitch Angle vs Radius
    ax2.plot(r_nodes, theta_deg, 'o-', color='r', label='Pitch Angle Distribution')
    ax2.set_xlabel('Radius r (m)')
    ax2.set_ylabel('Pitch Angle θ (degrees)')
    ax2.set_title('Pitch Angle (Twist) along the Blade')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('propeller_design.png')
    print("Script finished successfully. Plot saved to propeller_design.png")
