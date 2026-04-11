from pybemt.solver import Solver
import math

s = Solver('rotor.ini')
T, Q, P, section_df = s.run()

v_inf = 2.0
eta = T * v_inf / P if P > 0 else 0.0

rho = 1.225
n   = 7200 / 60
D   = 0.24
J   = v_inf / (n * D)
CT  = T  / (rho * n**2 * D**4)
CQ  = Q  / (rho * n**2 * D**5)
CP  = 2 * math.pi * CQ
eta2 = CT / CP * J if CP > 0 else 0.0

print("CLARKY Propeller BEMT Results")
print("="*45)
print("Thrust   T = %.4f N"   % T)
print("Torque   Q = %.6f Nm"  % Q)
print("Power    P = %.4f W"   % P)
print("Efficiency = %.4f (%.2f%%)" % (eta, eta*100))
print("-"*45)
print("Advance ratio J = %.4f" % J)
print("CT = %.6f" % CT)
print("CQ = %.6f" % CQ)
print("CP = %.6f" % CP)
print("="*45)

# 过滤掉异常截面（Re=0 或 |AoA|>30 的截面）
valid = section_df[(section_df['Re'] > 0) & (section_df['AoA'].abs() < 30)]
print("\n有效截面结果（已过滤异常）：")
print(valid.to_string(index=False))

print("\n异常截面（已排除）：")
invalid = section_df[~((section_df['Re'] > 0) & (section_df['AoA'].abs() < 30))]
print(invalid[['radius','AoA','Re','dT','dQ']].to_string(index=False))

valid = section_df[(section_df['Re'] > 0) & (section_df['AoA'].abs() < 30)]
print(valid[['radius','Re','AoA','Cl','Cd']].to_string(index=False))