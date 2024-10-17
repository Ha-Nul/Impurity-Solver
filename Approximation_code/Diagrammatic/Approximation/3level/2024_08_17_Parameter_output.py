import numpy as np

gamma_input_min, gamma_input_max = map(float,input("Insert gamma min and max value : ").split())
alpha_input_min, alpha_input_max = map(float,input("Insert alpha min and max value : ").split())

gamma_grid = int(input("Set Gamma interval : "))
alpha_grid = int(input("Set alpha interval : "))

gam_arr = np.linspace(gamma_input_min,gamma_input_max,gamma_grid)
alp_arr = np.linspace(alpha_input_min,alpha_input_max,alpha_grid)

with open("Param.txt", "w") as file:
    # gam_arr을 한 줄로 출력
    file.write(" ".join(f"{x:.2f}" for x in gam_arr) + "\n")
    # alp_arr을 한 줄로 출력
    file.write(" ".join(f"{x:.2f}" for x in alp_arr) + "\n")