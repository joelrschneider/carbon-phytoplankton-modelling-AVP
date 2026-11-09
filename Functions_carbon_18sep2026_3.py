# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:45:32 2026

@author: joels
"""
import numpy as np
import pandas as pd

def temp_dep_var(t_c, sal, tst): 

    
    t_k = t_c + 273.15
    # i =...                   #ionic strength is not initiatied. It is a lot of effort to calculate, so far now I leave it out.
    # iac = (0.5 * np.sqrt(i)) #ionic activity coefficient, I'm leaving it out because i is left out

    # Equilibrium constants carbon K(T,S)
    log_Kw = 148.9802 - (13847.26 / t_k) - (23.6521 * np.log(t_k)) + (-5.977 + (118.67 / t_k) + 1.0495 * np.log(t_k)) * sal**(0.5) - 0.0165 * sal
    Kw = np.exp(log_Kw)

    log_K0 = -60.2409 + (9345.17 / (t_k)) + (23.3585 * np.log(0.01 * t_k)) + sal * (0.023517 - (0.023656 * (0.01 * t_k)) + 0.0047036 * (0.01 * t_k)**2)	
    K0 = np.exp(log_K0)
      
    pK10 = -126.34048 + (6320.813 / t_k) + 19.568224 * np.log(t_k)
    A1 = 13.4191 * sal**0.5 + 0.0331 * sal - 5.33e-5 * sal**2
    B1 = -530.123 * sal**0.5 - 6.103 * sal
    C1 = -2.06950 * sal**0.5
    pK1S = A1 + (B1 / t_k) + C1 * np.log(t_k)
    pK1 = pK10 + pK1S
    K1 = 10**(-pK1)
    pK20 = -90.18333 + (5143.692 / t_k) + 14.613358 * np.log(t_k)
    
    A2 = 21.0894 * sal**0.5 + 0.1248 * sal - 3.687e-4 * sal**2
    B2 = -772.483 * sal**0.5 - 20.051 * sal
    C2 = -3.3336 * sal**0.5
    pK2S = A2 + (B2 / t_k) + C2 * np.log(t_k)
    pK2 = pK20 + pK2S
    K2 = 10**(-pK2)

    # Rate constants carbon chemistry, rates as function of T
    k31 = (np.exp(34.69 - (9252 / t_k)) / 1000) * tst	
    k41 = (np.exp(48.08 - (12720 / t_k)) / 1000000) * tst
    k13 = k31 / K1
    k14 = (k41 * K1) / Kw
    k_co2 = k31
    k_oh = k14

    # Method below are fits? 
    # k_co2 = np.exp(1246.98 - (61900 / t_k) - (183 * np.log(t_k)))
	# k_oh = 4.7E7 * np.exp(-23200 / (R * t_k))

    # Equilibrium constants phosphate
    pKp1 = 2.148 #- iac
    pKp2 = 7.199 #- (3 * iac)
    pKp3 = 12.35 #- (5 * iac)
    
    Kp1 = 10**(-pKp1)
    Kp2 = 10**(-pKp2)
    Kp3 = 10**(-pKp3)
    
    # CaCO3 
    pK_CaCO3 = -171.9065-0.077993*(t_k)+2839.319/(t_k)+71.595*np.log10(t_k)
    K_sp = 10**(pK_CaCO3) # Calcite solubility product

    # Put all output into a dictionary
    c_eq = {"Kw": Kw, "K0": K0, "K1": K1, "K2": K2,"K_sp": K_sp,            # C equilibrium constants
            "k13": k13, "k41": k41, "k_co2":k_co2, "k_oh": k_oh,            # Kinetic rate constants
            "Kp1": Kp1, "Kp2": Kp2, "Kp3": Kp3                              # Phosphate equilibrium constants
                    }
                   
    return c_eq

    
def ph_loop(oldst, H, c_eq):
    """
    Calculates an equilibrium between pH and alkalinity using a loop

    Parameters
    ----------
    - oldst : list
        Contains DIC, Alk, N, P_t, S, B. MUST contain values for DIC ("DIC"), Alkalinity ("Alk"), and phosphorus ("P_t") concentrations.
    - H : float
        Initial hydrogen concentration.
    - c_eq : dictionary
        The output of temp_dep_var(), contains equilibrium constaints.

    Returns
    -------
    - pH_new : float
        Gives an approximation of the current pH-value.
    
    """

    DIC, Alk, P_t = oldst[0], oldst[1], oldst[3]
    Kw, K1, K2 = c_eq["Kw"], c_eq["K1"], c_eq["K2"]
    Kp1, Kp2, Kp3 = c_eq["Kp1"], c_eq["Kp2"], c_eq["Kp3"]
    
    
    d_pH = 1.0  # Initialize difference in pH so that the loop can run
    i = 0       # Initialize iterations
    max_i = 100 # Set max nr of iterations

     # Only when the difference is bigger than 1.0E-6 apparently, we accept the new pH.
    while abs(d_pH) > 1.0E-6 and i<max_i:

        i += 1
        
        pH = -np.log10(H)
        OH = Kw / H
        
        alpha_c = H**2 + H * K1 + K1 * K2
        alpha_0 = H**2 / alpha_c
        alpha_1 = (H * K1) / alpha_c
        alpha_2 = (K1 * K2) / alpha_c
    
        HCO3 = alpha_1 * DIC
        CO3 = alpha_2 * DIC
        
        alpha = H**3 + (Kp1 * H**2) + (Kp1 * Kp2 * H) + (Kp1 * Kp2 * Kp3)
        alpha_p0 = (H**3) / alpha
        alpha_p1 = (Kp1 * H**2) / alpha
        alpha_p2 = (Kp1 * Kp2 * H) / alpha		
        alpha_p3 = (Kp1 * Kp2 * Kp3) / alpha			
    
        H3PO4 = alpha_p0 * P_t
        H2PO4 = alpha_p1 * P_t
        HPO4 = alpha_p2 * P_t
        PO4 = alpha_p3 * P_t
    
        p_01 = H3PO4 + H2PO4
        p_12 = H2PO4 + HPO4
        p_23 = PO4 + HPO4
        
        Alk_t = HCO3 + 2 * CO3 + HPO4 + 2 * PO4 - H3PO4 + OH - H	#// Dickson 1981
        d_Alk_t = Alk - Alk_t
    
        # Buffer intensity
        BI = ((1 / np.log10(np.exp(1.0))) * (H + OH + (DIC * (alpha_1 * (alpha_0 + alpha_2) + 4 * alpha_0 * alpha_2)) + (p_01 * (H / (H + Kp1)) * (Kp1 / (H + Kp1))) + (p_12 * (H / (H + Kp2)) * (Kp2 / (H + Kp2))) + (p_23 * (H / (H + Kp3)) * (Kp3 / (H + Kp3)))))

        if BI == 0:
            raise ZeroDivisionError("Cannot divide by a BI that is equal to 0.")
            
        d_pH = d_Alk_t / BI
        pH_new = pH + d_pH
        H = pow(10, -pH_new)

    if i >= max_i:
        print(f"Warning: PH_loop did not converge after {max_i} iterations. Final d_pH: {d_pH}")
    
    return pH_new
    
def beta_eq(t_c, wind, pH, c_eq, params):
    
    # unpack variables
    Kw, K1, K2 = c_eq["Kw"], c_eq["K1"], c_eq["K2"]
    k_co2, k_oh = c_eq["k_co2"], c_eq["k_oh"]
    A, Ea, R = params["A"], params["Ea"], params["R"]
    
    # Gas transfer velocity
    sc_CO2 = 1742 - (91.24* t_c) + (2.208*t_c**2) - (0.0219*t_c**3)

    if wind <= 4:
        k_600 = 1.0707 * wind + 1.5633

    else:
        k_600 = 1.0707 * 4 + 1.5633

    g_co2 = k_600 * (sc_CO2/600)**(-0.5)
    
    H = 10**(-pH)
    OH = Kw / H
    tau = ((H * H) / (K1 * K2 + K1 * H)) + 1
    t_k = t_c + 273.15
    Dc = A * np.exp(-Ea / (R * t_k))				# temperature dependence of diffusion constant (Jahne et al., 1987)
    z_bl = Dc / g_co2
    reac = k_co2 + (k_oh * OH)
    beta = tau / ((tau - 1) + (np.tanh((((reac * tau) / Dc)**0.5) * z_bl) / ((((reac * tau) / Dc)** 0.5) * z_bl)))

    return beta, g_co2

def concentrations(oldst, c_eq, params, I_in, beta, g_co2, pH, t_c):
    
    z_max, CO2_eq = params["z_max"], params["CO2_eq"]
    
    DIC, Alk, N, P_t, S, B, Ca = oldst[0:7]
    Kw, K1, K2, K_sp = c_eq["Kw"], c_eq["K1"], c_eq["K2"], c_eq["K_sp"]
    D, DIC_in, Alk_in, N_in, P_t_in, S_in, Ca_in = params["D"], params["DIC_in"], params["Alk_in"], params["N_in"], params["P_t_in"], params["S_in"], params["Ca_in"]  # Dilution parameters
    k, K_bg, h_i = params["k"], params["K_bg"], params["h_i"]   # Light parameters
    h_co2, h_hco3, v_c_max, f_c, r_max, mu_max_cons, Q_ten = params["h_co2"], params["h_hco3"], params["v_c_max"], params["f_c"], params["r_max"], params["mu_max_cons"], params["Q_ten"]   # Growth parameters
    cf = params["cf"]
    
    # pH and H, HCO3, and CO3 concentrations
    H = 10**(-pH)
    OH = Kw/H
    
    denominator = (H * H + H * K1 + K1 * K2)
    CO2 = ((H * H) / denominator) * DIC
    HCO3 = ((H * K1) / denominator) * DIC
    CO3 = ((K1 * K2) / denominator) * DIC
    
    CO2_atm = ((g_co2 * beta) / z_max) * (CO2_eq - CO2);

    # Dilution influx
    DIC_dil = D * (DIC_in - DIC)
    Alk_dil = D * (Alk_in - Alk)
    N_dil = D * (N_in - N)
    P_dil = D * (P_t_in - P_t)
    S_dil = D * (S_in - S)
    Ca_dil = D * (Ca_in - Ca)
    
    if I_in == 0:
        I_out = 0
        f_i = 0
    else:
        I_out = I_in * np.exp(-(k * B + K_bg) * z_max)
        f_i = (1 / (np.log(I_in / I_out))) * np.log((h_i + I_in) / (h_i + I_out))

    # C_uptake
    f_co2 = CO2 / (h_co2 + CO2)
    f_hco3 = HCO3 / (h_hco3 + HCO3)

    v_co2 = v_c_max * f_c * f_co2 * f_i
    v_hco3 = v_c_max * (1 - f_c) * f_hco3 * f_i

	# Respiration
    r = r_max    # /* pow(Q_ten, ((t_c-tc_old)/10));*/          
	
	# Growth
    v_c = v_co2 + v_hco3 - r
    mu_max = mu_max_cons * Q_ten**((t_c-20.0)/10)
    mu = mu_max * (v_c / v_c_max)
    
    # Calcification
    #SA_phyto = max(-0.0097 + 59.799917290081886 * B, 0)  # m2/L
    IAP = Ca * CO3
    #calcification = k_f * SA_phyto * (IAP - K_sp)    
    calcification = cf * B * (IAP - K_sp)

    variables = [I_out, f_i, mu, mu_max, v_c, v_c_max, v_co2]
    concentration = [OH, CO2, HCO3, CO3, CO2_atm]
    dil_influx = [DIC_dil, Alk_dil, N_dil, P_dil, S_dil, Ca_dil]
    calc_factors = [calcification, IAP, K_sp]
    
    return  variables, concentration, dil_influx, calc_factors


def derivatives(oldst, t, tst, params, variables, concentration, dil_influx, calc_factors):  #(oldst, t, tst, I_in, t_c, c_eq, wind, params, beta, g_co2, pH)

    # Unpacking parameters
    DIC, Alk, N, P_t, S, B, Ca = oldst[0:7]
    
    D, CO2_min, m = params["D"], params["CO2_min"], params["m"]
    PON_POC, POP_POC, S_C = params["PON_POC"], params["POP_POC"], params["S_C"]
    
    mu, v_c = variables[2], variables[4]
    CO2_atm = concentration[4]
    DIC_dil, Alk_dil, N_dil, P_dil, S_dil, Ca_dil = dil_influx[0], dil_influx[1], dil_influx[2], dil_influx[3], dil_influx[4], dil_influx[5]
    calcification = calc_factors[0]

    # Differential equations
    deriv = [0.0] * 7            # Initialize the derivative list

    # dPOC2 = (v_c * B)/tst
    # deriv[6] = dPOC2

    deriv[0] = (CO2_atm + DIC_dil - (v_c * B) + CO2_min - calcification) / tst   							# dDIC/dt
    deriv[5] = ((mu-D-m) * B ) / tst       													# dB/dt
    d_B = deriv[5]
    deriv[1] = (PON_POC + POP_POC + 2 * S_C) * d_B + ((Alk_dil - 2*calcification)/tst)             # dAlk/dt 
    deriv[2] = (N_dil/ tst) - (PON_POC * d_B)												# dN/dt
    deriv[3] = (P_dil/ tst) - (POP_POC * d_B)												# dPt/dt
    deriv[4] = (S_dil/ tst) - (S_C * d_B) 											# dS/dt
    deriv[6] = (Ca_dil - calcification)/ tst         # dCa/dt

    return deriv


def rk(oldst, t, tst, params, variables, concentration, dil_influx, calc_factors): #I_in, t_c, c_eq, wind, params, beta, g_co2, pH):

    
    nf = 7 # nr of differential equations RK integration
    
    deriv1, deriv2, deriv3, deriv4, calc_deriv, out = [0.0]*nf, [0.0]*nf, [0.0]*nf, [0.0]*nf, [0.0]*nf, [0.0]*nf

    for i in range(nf): 
        calc_deriv[i] = oldst[i] 
    deriv1 = derivatives(calc_deriv, t, tst, params, variables, concentration, dil_influx, calc_factors)

    for i in range(nf):
        calc_deriv[i] = oldst[i] + deriv1[i] /2.0
    deriv2 = derivatives(calc_deriv, t+0.5, tst, params, variables, concentration, dil_influx, calc_factors)

    for i in range(nf):
        calc_deriv[i] = oldst[i] + deriv2[i] /2.0
    deriv3 = derivatives(calc_deriv, t+0.5, tst, params, variables, concentration, dil_influx, calc_factors)

    for i in range(nf):
        calc_deriv[i] = oldst[i] + deriv3[i]
    deriv4 = derivatives(calc_deriv, t+1, tst, params, variables, concentration, dil_influx, calc_factors)

    for i in range(nf):
        out[i] = oldst[i] + (deriv1[i] + 2.0 * deriv2[i] + 2.0 * deriv3[i] + deriv4[i])/6

    return out




def prediction(temp_data, light_data, wind_data, params, oldst, tst, name_list):
    
    data = []
    H = params['H_in']
    
    t_c_in = temp_data[0]
    c_eq_in = temp_dep_var(t_c_in, params['sal'], tst)
    pH_in = -np.log10(H)
    K1, K2 = c_eq_in["K1"], c_eq_in["K2"]
    DIC_in = oldst[0]
    CO2_in  = ((H*H) / (H*H + H*K1 + K1*K2)) * DIC_in
    HCO3_in = ((H*K1) / (H*H + H*K1 + K1*K2)) * DIC_in
    CO3_in  = ((K1*K2) / (H*H + H*K1 + K1*K2)) * DIC_in
    
    data.append({"B": oldst[5], "d_B": 0.0,
                 "CO2": CO2_in, "HCO3": HCO3_in, "CO3": CO3_in, "Ca": oldst[6],
                 "DIC": DIC_in, "pH": pH_in, "Alk": oldst[1], "Alk_dil":0.0,
                 "N": oldst[2], "P_t": oldst[3], "S": oldst[4],
                 "I_in": 0.0, "I_out": 0.0, "f_i": 0.0,
                 "t_c": t_c_in,
                 "mu": 0.0, "mu_max": 0.0, "v_c": 0.0, "v_co2": 0.0,
                 "CO2_atm": 0.0, "CO2_min": params["CO2_min"], "DIC_dil": 0.0,
                })
    
    
    for i in range(len(temp_data)):   # Loop over each value for temperature (daily)
        t_c = temp_data[i]
        CO2_min_flux = 10**(0.0515*t_c + 1.9949)    # Benthic CO2 flux (from Tom de Ruyter, 2017), in mmol/m2/day
        params["CO2_min"] = CO2_min_flux/(1000*1300*24)  # CO2 input from mineralization, converted to mol/L/h. to go from m2 to L, the average lake depth in dm is also used.
    
        c_eq = temp_dep_var(t_c, params['sal'], tst)
        params["CO2_eq"] = params['atm_CO2'] * c_eq["K0"]      # dissolved CO2 in equilibrium with atmosphere, mol/L
        
        m = i * 24
        n = 0
    
        while n < 24:
            I_in = light_data[m+n]   # Loop over each value for light (hourly)
            wind = wind_data[m+n]    # Loop over each value for wind (hourly)
            
            
            for minute in range(60):
                t = (m+n)*60 + minute
                
                pH = ph_loop(oldst, H, c_eq)
                H = 10**(-pH)
                beta, g_co2 = beta_eq(t_c, wind, pH, c_eq, params)
                
                variables, concentration, dil_influx, calc_factors = concentrations(oldst, c_eq, params, I_in, beta, g_co2, pH, t_c)
    
                I_out, f_i, mu, mu_max, v_c, v_c_max, v_co2 = variables
                OH, CO2, HCO3, CO3, CO2_atm = concentration
                DIC_dil, Alk_dil, N_dil, P_dil, S_dil, Ca_dil = dil_influx
                calcification, IAP, K_sp = calc_factors
                
                deriv = derivatives(oldst, t, tst, params, variables, concentration, dil_influx, calc_factors) # Not a necessary step, but just to see the derivatives
                d_B = deriv[5]
                
                output_RK = rk(oldst, t, tst, params, variables, concentration, dil_influx, calc_factors)
                output_RK[1] = max(output_RK[1], 0.0)  # Alk cannot be negative
                output_RK[2] = max(output_RK[2], 0.0)  # N cannot be negative
                output_RK[3] = max(output_RK[3], 0.0)  # P_t cannot be negative
                output_RK[4] = max(output_RK[4], 0.0)  # S cannot be negative
                oldst = output_RK # States are replaced
    
                K1, K2 = c_eq["K1"], c_eq["K2"]
                DIC = oldst[0]
                
                CO2 = ((H * H) / (H * H + H * K1 + K1 * K2)) * DIC
                HCO3 = ((H * K1) / (H * H + H * K1 + K1 * K2)) * DIC
                CO3 = ((K1 * K2) / (H * H + H * K1 + K1 * K2)) * DIC
            
            pH = ph_loop(oldst, H, c_eq) 
            H = 10**(-pH)
            
            # Append all data to a dataframe
            data.append({"B":oldst[5], "d_B":d_B,
                         "CO2":CO2, "HCO3":HCO3, "CO3":CO3, "Ca":oldst[6],
                         "DIC":DIC, "pH":pH, "Alk":oldst[1], "Alk_dil":Alk_dil,
                         "N":oldst[2], "P_t":oldst[3], "S":oldst[4],
                         "I_in":I_in, "I_out":I_out, "f_i": f_i, # "f_co2": f_co2, "f_hco3": f_hco3,
                         "t_c":t_c, 
                         "mu": mu, "mu_max":mu_max, "v_c":v_c, "v_co2":v_co2,
                         "CO2_atm": CO2_atm, "CO2_min":params["CO2_min"], "DIC_dil": DIC_dil,
                         "calcification":calcification, "IAP":IAP, "K_sp":K_sp
                        })
                         
            
            n+=1
            
    df = pd.DataFrame(data, columns = name_list)

    return df