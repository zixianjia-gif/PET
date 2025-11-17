# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 13:25:01 2025

@author: Lenovo
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint

# ==========================================
# 1. 后端核心：动力学模型 (The "Brain")
# ==========================================

def get_k(T_kelvin, A, Ea):
    """Arrhenius方程计算速率常数"""
    R = 8.314  # J/(mol*K)
    return A * np.exp(-Ea / (R * T_kelvin))

def reaction_model(y, t, T_kelvin, params):
    """
    定义微分方程组
    y: [C_PET, C_EG, C_Oligomer, C_BHET]
    """
    C_PET, C_EG, C_Oligomer, C_BHET = y
    A1, Ea1, A2, Ea2 = params
    
    # 计算速率常数 k
    k1 = get_k(T_kelvin, A1, Ea1)
    k2 = get_k(T_kelvin, A2, Ea2)
    
    # 反应速率 (简化假设：一级反应)
    # Step 1: PET + EG -> Oligomer
    r1 = k1 * C_PET * C_EG
    # Step 2: Oligomer + EG -> BHET
    r2 = k2 * C_Oligomer * C_EG
    
    # 质量守恒 (dC/dt)
    dPET_dt = -r1
    dEG_dt  = -r1 - r2  # 假设每步消耗1分子EG
    dOli_dt = r1 - r2
    dBHET_dt= r2
    
    return [dPET_dt, dEG_dt, dOli_dt, dBHET_dt]

def adjust_params_by_catalyst(base_params, cat_amount):
    """
    这就是您报错缺失的函数。
    它的作用是将'催化剂用量'转化为'动力学参数的变化'。
    
    假设：催化剂主要提高指前因子 A (碰撞频率/活性位点增加)
    简单线性假设：A_new = A_base * (1 + 催化剂系数 * 用量)
    """
    base_A1, base_Ea1, base_A2, base_Ea2 = base_params
    
    # 设定一个增益系数，比如催化剂每增加1%，速率常数翻倍(仅作演示)
    efficiency_factor = 2.0 
    
    # 修正 A1 和 A2
    new_A1 = base_A1 * (1 + cat_amount * efficiency_factor)
    new_A2 = base_A2 * (1 + cat_amount * efficiency_factor)
    
    return (new_A1, base_Ea1, new_A2, base_Ea2)

# ==========================================
# 2. 前端界面：Streamlit App (The "Face")
# ==========================================

st.set_page_config(page_title="中石化大连院PET解聚数字孪生", layout="wide")

st.title("🏭 大连院PET醇解工艺·仿真器")
st.markdown("基于 **Python + 机理模型** 的虚拟反应工厂。")

# --- 左侧控制台 ---
st.sidebar.header("🎛️ 工艺参数设置")

# 1. 温度控制
T_celsius = st.sidebar.slider("反应温度 (°C)", min_value=160, max_value=260, value=196)
T_kelvin = T_celsius + 273.15

# 2. 配方控制
mol_ratio = st.sidebar.slider("醇酯比 (EG:PET)", min_value=1.0, max_value=10.0, value=4.0)
cat_percent = st.sidebar.number_input("催化剂用量 (wt%)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

# 3. 反应时间
run_time = st.sidebar.slider("反应时间 (min)", 30, 300, 180)

st.sidebar.markdown("---")
st.sidebar.info("调整滑块，右侧曲线将实时重算。")

# --- 中间计算逻辑 ---

# 初始条件：假设 PET 初始浓度为 1.0 mol/L
C0_PET = 1.0
C0_EG = C0_PET * mol_ratio 
Initial_State = [C0_PET, C0_EG, 0.0, 0.0] # [PET, EG, Oli, BHET]

# 基础动力学参数 (需根据实验拟合，这里是虚拟值)
# A1, Ea1 (PET->Oli), A2, Ea2 (Oli->BHET)
Base_Params = (1e6, 80000, 5e5, 85000) 

# *** 关键修复点：调用函数调整参数 ***
Real_Params = adjust_params_by_catalyst(Base_Params, cat_percent)

# 定义时间网格
t_grid = np.linspace(0, run_time, 100)

# 求解微分方程
solution = odeint(reaction_model, Initial_State, t_grid, args=(T_kelvin, Real_Params))

# --- 右侧结果展示 ---

# 将结果转换为 Pandas DataFrame 以便绘图
df_result = pd.DataFrame(solution, columns=["PET (原料)", "EG (溶剂)", "Oligomer (低聚物)", "BHET (产物)"])
df_result["Time (min)"] = t_grid
df_result.set_index("Time (min)", inplace=True)

# 布局：两列
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 反应动力学曲线")
    st.line_chart(df_result[["PET (原料)", "BHET (产物)", "Oligomer (低聚物)"]])

with col2:
    st.subheader("📊 最终结果预测")
    final_bhet = solution[-1, 3]
    final_conversion = (1.0 - solution[-1, 0] / C0_PET) * 100
    
    st.metric(label="BHET 最终浓度", value=f"{final_bhet:.3f} mol/L")
    st.metric(label="PET 转化率", value=f"{final_conversion:.1f} %")
    
    st.write("---")
    st.write("**当前动力学参数估算：**")
    st.code(f"k1 = {get_k(T_kelvin, Real_Params[0], Real_Params[1]):.4f}\nk2 = {get_k(T_kelvin, Real_Params[2], Real_Params[3]):.4f}")

# --- 底部说明 ---
st.markdown("---")
st.caption("注：此模型为简化的一级连串反应模型 (PET -> Oligomer -> BHET)。实际工程中需结合您的实验数据修正 Arrhenius 参数 (A, Ea)。")



#streamlit run pet_twin.py