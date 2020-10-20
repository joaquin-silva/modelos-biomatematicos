import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import plotly.express as px

def modelo_sird(t, z, alpha, beta, gamma):
    S, I, R, D = z
    N = S + I + R + D
    return [-beta*S*I/N, beta*S*I/N-(gamma+alpha)*I, gamma*I, alpha*I]

def main():
    st.title('Modelo SIR-D')

    st.header('Ecuaciones del modelo')
    st.latex(r'''
    \left\{
    \begin{array}{l}
    S'= -\frac{\beta S I}{N}\\
    I'= \frac{\beta S I}{N}-(\gamma+\alpha)I\\
    R'= \gamma I\\
    R'=\alpha I
    \end{array}
    \right.
    ''')

    st.sidebar.header('Parámetros del modelo')
    alpha = st.sidebar.number_input('alpha', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    beta = st.sidebar.number_input('alpha', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    gamma = st.sidebar.number_input('gamma', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    st.sidebar.header('Condiciones iniciales')
    S0 = st.sidebar.number_input('S0', min_value=0, max_value=10000000, value=80, step=10)
    I0 = st.sidebar.number_input('I0', min_value=0, max_value=10000000, value=10, step=10)
    R0 = st.sidebar.number_input('Q0', min_value=0, max_value=10000000, value=10, step=10)
    D0 = st.sidebar.number_input('R0', min_value=0, max_value=10000000, value=0, step=10)

    st.sidebar.header('Otras opciones')
    t_max = st.sidebar.number_input('t_max', min_value=10, max_value=1000, value=20, step=5)

    st.header('Solución del modelo')
    sol = solve_ivp(modelo_sird, [0, t_max], [S0, I0, R0, D0], args=(alpha, beta, gamma), dense_output=True)
    st.write(sol.message)

    t = np.arange(0, t_max, 0.01)
    Z = sol.sol(t)

    fig = go.Figure()
    names = ['S','I','R','D']
    for i, z in enumerate(Z):
        fig.add_trace(go.Scatter(
            x=t,
            y=z,
            name=names[i],
            marker_color=px.colors.qualitative.D3[i]
        ))

    fig.update_layout(
        title='Soluciones del sistema SIR-D',
        xaxis_title='t',
        template='ggplot2',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True) 