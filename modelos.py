import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import plotly.express as px

def modelo_siqr(t, z, a, sigma, theta, alpha, gamma):
    S, I, Q, R = z
    N = S + I + Q + R
    return [-a*S*(I+(1-sigma))/(N-sigma*Q), a*S*(I+(1-sigma))/(N-sigma*Q)-(theta+alpha)*I, theta*I-gamma*Q, alpha*I+gamma*Q]

st.beta_set_page_config(
    page_title="Modelos Biomatemáticos",
 	layout="centered",
 	initial_sidebar_state="expanded",
)

st.title('Modelo SIQR')

st.header('Ecuaciones del modelo')

st.latex(r'''
\left\{
\begin{array}{l}
S'= -aS\frac{I+(1-\sigma)Q}{N-\sigma Q},\\
I'= aS\frac{I+(1-\sigma)Q}{N-\sigma Q}-(\theta+\alpha)I,\\
Q'=\theta I-\gamma Q,\\
R'=\alpha I+\gamma Q.
\end{array}
\right.
''')

st.sidebar.header('Parámetros del modelo')

a = st.sidebar.number_input('a', min_value=0.0, max_value=10.0, value=2.0, step=0.1)
sigma = st.sidebar.number_input('sigma', min_value=0.0, max_value=10.0, value=0.1, step=0.1)
theta = st.sidebar.number_input('theta', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
alpha = st.sidebar.number_input('alpha', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
gamma = st.sidebar.number_input('gamma', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

st.sidebar.header('Condiciones iniciales')

S0 = st.sidebar.number_input('S0', min_value=0, max_value=10000000, value=100, step=10)
I0 = st.sidebar.number_input('I0', min_value=0, max_value=10000000, value=20, step=10)
Q0 = st.sidebar.number_input('Q0', min_value=0, max_value=10000000, value=10, step=10)
R0 = st.sidebar.number_input('R0', min_value=0, max_value=10000000, value=0, step=10)

st.sidebar.header('Otras opciones')
t_max = st.sidebar.number_input('t_max', min_value=10, max_value=1000, value=20, step=5)

st.header('Solución del modelo')

sol = solve_ivp(modelo_siqr, [0, t_max], [S0, I0, Q0, R0], args=(a, sigma, theta, alpha, gamma), dense_output=True)
st.write(sol.message)

t = np.arange(0, t_max, 0.01)
Z = sol.sol(t)

fig = go.Figure()
names = ['S','I','Q','R']

for i, z in enumerate(Z):
    fig.add_trace(go.Scatter(
        x=t,
        y=z,
        name=names[i],
        marker_color=px.colors.qualitative.D3[i]
    ))

fig.update_layout(
    title='Soluciones del sistema SIQR',
    xaxis_title='t',
    template='ggplot2',
    height=500
)

st.plotly_chart(fig, use_container_width=True) 