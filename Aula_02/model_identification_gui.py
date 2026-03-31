import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
from scipy.signal import lti, step
import threading

# ============================================================================
# FUNÇÕES DOS MODELOS (resposta ao degrau)
# ============================================================================

def step_response_fopdt(t, K, tau, theta, y0=0, u_step=1):
    """Resposta ao degrau de um sistema FOPDT"""
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            y[i] = y0 + K * u_step * (1 - np.exp(-(ti - theta) / tau))
    return y

def step_response_sopdt(t, K, tau1, tau2, theta, y0=0, u_step=1):
    """Resposta ao degrau de um sistema SOPDT superamortecido (tau1 != tau2)"""
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            t_ = ti - theta
            y[i] = y0 + K * u_step * (1 - (tau1 * np.exp(-t_/tau1) - tau2 * np.exp(-t_/tau2)) / (tau1 - tau2))
    return y

def step_response_integrator(t, K, tau, theta, y0=0, u_step=1):
    """Sistema integrador com atraso: G(s) = K/(s (tau s + 1)) e^{-theta s}"""
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            t_ = ti - theta
            y[i] = y0 + K * u_step * (t_ - tau * (1 - np.exp(-t_ / tau)))
    return y

def step_response_inverse(t, K, tau, theta, zero_factor, y0=0, u_step=1):
    """Sistema com resposta inversa (zero RHP): G(s) = K (1 - zero_factor*tau s) / (tau s + 1)^2"""
    # Simulação numérica com scipy.signal
    num = [-K * zero_factor * tau, K]
    den = [tau**2, 2*tau, 1]
    sys = lti(num, den)
    t_sim = np.linspace(0, max(t) + theta, 1000)
    _, y_sim = step(sys, T=t_sim)
    y_sim = y_sim * u_step + y0
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            idx = np.argmin(np.abs(t_sim - (ti - theta)))
            y[i] = y_sim[idx]
    return y

# ============================================================================
# FUNÇÕES DOS MODELOS CANDIDATOS (para identificação)
# ============================================================================

def fopdt_model(t, K, tau, theta, y0, u_step):
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            y[i] = y0 + K * u_step * (1 - np.exp(-(ti - theta) / tau))
    return y

def sopdt_model(t, K, tau1, tau2, theta, y0, u_step):
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            t_ = ti - theta
            y[i] = y0 + K * u_step * (1 - (tau1 * np.exp(-t_/tau1) - tau2 * np.exp(-t_/tau2)) / (tau1 - tau2))
    return y

def integrator_model(t, K, tau, theta, y0, u_step):
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            t_ = ti - theta
            y[i] = y0 + K * u_step * (t_ - tau * (1 - np.exp(-t_ / tau)))
    return y

def inverse_model(t, K, tau, theta, zero_factor, y0, u_step):
    # Usa a mesma lógica da simulação numérica
    num = [-K * zero_factor * tau, K]
    den = [tau**2, 2*tau, 1]
    sys = lti(num, den)
    t_sim = np.linspace(0, max(t) + theta, 1000)
    _, y_sim = step(sys, T=t_sim)
    y_sim = y_sim * u_step + y0
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            idx = np.argmin(np.abs(t_sim - (ti - theta)))
            y[i] = y_sim[idx]
    return y

# ============================================================================
# CLASSE PRINCIPAL DA INTERFACE
# ============================================================================

class ModelIdentificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Identificação de Modelos de Processos")
        self.root.geometry("1100x750")
        
        # Variáveis de controle
        self.true_model = tk.StringVar(value="FOPDT")
        self.noise_std = tk.DoubleVar(value=0.05)
        self.y0 = tk.DoubleVar(value=0.0)
        self.u_step = tk.DoubleVar(value=10.0)
        self.t_final = tk.DoubleVar(value=50.0)
        self.Ts = tk.DoubleVar(value=0.5)
        
        # Parâmetros dos modelos (valores padrão)
        self.params_fopdt = [2.0, 5.0, 2.0]  # K, tau, theta
        self.params_sopdt = [2.0, 3.0, 7.0, 1.5]  # K, tau1, tau2, theta
        self.params_integrator = [0.5, 4.0, 1.0]  # K, tau, theta
        self.params_inverse = [2.0, 5.0, 1.0, 0.8]  # K, tau, theta, zero_factor
        
        # Modelos candidatos (checkboxes)
        self.candidate_vars = {
            "FOPDT": tk.BooleanVar(value=True),
            "SOPDT": tk.BooleanVar(value=True),
            "Integrator": tk.BooleanVar(value=False),
            "Inverse": tk.BooleanVar(value=False)
        }
        
        # Dados
        self.t = None
        self.y_true = None
        self.y_meas = None
        self.results = []  # lista de dicionários com resultados
        
        # Criação da interface
        self.create_widgets()
        
        # Geração inicial de dados
        self.generate_data()
        
    def create_widgets(self):
        # Frame principal dividido em esquerda (controles) e direita (gráfico)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame esquerdo (controles)
        left_frame = ttk.LabelFrame(main_frame, text="Configuração", width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)
        
        # Frame direito (gráfico e resultados)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ========== ESQUERDA: Controles ==========
        # Modelo verdadeiro
        ttk.Label(left_frame, text="Modelo Verdadeiro:").pack(anchor=tk.W, pady=(5,0))
        model_combo = ttk.Combobox(left_frame, textvariable=self.true_model, 
                                   values=["FOPDT", "SOPDT", "Integrator", "Inverse"],
                                   state="readonly")
        model_combo.pack(fill=tk.X, pady=2)
        model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # Parâmetros do modelo verdadeiro (dinâmicos)
        self.params_frame = ttk.LabelFrame(left_frame, text="Parâmetros do Modelo Verdadeiro")
        self.params_frame.pack(fill=tk.X, pady=5)
        self.update_params_widgets()
        
        # Configuração do experimento
        ttk.Label(left_frame, text="Configuração do Experimento:").pack(anchor=tk.W, pady=(10,0))
        exp_frame = ttk.Frame(left_frame)
        exp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(exp_frame, text="y0:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(exp_frame, textvariable=self.y0, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(exp_frame, text="u_step:").grid(row=0, column=2, sticky=tk.W, padx=(10,0))
        ttk.Entry(exp_frame, textvariable=self.u_step, width=8).grid(row=0, column=3, padx=5)
        ttk.Label(exp_frame, text="t_final:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(exp_frame, textvariable=self.t_final, width=8).grid(row=1, column=1, padx=5)
        ttk.Label(exp_frame, text="Ts:").grid(row=1, column=2, sticky=tk.W, padx=(10,0))
        ttk.Entry(exp_frame, textvariable=self.Ts, width=8).grid(row=1, column=3, padx=5)
        ttk.Label(exp_frame, text="Ruído (std):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(exp_frame, textvariable=self.noise_std, width=8).grid(row=2, column=1, padx=5)
        
        # Botão gerar dados
        ttk.Button(left_frame, text="Gerar Dados", command=self.generate_data).pack(pady=10)
        
        # Modelos candidatos
        ttk.Label(left_frame, text="Modelos Candidatos:").pack(anchor=tk.W, pady=(10,0))
        cand_frame = ttk.Frame(left_frame)
        cand_frame.pack(fill=tk.X, pady=2)
        for i, (name, var) in enumerate(self.candidate_vars.items()):
            ttk.Checkbutton(cand_frame, text=name, variable=var).grid(row=i//2, column=i%2, sticky=tk.W, padx=5)
        
        # Botão identificar
        ttk.Button(left_frame, text="Identificar Modelos", command=self.identify_models).pack(pady=10)
        
        # Botão limpar resultados
        ttk.Button(left_frame, text="Limpar Resultados", command=self.clear_results).pack(pady=5)
        
        # ========== DIREITA: Gráfico e resultados ==========
        # Figura matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Área de texto para resultados
        self.text_output = tk.Text(right_frame, height=10, state=tk.DISABLED)
        self.text_output.pack(fill=tk.BOTH, expand=True, pady=(10,0))
        
    def update_params_widgets(self):
        # Limpa widgets antigos
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        model = self.true_model.get()
        if model == "FOPDT":
            params = self.params_fopdt
            labels = ["K", "τ", "θ"]
        elif model == "SOPDT":
            params = self.params_sopdt
            labels = ["K", "τ1", "τ2", "θ"]
        elif model == "Integrator":
            params = self.params_integrator
            labels = ["K", "τ", "θ"]
        elif model == "Inverse":
            params = self.params_inverse
            labels = ["K", "τ", "θ", "zero_factor"]
        else:
            return
        
        self.param_entries = []
        for i, (label, val) in enumerate(zip(labels, params)):
            f = ttk.Frame(self.params_frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=label, width=10).pack(side=tk.LEFT)
            entry = ttk.Entry(f, width=10)
            entry.insert(0, str(val))
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.param_entries.append(entry)
    
    def on_model_change(self, event=None):
        self.update_params_widgets()
    
    def get_true_params(self):
        model = self.true_model.get()
        params = []
        for entry in self.param_entries:
            try:
                params.append(float(entry.get()))
            except:
                params.append(0.0)
        return params
    
    def generate_data(self):
        """Gera dados do processo verdadeiro com base nos parâmetros atuais"""
        try:
            # Obter parâmetros
            true_params = self.get_true_params()
            model = self.true_model.get()
            y0 = self.y0.get()
            u_step = self.u_step.get()
            t_final = self.t_final.get()
            Ts = self.Ts.get()
            noise_std = self.noise_std.get()
            
            # Criar vetor de tempo
            self.t = np.arange(0, t_final, Ts)
            
            # Gerar resposta verdadeira
            if model == "FOPDT":
                K, tau, theta = true_params
                self.y_true = step_response_fopdt(self.t, K, tau, theta, y0, u_step)
            elif model == "SOPDT":
                K, tau1, tau2, theta = true_params
                self.y_true = step_response_sopdt(self.t, K, tau1, tau2, theta, y0, u_step)
            elif model == "Integrator":
                K, tau, theta = true_params
                self.y_true = step_response_integrator(self.t, K, tau, theta, y0, u_step)
            elif model == "Inverse":
                K, tau, theta, zero_factor = true_params
                self.y_true = step_response_inverse(self.t, K, tau, theta, zero_factor, y0, u_step)
            else:
                return
            
            # Adicionar ruído
            if noise_std > 0:
                self.y_meas = self.y_true + np.random.normal(0, noise_std * abs(self.y_true[-1] - y0), size=len(self.t))
            else:
                self.y_meas = self.y_true.copy()
            
            # Atualizar gráfico
            self.plot_data()
            
            # Mensagem
            self.append_text(f"Dados gerados para modelo {model}.\n")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao gerar dados: {e}")
    
    def plot_data(self):
        """Plota os dados experimentais e a resposta verdadeira (se disponível)"""
        self.ax.clear()
        if self.y_meas is not None:
            self.ax.plot(self.t, self.y_meas, 'bo', markersize=3, label='Dados experimentais')
        if self.y_true is not None:
            self.ax.plot(self.t, self.y_true, 'k-', linewidth=2, label='Processo real')
        self.ax.set_xlabel('Tempo (min)')
        self.ax.set_ylabel('Variável controlada')
        self.ax.set_title('Dados do processo')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.canvas.draw()
    
    def clear_plot(self):
        """Remove todas as curvas do gráfico (mantém apenas os dados)"""
        self.plot_data()  # Já faz o que precisamos
    
    def identify_models(self):
        """Executa a identificação dos modelos candidatos selecionados"""
        if self.t is None or self.y_meas is None:
            messagebox.showwarning("Aviso", "Gere dados primeiro.")
            return
        
        # Coletar modelos candidatos ativos
        candidates = []
        for name, var in self.candidate_vars.items():
            if var.get():
                candidates.append(name)
        
        if not candidates:
            messagebox.showwarning("Aviso", "Selecione pelo menos um modelo candidato.")
            return
        
        # Parâmetros comuns
        y0 = self.y0.get()
        u_step = self.u_step.get()
        
        self.results = []
        self.append_text("\n--- Iniciando identificação ---\n")
        
        # Executar identificação (pode ser demorado, então usar thread para não travar GUI)
        def run_identification():
            for name in candidates:
                try:
                    if name == "FOPDT":
                        func = lambda t, K, tau, theta: fopdt_model(t, K, tau, theta, y0, u_step)
                        p0 = [1.0, 5.0, 1.0]
                        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
                        popt, _ = curve_fit(func, self.t, self.y_meas, p0=p0, bounds=bounds, maxfev=5000)
                        y_fit = func(self.t, *popt)
                        rmse = np.sqrt(np.mean((self.y_meas - y_fit)**2))
                        self.results.append({
                            'name': name,
                            'params': popt,
                            'rmse': rmse,
                            'y_fit': y_fit
                        })
                        self.append_text(f"{name}: RMSE = {rmse:.4f} | Parâmetros: {popt}\n")
                        
                    elif name == "SOPDT":
                        func = lambda t, K, tau1, tau2, theta: sopdt_model(t, K, tau1, tau2, theta, y0, u_step)
                        p0 = [1.0, 3.0, 7.0, 1.0]
                        bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
                        popt, _ = curve_fit(func, self.t, self.y_meas, p0=p0, bounds=bounds, maxfev=5000)
                        y_fit = func(self.t, *popt)
                        rmse = np.sqrt(np.mean((self.y_meas - y_fit)**2))
                        self.results.append({
                            'name': name,
                            'params': popt,
                            'rmse': rmse,
                            'y_fit': y_fit
                        })
                        self.append_text(f"{name}: RMSE = {rmse:.4f} | Parâmetros: {popt}\n")
                        
                    elif name == "Integrator":
                        func = lambda t, K, tau, theta: integrator_model(t, K, tau, theta, y0, u_step)
                        p0 = [0.5, 4.0, 1.0]
                        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
                        popt, _ = curve_fit(func, self.t, self.y_meas, p0=p0, bounds=bounds, maxfev=5000)
                        y_fit = func(self.t, *popt)
                        rmse = np.sqrt(np.mean((self.y_meas - y_fit)**2))
                        self.results.append({
                            'name': name,
                            'params': popt,
                            'rmse': rmse,
                            'y_fit': y_fit
                        })
                        self.append_text(f"{name}: RMSE = {rmse:.4f} | Parâmetros: {popt}\n")
                        
                    elif name == "Inverse":
                        # Para o modelo inverso, precisamos passar zero_factor
                        func = lambda t, K, tau, theta, zero_factor: inverse_model(t, K, tau, theta, zero_factor, y0, u_step)
                        p0 = [1.0, 5.0, 1.0, 0.5]
                        bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, 1.0])
                        popt, _ = curve_fit(func, self.t, self.y_meas, p0=p0, bounds=bounds, maxfev=5000)
                        y_fit = func(self.t, *popt)
                        rmse = np.sqrt(np.mean((self.y_meas - y_fit)**2))
                        self.results.append({
                            'name': name,
                            'params': popt,
                            'rmse': rmse,
                            'y_fit': y_fit
                        })
                        self.append_text(f"{name}: RMSE = {rmse:.4f} | Parâmetros: {popt}\n")
                        
                except Exception as e:
                    self.append_text(f"Erro no modelo {name}: {e}\n")
            
            self.append_text("--- Identificação concluída ---\n")
            self.plot_results()
        
        # Executar em thread separada para não travar a interface
        threading.Thread(target=run_identification, daemon=True).start()
    
    def plot_results(self):
        """Adiciona as curvas dos modelos identificados ao gráfico existente"""
        self.ax.clear()
        # Plotar dados originais
        self.ax.plot(self.t, self.y_meas, 'bo', markersize=3, label='Dados experimentais')
        self.ax.plot(self.t, self.y_true, 'k-', linewidth=2, label='Processo real')
        # Plotar cada modelo identificado
        colors = ['r', 'g', 'c', 'm', 'orange']
        for i, res in enumerate(self.results):
            self.ax.plot(self.t, res['y_fit'], '--', color=colors[i % len(colors)], 
                         label=f"{res['name']} (RMSE={res['rmse']:.3f})")
        self.ax.set_xlabel('Tempo (min)')
        self.ax.set_ylabel('Variável controlada')
        self.ax.set_title('Identificação de modelos')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.canvas.draw()
    
    def clear_results(self):
        """Limpa os resultados e reexibe apenas os dados"""
        self.results = []
        self.plot_data()
        self.append_text("Resultados limpos.\n")
    
    def append_text(self, text):
        """Adiciona texto à área de saída"""
        self.text_output.config(state=tk.NORMAL)
        self.text_output.insert(tk.END, text)
        self.text_output.see(tk.END)
        self.text_output.config(state=tk.DISABLED)

# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelIdentificationApp(root)
    root.mainloop()