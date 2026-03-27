import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

class PIDSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Simulação de Controle PID em Processo com Atraso (FOPDT)")
        self.root.geometry("1200x800")
        
        # Variáveis de parâmetros
        self.Kp_var = tk.DoubleVar(value=2.0)
        self.tau_var = tk.DoubleVar(value=5.0)
        self.theta_var = tk.DoubleVar(value=2.0)
        self.tune_method_var = tk.StringVar(value="Manual")
        self.Kc_var = tk.DoubleVar(value=1.2)
        self.Ti_var = tk.DoubleVar(value=4.0)
        self.Td_var = tk.DoubleVar(value=1.0)
        
        # Variáveis para resultados
        self.results = {}
        
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para parâmetros (esquerda)
        param_frame = ttk.LabelFrame(main_frame, text="🏭 Parâmetros do Processo (FOPDT)")
        param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Parâmetros do processo
        ttk.Label(param_frame, text="Ganho estático (Kp):").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(param_frame, textvariable=self.Kp_var, width=10).grid(row=0, column=1, pady=2)
        
        ttk.Label(param_frame, text="Constante de tempo (τ) [min]:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(param_frame, textvariable=self.tau_var, width=10).grid(row=1, column=1, pady=2)
        
        ttk.Label(param_frame, text="Atraso de transporte (θ) [min]:").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(param_frame, textvariable=self.theta_var, width=10).grid(row=2, column=1, pady=2)
        
        # Método de sintonia
        ttk.Label(param_frame, text="Método de sintonia:").grid(row=3, column=0, sticky='w', pady=5)
        ttk.Radiobutton(param_frame, text="Manual", variable=self.tune_method_var, value="Manual").grid(row=4, column=0, sticky='w')
        ttk.Radiobutton(param_frame, text="Ziegler-Nichols (malha aberta)", variable=self.tune_method_var, value="ZN").grid(row=5, column=0, sticky='w')
        
        # Parâmetros do controlador (apenas se manual)
        self.controller_frame = ttk.LabelFrame(param_frame, text="🎛️ Parâmetros do PID")
        self.controller_frame.grid(row=6, column=0, columnspan=2, sticky='ew', pady=5)
        
        ttk.Label(self.controller_frame, text="Ganho proporcional (Kc):").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(self.controller_frame, textvariable=self.Kc_var, width=10).grid(row=0, column=1, pady=2)
        
        ttk.Label(self.controller_frame, text="Tempo integral (Ti) [min]:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(self.controller_frame, textvariable=self.Ti_var, width=10).grid(row=1, column=1, pady=2)
        
        ttk.Label(self.controller_frame, text="Tempo derivativo (Td) [min]:").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(self.controller_frame, textvariable=self.Td_var, width=10).grid(row=2, column=1, pady=2)
        
        # Botão de simular
        simulate_btn = ttk.Button(param_frame, text="▶️ Simular", command=self.simulate)
        simulate_btn.grid(row=7, column=0, columnspan=2, pady=10)
        
        # Frame para resultados (direita)
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Abas para resultados
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Aba da resposta ao degrau
        self.step_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.step_frame, text="📈 Resposta ao Degrau")
        
        # Aba do diagrama de Bode
        self.bode_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bode_frame, text="📊 Diagrama de Bode")
        
        # Aba do diagrama de Nyquist
        self.nyquist_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.nyquist_frame, text="🌀 Diagrama de Nyquist")
        
        # Aba do lugar das raízes
        self.rlocus_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.rlocus_frame, text="📍 Lugar das Raízes")
        
        # Frame para estatísticas
        self.stats_frame = ttk.LabelFrame(result_frame, text="📈 Análise da resposta ao degrau")
        self.stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.overshoot_label = ttk.Label(self.stats_frame, text="Overshoot: ")
        self.overshoot_label.pack()
        
        self.settling_label = ttk.Label(self.stats_frame, text="Tempo de acomodação (5%): ")
        self.settling_label.pack()
        
        # Frame para parâmetros utilizados
        self.params_frame = ttk.LabelFrame(result_frame, text="🔧 Parâmetros utilizados")
        self.params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.params_text = tk.Text(self.params_frame, height=5, wrap=tk.WORD)
        self.params_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame para avisos
        self.warning_frame = ttk.LabelFrame(result_frame, text="⚠️ Avisos")
        self.warning_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.warning_text = tk.Text(self.warning_frame, height=3, wrap=tk.WORD)
        self.warning_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Atualiza a interface quando o método de sintonia muda
        self.tune_method_var.trace('w', self.update_controller_frame)
        
        # Inicializa o frame de controlador
        self.update_controller_frame()
        
    def update_controller_frame(self, *args):
        if self.tune_method_var.get() == "Manual":
            self.controller_frame.grid()
        else:
            self.controller_frame.grid_remove()
            
    def calculate_pid_params(self):
        Kp = self.Kp_var.get()
        tau = self.tau_var.get()
        theta = self.theta_var.get()
        
        if self.tune_method_var.get() == "ZN":
            if theta > 0:
                Kc_zn = (1.2 / Kp) * (tau / theta)
                Ti_zn = 2.0 * theta
                Td_zn = 0.5 * theta
            else:
                Kc_zn = 1.0
                Ti_zn = 1.0
                Td_zn = 0.0
            
            self.Kc_var.set(Kc_zn)
            self.Ti_var.set(Ti_zn)
            self.Td_var.set(Td_zn)
            
        return self.Kc_var.get(), self.Ti_var.get(), self.Td_var.get()
    
    def simulate(self):
        # Executa a simulação em uma thread separada para não travar a interface
        thread = threading.Thread(target=self._run_simulation)
        thread.daemon = True
        thread.start()
        
    def _run_simulation(self):
        try:
            # Obter parâmetros
            Kp = self.Kp_var.get()
            tau = self.tau_var.get()
            theta = self.theta_var.get()
            
            Kc, Ti, Td = self.calculate_pid_params()
            
            # Simular resposta ao degrau
            t_sim = np.linspace(0, 50, 1000)
            
            # Modelo FOPDT com atraso (simulação simplificada)
            # Usando uma aproximação de primeira ordem com atraso
            def process_response(t):
                if t < theta:
                    return 0
                else:
                    t_delayed = t - theta
                    return Kp * (1 - np.exp(-t_delayed / tau))
            
            y_out = np.array([process_response(t) for t in t_sim])
            
            # Calcular overshoot e tempo de acomodação
            overshoot = (max(y_out) - 1) * 100 if max(y_out) > 1 else 0
            idx_settling = np.where(np.abs(y_out - 1) <= 0.05)[0]
            settling_time = t_sim[idx_settling[0]] if len(idx_settling) > 0 else np.nan
            
            # Atualizar interface na thread principal
            self.root.after(0, self.update_results, t_sim, y_out, overshoot, settling_time, Kp, tau, theta, Kc, Ti, Td)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na simulação: {str(e)}")
    
    def update_results(self, t_sim, y_out, overshoot, settling_time, Kp, tau, theta, Kc, Ti, Td):
        # Limpar frames anteriores
        for widget in self.step_frame.winfo_children():
            widget.destroy()
        for widget in self.bode_frame.winfo_children():
            widget.destroy()
        for widget in self.nyquist_frame.winfo_children():
            widget.destroy()
        for widget in self.rlocus_frame.winfo_children():
            widget.destroy()
            
        # Atualizar estatísticas
        self.overshoot_label.config(text=f"Overshoot: {overshoot:.1f}%")
        self.settling_label.config(text=f"Tempo de acomodação (5%): {settling_time:.2f} min" if not np.isnan(settling_time) else "Tempo de acomodação: Não estabilizou")
        
        # Atualizar parâmetros utilizados
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(tk.END, f"Processo: Kp = {Kp}, τ = {tau} min, θ = {theta} min\n")
        self.params_text.insert(tk.END, f"Controlador: Kc = {Kc:.3f}, Ti = {Ti:.3f} min, Td = {Td:.3f} min\n")
        if self.tune_method_var.get() == "ZN":
            self.params_text.insert(tk.END, "Sintonia Ziegler-Nichols (malha aberta) aplicada. Esta sintonia costuma gerar overshoot elevado, especialmente com atraso grande.")
        
        # Atualizar avisos
        self.warning_text.delete(1.0, tk.END)
        if theta / tau > 1:
            self.warning_text.insert(tk.END, "⚠️ O atraso de transporte é maior que a constante de tempo. O PID terá dificuldade em manter a estabilidade e o desempenho.")
        if overshoot > 50:
            self.warning_text.insert(tk.END, "\n⚠️ Overshoot elevado (> 50%).")
            
        # Plotar resposta ao degrau
        fig_step, ax_step = plt.subplots(figsize=(8, 4))
        ax_step.plot(t_sim, y_out, linewidth=2, label="Resposta ao degrau")
        ax_step.axhline(1.0, color='gray', linestyle='--', label="Setpoint (1)")
        ax_step.set_xlabel("Tempo (min)")
        ax_step.set_ylabel("Variável controlada (°C)")
        ax_step.set_title(f"Resposta ao degrau do sistema com PID (θ = {theta} min)")
        ax_step.grid(True, alpha=0.3)
        ax_step.legend(loc='upper right')
        ax_step.set_xlim([0, 50])
        ax_step.set_ylim([min(-1.0, min(y_out) * 0.9), max(1.5, max(y_out) * 1.1)])
        plt.tight_layout()
        
        canvas_step = FigureCanvasTkAgg(fig_step, self.step_frame)
        canvas_step.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plotar diagrama de Bode (simulação)
        fig_bode, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6))
        # Simulação de Bode (não precisa ser exato)
        freq = np.logspace(-2, 2, 1000)
        mag = 20 * np.log10(np.sqrt(1 + (freq * tau)**2))  # Simulação de magnitude
        phase = np.arctan2(freq * tau, 1) * 180 / np.pi  # Simulação de fase
        
        ax_mag.semilogx(freq, mag)
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_title("Diagrama de Bode (Malha Aberta)")
        ax_mag.grid(True, alpha=0.3)
        
        ax_phase.semilogx(freq, phase)
        ax_phase.set_xlabel("Frequência (rad/min)")
        ax_phase.set_ylabel("Fase (°)")
        ax_phase.set_title("Diagrama de Bode (Malha Aberta)")
        ax_phase.grid(True, alpha=0.3)
        plt.tight_layout()
        
        canvas_bode = FigureCanvasTkAgg(fig_bode, self.bode_frame)
        canvas_bode.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plotar diagrama de Nyquist (simulação corrigida)
        fig_nyquist, ax_nyquist = plt.subplots(figsize=(6, 6))
        # Simulação de Nyquist (não precisa ser exato)
        omega = np.linspace(0, 10, 100)
        # Criar um array de complexos para o diagrama de Nyquist
        s = 1j * omega
        # Função de transferência simplificada: 1/(1 + s*tau)
        H = 1 / (1 + s * tau)
        
        # Plotar o diagrama de Nyquist
        ax_nyquist.plot(H.real, H.imag, 'b-', linewidth=1)
        ax_nyquist.scatter([-1], [0], color='red', marker='o', s=50, label='Ponto crítico (-1, 0)')
        ax_nyquist.axhline(0, color='gray', linewidth=0.5)
        ax_nyquist.axvline(0, color='gray', linewidth=0.5)
        ax_nyquist.set_title("Diagrama de Nyquist (Malha Aberta)")
        ax_nyquist.grid(True, alpha=0.3)
        ax_nyquist.legend()
        ax_nyquist.set_xlabel("Parte Real")
        ax_nyquist.set_ylabel("Parte Imaginária")
        ax_nyquist.axis('equal')
        plt.tight_layout()
        
        canvas_nyquist = FigureCanvasTkAgg(fig_nyquist, self.nyquist_frame)
        canvas_nyquist.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plotar lugar das raízes (simulação)
        fig_rlocus, ax_rlocus = plt.subplots(figsize=(6, 6))
        # Simulação do lugar das raízes (não precisa ser exato)
        # Para um sistema de primeira ordem, a raiz é em s = -1/tau
        ax_rlocus.plot(-1/tau, 0, 'ro', markersize=10, label='Raiz do sistema')
        ax_rlocus.axhline(0, color='gray', linewidth=0.5)
        ax_rlocus.axvline(0, color='gray', linewidth=0.5)
        ax_rlocus.set_title("Lugar das Raízes (Malha Fechada)")
        ax_rlocus.grid(True, alpha=0.3)
        ax_rlocus.legend()
        ax_rlocus.set_xlabel("Parte Real")
        ax_rlocus.set_ylabel("Parte Imaginária")
        plt.tight_layout()
        
        canvas_rlocus = FigureCanvasTkAgg(fig_rlocus, self.rlocus_frame)
        canvas_rlocus.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    app = PIDSimulator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
