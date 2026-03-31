import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import StateSpace, lsim, step

class StateSpaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modelagem em Espaço de Estados")
        self.root.geometry("1200x800")

        # Variáveis para dimensões
        self.n = tk.IntVar(value=2)          # número de estados
        self.m = tk.IntVar(value=1)          # número de entradas
        self.p = tk.IntVar(value=1)          # número de saídas

        # Widgets de controle
        self.create_control_frame()

        # Frame para as matrizes (será preenchido dinamicamente)
        self.matrix_frame = ttk.LabelFrame(self.root, text="Matrizes do Sistema")
        self.matrix_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Frame para botões e resultados
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=5)

        ttk.Button(self.button_frame, text="Atualizar Matrizes", command=self.update_matrix_grid).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Simular (Degrau)", command=self.simulate_step).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Carregar Exemplo", command=self.load_example).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Limpar Gráficos", command=self.clear_plots).pack(side=tk.LEFT, padx=5)

        # Frame para gráfico e texto de saída
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Figura matplotlib
        self.fig, self.axs = plt.subplots(1, 1, figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Área de texto para informações (autovalores, etc.)
        self.info_text = tk.Text(self.plot_frame, height=8, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=(5,0))

        # Inicializa a grade de matrizes
        self.update_matrix_grid()

    def create_control_frame(self):
        """Cria a parte superior com seletores de dimensões"""
        dim_frame = ttk.LabelFrame(self.root, text="Dimensões do Sistema")
        dim_frame.pack(pady=5, padx=10, fill=tk.X)

        ttk.Label(dim_frame, text="Número de estados (n):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(dim_frame, from_=1, to=5, textvariable=self.n, width=5, command=self.update_matrix_grid).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(dim_frame, text="Número de entradas (m):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(dim_frame, from_=1, to=3, textvariable=self.m, width=5, command=self.update_matrix_grid).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(dim_frame, text="Número de saídas (p):").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(dim_frame, from_=1, to=3, textvariable=self.p, width=5, command=self.update_matrix_grid).grid(row=0, column=5, padx=5, pady=2)

    def update_matrix_grid(self):
        """Cria ou atualiza a grade de entradas para as matrizes A, B, C, D"""
        # Limpa o frame atual
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        n = self.n.get()
        m = self.m.get()
        p = self.p.get()

        # Cria um canvas com scroll para matrizes grandes (caso n > 4)
        canvas = tk.Canvas(self.matrix_frame)
        scrollbar = ttk.Scrollbar(self.matrix_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # A matriz
        ttk.Label(scrollable_frame, text="Matriz A (n x n)", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=n, pady=5)
        self.A_entries = []
        for i in range(n):
            row_entries = []
            for j in range(n):
                entry = ttk.Entry(scrollable_frame, width=8)
                entry.grid(row=i+1, column=j, padx=2, pady=2)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.A_entries.append(row_entries)

        # Matriz B
        ttk.Label(scrollable_frame, text="Matriz B (n x m)", font=("Arial", 10, "bold")).grid(row=0, column=n+2, columnspan=m, pady=5)
        self.B_entries = []
        for i in range(n):
            row_entries = []
            for j in range(m):
                entry = ttk.Entry(scrollable_frame, width=8)
                entry.grid(row=i+1, column=n+2+j, padx=2, pady=2)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.B_entries.append(row_entries)

        # Matriz C
        ttk.Label(scrollable_frame, text="Matriz C (p x n)", font=("Arial", 10, "bold")).grid(row=n+2, column=0, columnspan=n, pady=5)
        self.C_entries = []
        for i in range(p):
            row_entries = []
            for j in range(n):
                entry = ttk.Entry(scrollable_frame, width=8)
                entry.grid(row=n+3+i, column=j, padx=2, pady=2)
                entry.insert(0, "1" if i==j else "0")
                row_entries.append(entry)
            self.C_entries.append(row_entries)

        # Matriz D
        ttk.Label(scrollable_frame, text="Matriz D (p x m)", font=("Arial", 10, "bold")).grid(row=n+2, column=n+2, columnspan=m, pady=5)
        self.D_entries = []
        for i in range(p):
            row_entries = []
            for j in range(m):
                entry = ttk.Entry(scrollable_frame, width=8)
                entry.grid(row=n+3+i, column=n+2+j, padx=2, pady=2)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.D_entries.append(row_entries)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def get_matrices(self):
        """Lê as matrizes dos campos de entrada e retorna A, B, C, D como arrays numpy"""
        n = self.n.get()
        m = self.m.get()
        p = self.p.get()
        try:
            A = np.array([[float(self.A_entries[i][j].get()) for j in range(n)] for i in range(n)])
            B = np.array([[float(self.B_entries[i][j].get()) for j in range(m)] for i in range(n)])
            C = np.array([[float(self.C_entries[i][j].get()) for j in range(n)] for i in range(p)])
            D = np.array([[float(self.D_entries[i][j].get()) for j in range(m)] for i in range(p)])
            return A, B, C, D
        except ValueError:
            messagebox.showerror("Erro", "Entradas inválidas. Verifique se todas as células contêm números.")
            return None, None, None, None

    def simulate_step(self):
        """Simula a resposta ao degrau do sistema para cada entrada (um degrau unitário)"""
        A, B, C, D = self.get_matrices()
        if A is None:
            return

        n = self.n.get()
        m = self.m.get()
        p = self.p.get()

        # Criar sistema em espaço de estados
        sys = StateSpace(A, B, C, D)

        # Tempo de simulação
        t = np.linspace(0, 10, 1000)

        # Para cada entrada, aplicamos um degrau unitário nela e zeros nas demais
        responses = []  # cada elemento é uma matriz (len(t), p)
        input_labels = []

        for input_idx in range(m):
            # Construir a entrada: degrau unitário no canal input_idx
            u = np.zeros((len(t), m))
            u[:, input_idx] = 1.0
            # Simular
            _, y, _ = lsim(sys, U=u, T=t)
            # Garantir que y seja 2D: (len(t), p)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            responses.append(y)
            input_labels.append(f"Entrada {input_idx+1}")

        # Plotar os resultados
        self.clear_plots()

        # Organizar subplots: se p (número de saídas) > 1, colocar cada saída em um subplot
        n_plots = p
        self.fig.clear()
        if n_plots == 1:
            ax = self.fig.add_subplot(111)
            for idx, resp in enumerate(responses):
                ax.plot(t, resp[:, 0], label=f"Saída (entrada {idx+1})")
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Saída")
            ax.grid(True)
            ax.legend()
        else:
            for i in range(p):
                ax = self.fig.add_subplot(p, 1, i+1)
                for idx, resp in enumerate(responses):
                    ax.plot(t, resp[:, i], label=f"Entrada {idx+1}")
                ax.set_ylabel(f"Saída {i+1}")
                ax.grid(True)
                ax.legend()
                if i == p-1:
                    ax.set_xlabel("Tempo (s)")
        self.fig.tight_layout()
        self.canvas.draw()

        # Calcular e exibir propriedades do sistema
        self.show_system_properties(A, B, C, D)

    def show_system_properties(self, A, B, C, D):
        """Calcula autovalores, controlabilidade e observabilidade"""
        n = A.shape[0]
        eigvals = np.linalg.eigvals(A)

        # Controlabilidade (rank da matriz de controlabilidade)
        controllability_matrix = B
        for i in range(1, n):
            controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))
        rank_cont = np.linalg.matrix_rank(controllability_matrix)
        controllable = (rank_cont == n)

        # Observabilidade (rank da matriz de observabilidade)
        observability_matrix = C
        for i in range(1, n):
            observability_matrix = np.vstack((observability_matrix, C @ np.linalg.matrix_power(A, i)))
        rank_obs = np.linalg.matrix_rank(observability_matrix)
        observable = (rank_obs == n)

        # Exibir no widget de texto
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "=== PROPRIEDADES DO SISTEMA ===\n")
        self.info_text.insert(tk.END, f"Autovalores: {eigvals}\n")
        self.info_text.insert(tk.END, f"Rank da controlabilidade: {rank_cont} (n={n}) → {'Controlável' if controllable else 'Não controlável'}\n")
        self.info_text.insert(tk.END, f"Rank da observabilidade: {rank_obs} (n={n}) → {'Observável' if observable else 'Não observável'}\n")
        self.info_text.config(state=tk.DISABLED)

    def clear_plots(self):
        """Limpa a figura e o texto de informações"""
        self.fig.clear()
        self.canvas.draw()
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.config(state=tk.DISABLED)

    def load_example(self):
        """Carrega um exemplo predefinido: um tanque com dois estados"""
        # Exemplo: dois tanques em série (modelo de nível)
        # Estados: níveis dos tanques; entrada: vazão de alimentação; saída: nível do segundo tanque
        n = 2
        m = 1
        p = 1
        self.n.set(n)
        self.m.set(m)
        self.p.set(p)
        self.update_matrix_grid()

        # Preencher A, B, C, D com valores do exemplo
        A_ex = [[-0.5, 0], [0.5, -0.3]]
        B_ex = [[0.5], [0]]
        C_ex = [[0, 1]]
        D_ex = [[0]]

        for i in range(n):
            for j in range(n):
                self.A_entries[i][j].delete(0, tk.END)
                self.A_entries[i][j].insert(0, str(A_ex[i][j]))
        for i in range(n):
            for j in range(m):
                self.B_entries[i][j].delete(0, tk.END)
                self.B_entries[i][j].insert(0, str(B_ex[i][j]))
        for i in range(p):
            for j in range(n):
                self.C_entries[i][j].delete(0, tk.END)
                self.C_entries[i][j].insert(0, str(C_ex[i][j]))
        for i in range(p):
            for j in range(m):
                self.D_entries[i][j].delete(0, tk.END)
                self.D_entries[i][j].insert(0, str(D_ex[i][j]))

        messagebox.showinfo("Exemplo", "Exemplo carregado: dois tanques em série.\nAgora clique em 'Simular (Degrau)'.")

if __name__ == "__main__":
    root = tk.Tk()
    app = StateSpaceApp(root)
    root.mainloop()