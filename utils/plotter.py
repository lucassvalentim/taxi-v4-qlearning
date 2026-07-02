import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_learning_curve(df_or_dict, window=100, output_path=None):
    """
    Plota a curva de aprendizado (recompensa total por episódio).
    Suporta um DataFrame único (com dados brutos) ou um dicionário de DataFrames 
    para comparação direta em artigos científicos.
    """
    plt.figure(figsize=(12, 6))
    
    # Caso 1: Foi passado um dicionário para COMPARAÇÃO (Q-Learning vs SARSA)
    if isinstance(df_or_dict, dict):
        # Diferenciação por estilo de linha para garantir leitura clara (mesmo em P&B)
        linestyles = {"Q-Learning": "-", "SARSA": "--"}
        
        for label, df in df_or_dict.items():
            df_copy = df.copy()
            df_copy['moving_avg_reward'] = df_copy['total_reward'].rolling(window=window).mean()
            
            current_style = linestyles.get(label, "-")
            
            # Plota apenas a média móvel para evitar poluição visual
            sns.lineplot(
                x='episode', y='moving_avg_reward', data=df_copy, 
                label=f'{label} (Média Móvel, w={window})', 
                linewidth=2.5, linestyle=current_style
            )
            
    # Caso 2: Foi passado apenas um DataFrame único (Comportamento Original)
    else:
        df_copy = df_or_dict.copy()
        # Plot dos dados brutos com opacidade reduzida
        sns.lineplot(x='episode', y='total_reward', data=df_copy, alpha=0.3, label='Recompensa Bruta')
        
        # Cálculo e plot da média móvel
        df_copy['moving_avg_reward'] = df_copy['total_reward'].rolling(window=window).mean()
        sns.lineplot(x='episode', y='moving_avg_reward', data=df_copy, color='red', linewidth=2, label=f'Média Móvel (w={window})')
    
    plt.title('Learning Curve: Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right') # Geralmente as recompensas sobem, então a legenda fica melhor embaixo
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Ajustes acadêmicos: bbox_inches impede cortes e dpi=300 garante qualidade de impressão
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Gráfico salvo com sucesso em: {output_path}")
    
    plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_generic(df_or_dict, x, y, title, xlabel, ylabel, output_path=None, moving_avg_window=None):
    """
    Função genérica de plotagem. Suporta um DataFrame único ou um dicionário 
    de DataFrames para plotagem comparativa.
    """
    plt.figure(figsize=(12, 6))
    
    # Caso 1: Foi passado um dicionário para COMPARAÇÃO (ex: {"Q-Learning": df1, "SARSA": df2})
    if isinstance(df_or_dict, dict):
        # Definimos estilos diferentes para que um não esconda o outro na sobreposição
        linestyles = {"Q-Learning": "-", "SARSA": "--"} # Sólido para Q-Learning, Tracejado para SARSA
        
        for label, df in df_or_dict.items():
            df_copy = df.copy()
            # Pega o estilo correspondente (usa linha sólida '-' como padrão se não achar)
            current_style = linestyles.get(label, "-") 
            
            if moving_avg_window:
                df_copy['moving_avg'] = df_copy[y].rolling(window=moving_avg_window).mean()
                sns.lineplot(
                    x=x, y='moving_avg', data=df_copy, 
                    label=label, linewidth=2.5, linestyle=current_style
                )
            else:
                sns.lineplot(x=x, y=y, data=df_copy, label=label, linestyle=current_style)
                
    # Caso 2: Foi passado apenas um DataFrame único (Comportamento Original)
    else:
        df_copy = df_or_dict.copy()
        if moving_avg_window:
            sns.lineplot(x=x, y=y, data=df_copy, alpha=0.3, label='Dados Brutos')
            df_copy['moving_avg'] = df_copy[y].rolling(window=moving_avg_window).mean()
            sns.lineplot(x=x, y='moving_avg', data=df_copy, color='red', linewidth=2, label=f'Média Móvel (w={moving_avg_window})')
        else:
            sns.lineplot(x=x, y=y, data=df_copy)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # --- CORREÇÃO: Linhas que salvam o arquivo no disco ---
    if output_path:
        # Cria os diretórios caso não existam
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Gráfico salvo com sucesso em: {output_path}")
        
    plt.close()
    
def plot_comparison(results_dict, x, y, title, xlabel, ylabel, output_path, window=1000):
    """
    Plots multiple curves for comparison (e.g., different alphas or gammas)
    and saves the result to a file.
    results_dict: { 'label': dataframe }
    """
    plt.figure(figsize=(12, 6))
    
    for label, df in results_dict.items():
        # Calculate moving average
        df_copy = df.copy()
        df_copy['moving_avg'] = df_copy[y].rolling(window=window).mean()
        sns.lineplot(x=x, y='moving_avg', data=df_copy, label=label, linewidth=2)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    plt.close()
    
    print(f"Gráfico salvo com sucesso em: {output_path}")

def plot_comparison_sensitivity(results_dict, x, y, title, xlabel, ylabel, output_path, window=1000):
    """
    Plota múltiplas curvas para comparação de sensibilidade de hiperparâmetros.
    Garante formatação acadêmica e salvamento correto em disco.
    
    results_dict: Dicionário no formato { 'Rótulo (ex: Alpha=0.1)': DataFrame }
    """
    # Configura um tema limpo e profissional do Seaborn
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 6))
    
    # Loop pelas combinações do dicionário para plotar cada curva
    for label, df in results_dict.items():
        df_copy = df.copy()
        
        # Calcula a média móvel para suavizar a curva de recompensas
        # min_periods=1 evita que o gráfico comece em branco no início do treino
        df_copy['moving_avg'] = df_copy[y].rolling(window=window, min_periods=1).mean()
        
        # Plota a linha suavizada com espessura ideal para leitura
        sns.lineplot(x=x, y='moving_avg', data=df_copy, label=label, linewidth=2.5)
        
    # Customização de títulos e eixos seguindo padrões acadêmicos
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Posiciona a legenda de forma inteligente onde menos atrapalhar os dados
    plt.legend(loc='best', fontsize=11, frameon=True, shadow=False)
    
    # --- SALVAMENTO EM DISCO CORRIGIDO ---
    if output_path:
        # Garante a criação automática da pasta de destino se ela não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # dpi=300 garante a nitidez exigida por periódicos e simpósios
        # bbox_inches='tight' impede que elementos da legenda ou eixos saiam cortados
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Gráfico salvo com sucesso em: {output_path}")
        
    # Fecha a figura atual para liberar a memória RAM do sistema
    plt.close()
    
def plot_dual_axis(df, x, y1, y2, title, xlabel, y1label, y2label, output_path, window=1000):
    """
    Plots two metrics with different Y-axes (e.g., Reward and Epsilon).
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # First axis (Reward)
    color1 = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color=color1)
    
    # Moving average for y1
    df_copy = df.copy()
    df_copy['y1_moving_avg'] = df_copy[y1].rolling(window=window).mean()
    sns.lineplot(x=x, y='y1_moving_avg', data=df_copy, ax=ax1, color=color1, label=y1label)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Second axis (Epsilon)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(y2label, color=color2)
    sns.lineplot(x=x, y=y2, data=df, ax=ax2, color=color2, label=y2label, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(title)
    fig.tight_layout()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
def plot_robustness(data_or_dict, x, y, title, xlabel, ylabel, output_path, window=1000):
    """
    Plota a média e o desvio padrão (sombreado) de múltiplas execuções.
    Suporta uma lista única de DataFrames ou um dicionário contendo listas por agente.
    """
    plt.figure(figsize=(12, 6))
    
    # Caso 1: Dicionário para COMPARAÇÃO (ex: {"Q-Learning": [df1, df2...], "SARSA": [df1, df2...]})
    if isinstance(data_or_dict, dict):
        linestyles = {"Q-Learning": "-", "SARSA": "--"}
        
        for label, data_list in data_or_dict.items():
            processed_data = []
            for i, df in enumerate(data_list):
                df_copy = df.copy()
                df_copy['moving_avg'] = df_copy[y].rolling(window=window).mean()
                df_copy['run_id'] = i
                processed_data.append(df_copy)
            
            combined_processed = pd.concat(processed_data).reset_index(drop=True)
            current_style = linestyles.get(label, "-")
            
            # O Seaborn calcula a média e o desvio padrão (sd) automaticamente a partir do 'run_id'
            sns.lineplot(
                x=x, y='moving_avg', data=combined_processed, 
                errorbar='sd', linewidth=2.5, linestyle=current_style,
                label=f'{label} (Média ± DP)'
            )
            
    # Caso 2: Lista única de DataFrames (Comportamento Original)
    else:
        processed_data = []
        for i, df in enumerate(data_or_dict):
            df_copy = df.copy()
            df_copy['moving_avg'] = df_copy[y].rolling(window=window).mean()
            df_copy['run_id'] = i
            processed_data.append(df_copy)
            
        combined_processed = pd.concat(processed_data).reset_index(drop=True)
        sns.lineplot(x=x, y='moving_avg', data=combined_processed, errorbar='sd', linewidth=2, label='Mean Reward (± Std)')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # Ajustes finos para submissão científica
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Gráfico de robustez salvo com sucesso em: {output_path}")
        
    plt.close()
