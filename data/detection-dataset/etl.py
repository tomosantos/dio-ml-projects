import subprocess
import os

# Diretórios de origem e destino
input_dir = "raw"
output_dir = "labeled"

# Certificando que o diretório de destino existe
os.makedirs(output_dir, exist_ok=True)

# Lista de arquivos no diretório de origem
json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

# Iterando sobre os arquivos
for json_file in json_files:
    input_path = os.path.join(input_dir, json_file)
    output_path = os.path.join(output_dir, json_file.replace(".json", "_labeled"))
    
    # Comando para execução do script de rotulação
    command = ["labelme_export_json", input_path, "-o", output_path]
    
    try:
        # Executando o comando
        subprocess.run(command, check=True)
        print(f"Processado: {json_file}")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao processar {json_file}: {e}")