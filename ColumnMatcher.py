import os
import pandas as pd
import json
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import deque
from dotenv import load_dotenv

# ====================
# CONFIGURACIÓN GEMINI
# ====================
load_dotenv()  # Carga las variables de entorno desde el archivo .env
api_key = os.getenv("GEMINI_API_KEY")  # Lee la clave de la variable de entorno
if not api_key:
    raise ValueError("No se encontró la clave API de GEMINI en el archivo .env. Por favor, asegúrate de agregarla.")

genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config
)

# ============================================
# RATE-LIMITER: para no exceder 10 llamadas/min
# ============================================
MAX_CALLS_PER_MINUTE = 10
call_times = deque()  # Aquí guardaremos los timestamps de cada llamada exitosa.

def gemini_rate_limited_call(user_message):
    """
    Envía un mensaje a Gemini, asegurándose de no exceder 10 RPM.
    Bloquea la ejecución si se llega al límite.
    """
    while len(call_times) >= MAX_CALLS_PER_MINUTE and (time.time() - call_times[0] < 60):
        sleep_time = 60 - (time.time() - call_times[0])
        print(f"[RateLimiter] Se alcanzó el límite de llamadas por minuto. Durmiendo {sleep_time:.1f} seg...")
        time.sleep(sleep_time)

    # Limpia timestamps antiguos
    while call_times and (time.time() - call_times[0] >= 60):
        call_times.popleft()

    # Realiza la llamada a Gemini
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(user_message)

    # Registra el timestamp de esta nueva llamada
    call_times.append(time.time())

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        print(f"[ERROR] No se pudo decodificar el JSON: {e}")
        return []

# ==========================================
# FUNCIÓN PARA DIVIDIR EN BLOQUES POR TOKENS
# ==========================================
def divide_into_token_safe_batches(target_column, model, max_tokens, col_name):
    """
    Divide la lista 'target_column' en bloques de máximo 'max_tokens' tokens,
    procesando de 20 filas en 20 filas. 
    Convierte cada fila en un dict con la clave definida por 'col_name':
    
    [
      {
        "<col_name>": "1501"
      },
      {
        "<col_name>": "1502"
      }
    ]
    """
    batches = []
    current_batch = []
    chunk_index = 1
    current_token_count = 0

    for i in range(0, len(target_column), 20):
        # Convierto este grupo de 20 elementos en un JSON con clave dinámica 'col_name'
        group_dicts = [{col_name: str(x)} for x in target_column[i : i + 20]]

        # Contamos los tokens de este grupo ya formateado como JSON
        group_token_count = model.count_tokens(json.dumps(group_dicts, ensure_ascii=False, indent=4)).total_tokens

        # Si excedemos el máximo de tokens, "cerramos" el batch anterior y empezamos uno nuevo
        if current_token_count + group_token_count > max_tokens:
            if current_batch:  # Agregar lote anterior si no está vacío
                batches.append(current_batch)
                print(f"[INFO] Agrupado bloque {chunk_index}...")
                chunk_index += 1
            current_batch = group_dicts
            current_token_count = group_token_count
        else:
            current_batch.extend(group_dicts)
            current_token_count += group_token_count

    # Agregar el último batch pendiente
    if current_batch:
        batches.append(current_batch)
        print(f"[INFO] Agrupado bloque {chunk_index}...")

    return batches

def process_block(source_json, batch, source_col_name, target_col_name, batch_idx, total_batches):
    """
    Prepara el prompt y realiza la llamada a Gemini (respetando el límite de llamadas por minuto).
    """
    # 'batch' ya viene en formato [{ "<target_col_name>": "..." }, ...]
    batch_json = json.dumps(batch, ensure_ascii=False)

    user_message = (
        f"Here is the source list ({source_col_name}): {source_json}\n\n"
        f"Here are the entries to match ({target_col_name}): {batch_json}.\n\n"
        f"Please match each entry exactly as it appears in both lists, preserving original spelling and formatting. "
        f"Return only a valid JSON array of objects, where each object has the structure:\n\n"
        f"{{\n"
        f"  \"{target_col_name}\": \"original_value_from_target\",\n"
        f"  \"{source_col_name}\": \"matched_value_from_source_or_blank\"\n"
        f"}}\n\n"
        f"If there is no suitable match for an entry, return an empty string (\"\") for the matched value. "
        f"Do not include any additional text or explanations. Make sure the JSON is valid and does not contain trailing commas, extraneous keys, or any other text."
    )

    print(f"[INFO] Ejecutando bloque {batch_idx}/{total_batches}...")
    result = gemini_rate_limited_call(user_message)
    return result if isinstance(result, list) else []

def match_using_gemini(source_column, target_column, source_col_name, target_col_name):
    """
    Procesa en lotes el 'target_column', llama a Gemini (respetando el límite de llamadas por minuto)
    y concatena los resultados.
    """
    source_json = json.dumps(source_column, ensure_ascii=False)
    matches = []

    # Ajusta aquí el max_tokens según tu límite
    batches = divide_into_token_safe_batches(target_column, model, max_tokens=4000, col_name=target_col_name)
    total_batches = len(batches)

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {}
        for idx, batch in enumerate(batches, start=1):
            future = executor.submit(
                process_block,
                source_json,
                batch,
                source_col_name,
                target_col_name,
                idx,
                total_batches
            )
            future_to_index[future] = idx

        for future in as_completed(future_to_index):
            batch_idx = future_to_index[future]
            try:
                result = future.result()
                matches.extend(result)
            except Exception as e:
                print(f"[ERROR] Bloque {batch_idx} falló: {e}")

    return matches

# ====================
# FUNCIÓN PRINCIPAL
# ====================
if __name__ == "__main__":
    # Solicitar inputs al usuario
    source_file = input("Introduce la ruta del archivo origen: ").strip('"').strip()
    source_sheet = input("Introduce el nombre de la hoja del archivo origen: ").strip()
    source_column_name = input("Introduce el nombre de la columna del archivo origen: ").strip()

    print("\n")

    target_file = input("Introduce la ruta del archivo destino: ").strip('"').strip()
    target_sheet = input("Introduce el nombre de la hoja del archivo destino: ").strip()
    target_column_name = input("Introduce el nombre de la columna del archivo destino: ").strip()

    # Cargar datos desde Excel y eliminar duplicados
    source_data = pd.read_excel(source_file, sheet_name=source_sheet)[source_column_name] \
                   .dropna().drop_duplicates().tolist()
    target_data = pd.read_excel(target_file, sheet_name=target_sheet)[target_column_name] \
                   .dropna().drop_duplicates().tolist()

    # Realizar el match
    matches = match_using_gemini(source_data, target_data, source_column_name, target_column_name)

    # Obtén el directorio del script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Guardar resultados en un archivo JSON
    json_output_file = os.path.join(script_directory, "ColumnMatcher.json")
    with open(json_output_file, "w", encoding="utf-8") as json_file:
        json.dump(matches, json_file, ensure_ascii=False, indent=4)

    print(f"[OK] Se generó el archivo JSON de resultados: {json_output_file}")

    # Guardar resultados en un archivo Excel
    output_file = os.path.join(script_directory, "ColumnMatcher.xlsx")
    pd.DataFrame(matches).to_excel(output_file, index=False)

    print(f"[OK] Se generó el archivo de resultados: {output_file}")

    input("Presiona Enter para salir...")
