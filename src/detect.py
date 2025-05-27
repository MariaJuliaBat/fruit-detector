print(">>> detect.py ESTÁ RODANDO <<<")
# src/detect.py
import os
import multiprocessing
import numpy as np
import cv2
import tensorflow as tf
import pyttsx3

# --- CONFIGURAÇÃO DOS CAMINHOS ---
DIR_PATH    = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH  = os.path.join(DIR_PATH, '..', 'model', 'fruit_model.h5')
LABELS_PATH = os.path.join(DIR_PATH, '..', 'model', 'labels.txt')

# Debug: verificar paths
print(f"Usando modelo em: {MODEL_PATH}")
print(f"Usando labels em: {LABELS_PATH}")

# --- CARREGA AS CLASSES ---
try:
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"Classes carregadas: {classes}")
except Exception as e:
    print(f"Erro ao ler labels: {e}")
    raise

# --- CARREGA O MODELO ---
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    raise

# --- PROCESSO DE TEXTO-TO-SPEECH (TTS) ---
def tts_loop(queue):
    engine = pyttsx3.init()
    last_msg = ''
    while True:
        msg = queue.get()
        while not queue.empty():
            msg = queue.get()
        if msg != last_msg and msg != 'Background':
            last_msg = msg
            engine.say(msg)
            engine.runAndWait()


def main():
    print("Iniciando captura da webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        return
    CONF_THRESHOLD = 0.90  # 90%

    # inicia o processo de TTS
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=tts_loop, args=(q,), daemon=True)
    p.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha na captura do frame.")
            break

        # pré-processamento: redimensiona e normaliza para o TM
        try:
            img = cv2.resize(frame, (224, 224))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Erro no pré-processamento: {e}")
            continue
        arr = (np.asarray(img_rgb, dtype=np.float32) / 127.0) - 1.0
        data = np.expand_dims(arr, axis=0)

        # predição
        try:
            preds = model.predict(data)[0]
        except Exception as e:
            print(f"Erro na predição: {e}")
            continue
        idx   = int(np.argmax(preds))
        score = float(preds[idx])
        label = classes[idx]

        # exibe resultado na imagem
        text = f"{label}: {int(score * 100)}%"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # envia para TTS se acima do limiar
        if score >= CONF_THRESHOLD:
            q.put(label)

        # mostra na tela e sai com 'q'
        cv2.imshow("Fruit Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Saindo...")
            break

    cap.release()
    cv2.destroyAllWindows()
    p.terminate()

if __name__ == '__main__':
    main()
