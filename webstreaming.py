from motion_detection.singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# Iniciliza o frame de exibição
saidaFrame = None

# Iniciliza uma trava de segurança do processamento no servidor
lock = threading.Lock()

app = Flask(__name__)


camera = False;

# Inicializa a transmissão do video/livestreaming a ser trabalhado
if camera:
	vs = VideoStream(src=0).start()
else:
	vs = VideoStream(src="http://200.122.223.34:8001/mjpg/video.mjpg").start()

time.sleep(2.0)

@app.route("/")
def index():
	return render_template("opencv.html")

def detect_motion(frameCount):
	
	global vs, saidaFrame, lock
	
	# Inicilizar o motor de Detecção e o total de frames lidos
	md = SingleMotionDetector(acumPeso=0.1)
	total = 0

	# Executar continuamente ou até ser morto por um break
	while True:
		
		# Leitura do frame do vídeo
		frame = vs.read()
		
		
		if frame is None:
			break
		print("Lido")
   		# Redimensiona o tamanho do frame
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converte a coloração do frame para escala cinzenta
		gray = cv2.GaussianBlur(gray, (7, 7), 0) # Aplica um efeito de Gaussian Blur
		
		# Pegar e aplicar o timestamp no canto do frame
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		
		# Quando houver frames suficientes para começar a detecção (definido a partir do frameCount)
		if total > frameCount:
			
			# Executa metodo para detecção de movimento no frame trabalhado
			motion = md.detect(gray)
			
			# Se houver movimento detectado
			if motion is not None:
				
				# Desenha a caixa de movimento sobre o frame a partir das coordenadas encontradas na detecção
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(frame, (minX, minY), (maxX, maxY),(0, 0, 255), 2)
		
		
		# Atualiza o background e incrementa o número de frames lidos
		md.update(gray)
		total += 1
		
		# Determina o frame de exibição de forma que todos os clientes estejam sincronizados com a saida obtida
		with lock:
			saidaFrame = frame.copy()

def generate():
	
	global saidaFrame, lock
	
	# Exibir continuamente os frames de exibição que são obtidos
	while True:
		
		# Garantir a sincronia de exibição dos frames por todos os clientes
		with lock:
			
			# Se não foi recebido um frame de exibição
			if saidaFrame is None:
				continue
			
			# Codificar o frame de exibição em JPG
			(flag, encodedImage) = cv2.imencode(".jpg", saidaFrame)
			
			# Caso o frame não tenha sido codificado
			if not flag:
				continue
		
		# Gerar em formato binário a imagem a ser reproduzida na página
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
	# Todos os frames serão servidos a partir do endereço /video_feed a partir do retorno da imagem gerada juntamente ao seu tipo
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
	
	# Definir e pegar os argumentos passados na execução do programa, incluindo o ip, a porta e o limitador de frames
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,help="Endereço de Ip")
	ap.add_argument("-o", "--port", type=int, required=True,help="Número da porta")
	ap.add_argument("-f", "--frame-count", type=int, default=32,help="Número de frames para gerar o background")
	args = vars(ap.parse_args())
	
	# Iniciar o processamento em segundo plano da detecção de movimento
	t = threading.Thread(target=detect_motion, args=(args["frame_count"],))
	t.daemon = True
	t.start()
	
	# Iniciar servidor web 
	app.run(host=args["ip"], port=args["port"], debug=True,threaded=True, use_reloader=False)


# Encerrar a captura do vídeo
vs.stop()