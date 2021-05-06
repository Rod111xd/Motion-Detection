import numpy as np
import imutils
import cv2

class SingleMotionDetector:
	def __init__(self, acumPeso=0.5):
		# Construtor para definir os atributos
		self.acumPeso = acumPeso   # Peso acumulativo
		self.bg = None # Background

	def update(self, imagem):
		# Caso seja a primeira vez executada o método update, define o background inicial
		if self.bg is None:
			self.bg = imagem.copy().astype("float")
			return
		
		# Acumula o peso atualizando o background
		cv2.accumulateWeighted(imagem, self.bg, self.acumPeso)

	def detect(self, imagem, tVal=25):
		
		# Calcular a diferença absoluta do background com o frame atual
		delta = cv2.absdiff(self.bg.astype("uint8"), imagem)
		# Definir a constante de limite para gerar um foreground
		thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]
		
		# Realizar efeitos sobre o threshold gerado para otimizar a deteção
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)

		
		# Procurar e pegar contornos envolvendo o movimento detectado
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		# Inicializar as coordenadas mínimas e máximas da caixa de movimento
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)

		# Caso não encontre contornos
		if len(cnts) == 0:
			return None
		
		for c in cnts:
			
			# Computar os vértices da caixa de movimento
			(x, y, w, h) = cv2.boundingRect(c)
			(minX, minY) = (min(minX, x), min(minY, y))
			(maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))
		
		
		return (thresh, (minX, minY, maxX, maxY))

