#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
	ISIR CNRS/UPMC
	12/06/2017
""" 

import numpy as np
import cv2
import classif_mehdi_fcn

import analytics
import gen_ydata

from threading import Thread, Lock
#from Queue import Queue

from skimage import io #
from socketIO_client_nexus import SocketIO, LoggingNamespace #

#socketIO = SocketIO('93.24.79.14', 3333) #
#socketIO = SocketIO('localhost', 3333, LoggingNamespace) #

tags_to_labels, labels_to_tags = gen_ydata.load_mappings()

def entropy_from_hmaps(hmaps):
	entropy = np.zeros(hmaps.shape[:2])
	for i in range(len(classif_mehdi_fcn.classes)):
		entropy += - hmaps[:,:,i]*np.log(hmaps[:,:,i])
	return entropy




def compute_histograms(hmaps,entropy_th=None):
	histos = list()
	if(entropy_th):
		entropy = entropy_from_hmaps(hmaps)
		good_pixels = np.array(entropy<entropy_th,dtype=np.float)
	else:
		good_pixels=None
	for i in range(len(classif_mehdi_fcn.classes)):
		h, _ = np.histogram(hmaps[:,:,i], range=(0.,1.), weights=good_pixels)
		histos.append(h)
	return histos
	

def crop_and_resize(img,target=(224,224)):
	h,w,c = img.shape
	smalldim = min(h,w)
	bigdim = max(h,w)
	small_first = h<w
	bigoffset = (bigdim-smalldim)/2
	squarecrop = img[:,bigoffset:(bigoffset+smalldim),:] if(small_first) else img[bigoffset:(bigoffset+smalldim),:,:]
	res = cv2.resize(np.array(squarecrop,dtype=float),target, interpolation = cv2.INTER_AREA)
	return res[:,:,::-1] #Color order is inverted between cv2 and PIL !


def process_histograms(histos):
	out = list()
	for h in histos:	
		out.append(np.sum(analytics.mask_yes*h)/np.sum(h))
	return out


def display_results(scores,th=0.001):
	order = np.argsort(scores)[::-1]
	print("Objects detected:")
	for i in order:
		if scores[i]<th:
			break
		c = classif_mehdi_fcn.classes[i]
		print("- %s (%s, score %.04f)" % (labels_to_tags[c], c, scores[i]))


class CameraInterface:
	def __init__(self,video_dev_id=0):
		self.cap = cv2.VideoCapture(video_dev_id)
		self.running = False
		self.frame_lock = Lock()
		self.new = False

	def grabber_thread(self):
		while self.running:
			#print "Thread running"
			(grabbed, frame) = self.cap.read()
			self.frame_lock.acquire()
			self.current_frame = frame.copy()
			self.new = True
			self.frame_lock.release()

	def start(self):
		self.running = True
		self.thread = Thread(target=self.grabber_thread, args=())
		self.thread.start()


	def stop(self):
		self.running = False
		self.thread.join()
		self.cap.release()
	
	def get_frame(self):
		self.frame_lock.acquire()
		if(self.new):
			frame = self.current_frame.copy()
			self.new = False
			self.frame_lock.release()
			return frame
		else:
			self.frame_lock.release()
			return None


class ImagesPlaybackSource:
	def __init__(self,directory="",pattern="image-%04d.jpg",imgUrl="",loop=True): #
		self.dir = directory
		self.pattern = pattern
		self.imgUrl = imgUrl #
		self.i = 1
		self.loop = loop


	def start(self):
		pass

	def stop(self):
		pass

	def get_frame(self):
		img = None
		if (self.dir): #
			path = self.dir + '/' + (self.pattern % self.i)
			print("reading image:", path)
			try:
				img = cv2.imread(path)
			except:
				print("Failed to retrieve image")
			if img is None and self.loop:
				self.i = 1
			else:
				self.i += 1
		elif (self.imgUrl):
			img = cv2.cvtColor(io.imread(self.imgUrl), cv2.COLOR_RGB2BGR)
		return img



class ProductIdentifier:
	def __init__(self,images_source, plotting=True):
		self.source = images_source
		self.network = classif_mehdi_fcn.load_network("lastweights-fullnet.h5",exclusive=False)
		self.ea = analytics.EAfilter(plot=plotting)
		self.plotting = plotting
		self.iframe = 0
		#socketIO.on('validate', self.valid_reco) #
		
		
	def process_frame(self,frame,visu=True):
		if(visu):
			cv2.imshow('Test out',frame)
		res = crop_and_resize(frame)
		for i in range(3):
			res[:,:,i] -= classif_mehdi_fcn.vgg_rgboffsets[i]
		resexpanded = np.expand_dims(res,0)
		out = self.network.predict(resexpanded)
		hmaps = out[0]
		#entropy = entropy_from_hmaps(hmaps)
		histos = compute_histograms(hmaps)
		scores = process_histograms(histos)
		self.ea.add_data(scores)
		print("=== Frame %d ===" % self.iframe)
		#self.ea.print_nbest(4,0.5)
		message = self.ea.print_nbest(4,0.5) #
		#if message: #
		#	socketIO.emit('product found', {'ids': filter(bool, message.split("|"))}) #
		#socketIO.wait(seconds=.2)
		self.iframe += 1
		#display_results(scores)


	def valid_reco(self):
		print("===== VALIDATED RESULT =====")
		self.ea.print_nbest(4,0.5)
		self.iframe = 0
		self.ea.reset()

		
	def start(self):
		self.source.start()
		while(True):
			#print("running")
			frame = self.source.get_frame()
			if frame is not None:
				self.process_frame(frame,visu=self.plotting)
			if(len(self.ea.get_nbest(0,1.0))>0):
				self.valid_reco()
			code = cv2.waitKey(1) & 0xFF
			if code == ord('q'):
				self.source.stop()
				break
			elif code == ord(' '):
				self.valid_reco()
		cv2.destroyAllWindows()
