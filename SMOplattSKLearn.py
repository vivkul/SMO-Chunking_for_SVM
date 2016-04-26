# With Leukemia, took 23.169921 seconds (C=3, tol=10^-3)
# Here secondChoiceHeuristic can take a lot of time. It can be optimized
import math
import numpy as np
import random
from sklearn.datasets import load_svmlight_file
from datetime import datetime

startTime = datetime.now()

tenMinus8 = math.pow(10, -8)
eps = math.pow(10, -3)
tol = eps
diffTol = math.pow(10, -5)
noBound = 0

point,target = load_svmlight_file('Leukemia/leu')
pointT,targetT = load_svmlight_file('Leukemia/leu.t')
sizeW = 7129

# point,target = load_svmlight_file('covtype.binary/covtype.libsvm.binary_100')
# target = (target-1.5)*2
# sizeW = 54

# target,point,ind = svm_read_problem('covtype.binary/covtype.libsvm.binary.scale')
# target = (target-1.5)*2
# sizeW = 54
# length = 70*len(target)/100
# total = len(target)
# targetT = target[length:total]
# pointT = point[length:total]
# indT = ind[length:total]

# targetT,pointT,indT = svm_read_problem('rcv1.binary/rcv1_test.binary')
# target,point,ind = svm_read_problem('rcv1.binary/rcv1_train.binary')
# sizeW = 47236

length = len(target)
W = np.zeros(sizeW)
alpha = np.zeros(length)
b = 0
C = 3 #TODO

def accuracyTraining():
	correct = 0.0
	for x in range(0, length):
		gX = point[x].dot(W.T)[0] - b
		if(gX > 0 and target[x] == 1):
			correct += 1.0
		if(gX < 0 and target[x] == -1):
			correct += 1.0
	print correct/length

def accuracyTest():
	correct = 0.0
	lengthT = len(targetT)
	for x in range(0, lengthT):
		gX = pointT[x].dot(W.T)[0] - b
		if(gX > 0 and targetT[x] == 1):
			correct += 1.0
		if(gX < 0 and targetT[x] == -1):
			correct += 1.0
	print correct/lengthT

def stop_Criteria3(): #TODO Update W and b before this
	i = 0
	M = 99999
	m = -99999
	while (i < length):
		alphai = alpha[i]
		if((alphai < C and target[i] == 1) or (alphai == C and target[i] == -1)):
			E = target[i] - point[i].dot(W.T)[0]
			if(m < E):
				m = E
		elif((alphai < C and target[i] == -1) or (alphai == C and target[i] == 1)):
			E = target[i] - point[i].dot(W.T)[0]
			if(M > E):
				M = E
		i += 1
	print m - M
	if (m - M <= tol):
		return 1
	else:
		return 0

def stop_Criteria2(): #TODO Update W and b before this
	sumAlpha = 0
	sumSlack = 0
	i = 0
	while (i < length):
		sumAlpha += alpha[i]
		i += 1
	i = 0
	while (i < length):
		E = 1 - target[i]*(point[i].dot(W.T)[0] - b)
		if(E > 0):
			sumSlack = sumSlack + E
		i += 1
	L = np.dot(W,W)/2 - sumAlpha
	criteria2 = 2*L + sumAlpha + C*sumSlack  
	if ( criteria2 < tol and criteria2 > -tol):
		return 1
	else: 
		return 0


def check_KKT(I):
	gX = target[I]*(point[I].dot(W.T)[0] - b)-1
	if alpha[I] == 0 and gX >= -tol:
		return 0
	if alpha[I] == C and gX <= tol:
		return 0
	if alpha[I] > 0 and alpha[I] < C and gX >= -tol and gX <= tol:
		return 0
	return 1

def secondChoiceHeuristic(i2):
	E2 = point[i2].dot(W.T)[0] - b - target[i2]
	if E2 > 0:
		minE1 = E2
		minI = i2
		for x in range(0, length):
			E_e = point[x].dot(W.T)[0] - b - target[x]
			if E_e < minE1:
				minE1 = E_e
				minI = x
		return minI
	else:
		maxE1 = E2
		maxI = i2
		for x in range(0, length):
			E_e = point[x].dot(W.T)[0] - b - target[x]
			if E_e > maxE1:
				maxE1 = E_e
				maxI = x
		return maxI

def takeStep(i1,i2):
	global W
	global alpha
	global b
	global noBound
	if i1 == i2:
		return 0
	alph1 = alpha[i1]
	y1 = target[i1]
	E1 = point[i1].dot(W.T)[0] - b - y1 
	alph2 = alpha[i2]
	y2 = target[i2]
	E2 = point[i2].dot(W.T)[0] - b - y2 
	s = y1*y2
	if s > 0:
		L = max(0, alph1+alph2-C)
		H = min(C, alph1+alph2)
	else:
		L = max(0, -alph1+alph2)
		H = min(C, -alph1+alph2+C)
	if L == H:
		return 0
	k11 = point[i1].dot(point[i1].T)[0,0]
	k12 = point[i1].dot(point[i2].T)[0,0]
	k22 = point[i2].dot(point[i2].T)[0,0]
	eta = 2*k12-k11-k22
	if eta < 0:
		a2 = alph2 - y2*(E1-E2)/eta
		if a2 < L:
			a2 = L
		elif a2 > H:
			a2 = H
	else:
		sum_alpha = 0
		for x in range(0, length):
			if (x != i1 and x != i2):
				sum_alpha = sum_alpha + alpha[x]
		a2 = L
		a1 = alph1 + s*(alph2 - a2)
		w1 = np.array(W + (y1*(a1 - alph1)*point[i1] + y2*(a2 - alph2)*point[i2]))[0]
		Lobj = sum_alpha+a1+a2-np.dot(w1,w1)/2
		a2 = H
		a1 = alph1 + s*(alph2 - a2)
		w2 = np.array(W + (y1*(a1 - alph1)*point[i1] + y2*(a2 - alph2)*point[i2]))[0]
		Hobj = sum_alpha+a1+a2-np.dot(w2,w2)/2
		if Lobj > Hobj + eps:
			a2 = L
		elif Lobj < Hobj - eps:
			a2 = H
		else:
			a2 = alph2
	if a2 < tenMinus8:
		a2 = 0
	elif a2 > C - tenMinus8:
		a2 = C
	if(abs(a2-alph2) < diffTol):
		return 0
	a1 = alph1+s*(alph2-a2)
	b1 = E1 + y1*(a1 - alph1)*k11 + y2*(a2 - alph2)*k12 + b
	b2 = E2 + y1*(a1 - alph1)*k12 + y2*(a2 - alph2)*k22 + b
	b = (b1 + b2) / 2 
	W = np.array(W + (y1*(a1 - alph1)*point[i1] + y2*(a2 - alph2)*point[i2]))[0]
	if((alpha[i1] == C or alpha[i1] == 0) and (a1 != C and a1 != 0)):
		noBound = noBound + 1
	if((alpha[i1] != C and alpha[i1] != 0) and (a1 == C or a1 == 0)):
		noBound = noBound - 1
	if((alpha[i2] == C or alpha[i2] == 0) and (a2 != C and a2 != 0)):
		noBound = noBound + 1
	if((alpha[i2] != C and alpha[i2] != 0) and (a2 == C or a2 == 0)):
		noBound = noBound - 1
	alpha[i1] = a1
	alpha[i2] = a2
	return 1

def examineExample(i2):
	y2 = target[i2]
	alph2 = alpha[i2]
	E2 = point[i2].dot(W.T)[0] - b - y2
	r2 = E2*y2
	if ((r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0)): # tol is tolerance, it means that i2 doesn't satisfy KKT
		if noBound > 1:
			i1 = secondChoiceHeuristic(i2)
			if takeStep(i1, i2):
				return 1
		i = random.randrange(0, length-1, 1)
		for x in range(0, length):
			if(alpha[i] != 0 and alpha[i] != C):
				i1 = i
				if takeStep(i1, i2):
					return 1
			i = i+1
			if i > length-1:
				i = 1
		i = random.randrange(0, length-1, 1)
		for x in range(0, length):
			i1 = i
			if takeStep(i1, i2):
				return 1
			i = i+1
			if i > length-1:
				i = 1
	return 0

def main():
	numChanged = 0
	examineAll = 1
	while (numChanged > 0 or examineAll):
		numChanged = 0
		if examineAll:
			for I in range(0, length):
				if check_KKT(I):	# If I doesn't satisfy KKT condition within eps
					numChanged += examineExample(I) 
		else:
			for I in range(0, length):
				if(alpha[I] != 0 and alpha[I] != C):
					if check_KKT(I):	# If I doesn't satisfy KKT condition within eps
						numChanged += examineExample(I) 
		if examineAll == 1:
			examineAll = 0
		elif (numChanged == 0):
			examineAll = 1
		newCriteria = stop_Criteria3() # TODO
		if (((numChanged == 0 and examineAll == 1) or (numChanged == 0 and examineAll == 0)) and newCriteria == 0): # Here condn before "or" means that no further updation is requiered but will still go to next while loop and the condn after "or" means that the final while loop is also done
			print 'First Criteria stopped but new didnt'
		elif (((numChanged == 0 and examineAll == 1) or (numChanged == 0 and examineAll == 0)) == False and newCriteria == 1):
			print 'New Criteria stopped but First didnt'
		elif (((numChanged == 0 and examineAll == 1) or (numChanged == 0 and examineAll == 0)) and newCriteria == 1):
			print 'Both criteria stopped'
		if (newCriteria == 1):
			stoppingCriteria = 1
	accuracyTraining()
	accuracyTest()

main()
# np.save('alpha', alpha)
# np.save('omega',W)
print datetime.now() - startTime