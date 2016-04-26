# Time to run it was 01:47.561827 sec (C=3, tol=10^-3, chunkSize = length/4, M = 3*chunkSize/4)
# The accuracy is same as that of SMOplatt1.py. 
# W and b are almost same to that in SMOplatt.
# No. of support vectors in both is almost 31.
# For all TODO, check other possibilities
import math
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from dataParser import svm_read_problem
from datetime import datetime

startTime = datetime.now()

tenMinus8 = math.pow(10, -8)
tol = math.pow(10, -3)

target,point,ind = svm_read_problem('Leukemia/leu')
targetT,pointT,indT = svm_read_problem('Leukemia/leu.t')
sizeW = 7129


W = np.zeros(sizeW)
length = len(target)
chunkSize = length/4 #TODO
used = np.zeros(length)
used.fill(-1) # make all elements -1
present = np.zeros(length)
present.fill(-1)
solution = np.zeros(chunkSize)
M = 3*chunkSize/4 #TODO
adder = np.zeros(M)
e = np.ones(chunkSize)
J = np.concatenate((-np.identity(chunkSize),np.identity(chunkSize)),axis = 0)
C = 3 #TODO
n = np.concatenate((np.zeros(chunkSize),C*e),axis = 0)
d = matrix([0.0])
b = 0
L = 0 	#Dual optimal

def assignW(id1, val1):
	global W
	W = np.zeros(sizeW)
	p1 = 0
	num1 = len(id1)
	while p1 < num1:
		a1 = int(id1[p1])
		W[a1-1] = val1[p1]
		p1+=1


def multList(P,S):
	return map(lambda x: x * P, S)

def addWP1P2(val3, id1, val1, id2, val2): 
	p1 = 0
	p2 = 0
	num1 = len(id1)
	num2 = len(id2)
	while p1 < num1:
		a1 = int(id1[p1])
		val3[a1-1] = val3[a1-1] + val1[p1]
		p1+=1
	while p2 < num2:
		a2 = int(id2[p2])
		val3[a2-1] = val3[a2-1] + val2[p2]
		p2+=1
	return val3

def dotProductP1(val2,id1,val1):
	p1 = 0
	dot = 0
	num1 = len(id1)
	while p1 < num1:
		a1 = int(id1[p1])
		dot += val1[p1]*val2[a1-1]
		p1+=1
	return dot

def dotProductP1P2(id1,val1,id2,val2):
	p1 = 0
	p2 = 0
	dot = 0
	num1 = len(id1)
	num2 = len(id2)
	while p1 < num1 and p2 < num2:
		a1 = int(id1[p1])
		a2 = int(id2[p2])
		if a1 == a2:
			dot += val1[p1]*val2[p2]
			p1+=1
			p2+=1
		elif a1 > a2:
			p2+=1
		else:
			p1+=1
	return dot

def accuracyTraining():
	correct = 0.0
	for x in range(0, length):
		gX = dotProductP1(W,ind[x],point[x]) + b
		if(gX > 0 and target[x] == 1):
			correct += 1.0
		if(gX < 0 and target[x] == -1):
			correct += 1.0
	print correct/length

def accuracyTest():
	correct = 0.0
	lengthT = len(targetT)
	for x in range(0, lengthT):
		gX = dotProductP1(W,indT[x],pointT[x]) + b
		if(gX > 0 and targetT[x] == 1):
			correct += 1.0
		if(gX < 0 and targetT[x] == -1):
			correct += 1.0
	print correct/lengthT

def correct_solution():
	global solution
	i = 0
	while (i < length and used[i]!=-1):
		if (solution[i] < tenMinus8):
			solution[i] = 0
		elif solution[i] > C - tenMinus8:
			solution[i] = C
		i += 1

def update_W():
	global W
	assignW(ind[used[0]],multList(solution[0]*target[used[0]],point[used[0]]))
	i = 1
	while (i < length and used[i] != -1):
		W = addWP1P2(W, ind[used[i]], multList(solution[i]*target[used[i]],point[used[i]]), [], [])
		i += 1

def update_b():
	global b
	i = 0
	while (i < length and used[i] != -1):
		if(solution[i]>0 and solution[i]<C):
			b = target[used[i]] - dotProductP1(W,ind[used[i]],point[used[i]]) #TODO is b calculated rightly? or should I take average?
			break
		i += 1

def stop_Criteria3(): #TODO Update W and b before this
	i = 0
	M = 99999
	m = -99999
	while (i < length):
		if (present[i] != -1):
			alpha = solution[present[i]]
		else:
			alpha = 0
		if((alpha < C and target[i] == 1) or (alpha == C and target[i] == -1)):
			E = target[i] - dotProductP1(W,ind[i],point[i])
			if(m < E):
				m = E
		elif((alpha < C and target[i] == -1) or (alpha == C and target[i] == 1)):
			E = target[i] - dotProductP1(W,ind[i],point[i])
			if(M > E):
				M = E
		i += 1
	if (m - M <= tol):
		return 1
	else:
		return 0

def stop_Criteria2(): #TODO Update W and b before this
	sumAlpha = 0
	sumSlack = 0
	i = 0
	while (i < length and used[i] != -1):
		sumAlpha += solution[i]
		i += 1
	i = 0
	while (i < length):
		E = 1 - target[i]*(dotProductP1(W,ind[i],point[i]) + b)
		if(E > 0):
			sumSlack = sumSlack + E
		i += 1
	criteria2 = 2*L + sumAlpha + C*sumSlack 
	if ( criteria2 < tol and criteria2 > -tol):
		return 1
	else: 
		return 0


def check_Criteria(): #returns 1 if the stopping criteria is met
	global adder
	update_W()
	update_b()
	i = 0
	j = 0
	errorArray = np.zeros(shape=(length-chunkSize,2))
	while (i < length):
		if(present[i] == -1):
			errorArray[j,0] = target[i]*(dotProductP1(W,ind[i],point[i]) + b)-1
			errorArray[j,1] = i
			j += 1
		i += 1
	errorArray = errorArray[errorArray[:,0].argsort()] #Sort in ascending order
	adder.fill(-1)
	i = 0
	while(i<M and errorArray[i,0] <= -tol):
		adder[i] = errorArray[i,1]
		i += 1
	if(adder[0] == -1):
		return 1
	else:
		return 0

def main():
	global used
	global chunkSize
	global solution
	global present
	global e
	global L
	H = np.zeros(shape=(chunkSize,chunkSize))
	i = 0
	while (i < chunkSize):
		j = 0
		used[i] = i
		present[i] = i
		while (j < chunkSize):
			H[i,j] = target[i]*target[j]*dotProductP1P2(ind[i],point[i],ind[j],point[j])
			j += 1
		i += 1
	P = matrix(H)
	q = matrix(-e)
	G = matrix(J)
	h = matrix(n)
	A = matrix(target[0:chunkSize]).T
	sol = solvers.qp(P,q,G,h,A,d)
	solution = np.array(sol['x'])[:,0]
	L = sol['primal objective']
	# correct_solution()
	stoppingCriteria = check_Criteria();
	newCriteria = stop_Criteria3() # TODO
	if (stoppingCriteria == 1 and newCriteria == 0):
		print 'First Criteria stopped but new didnt'
	elif (stoppingCriteria == 0 and newCriteria == 1):
		print 'New Criteria stopped but First didnt'
	elif (stoppingCriteria == 1 and newCriteria == 1):
		print 'Both criteria stopped'
	if (newCriteria == 1):
		stoppingCriteria = 1
	while(stoppingCriteria != 1):
		i = 0
		j = 0
		cs = chunkSize-1
		uCS = chunkSize 	#to update chunkSize
		while(i < length):
			if(i < chunkSize and solution[i] == 0):
				if(j < M and adder[j] != -1):
					used[i] = adder[j]
					present[adder[j]] = i
					j += 1
				else:
					while(cs >= i and solution[cs] == 0):
						present[used[cs]] = -1
						used[cs] = -1
						cs -= 1
					if(cs > i):
						present[used[i]] = -1
						used[i] = used[cs]
						used[cs] = -1
						solution[cs] = 0
						cs -= 1
					else:
						uCS = i
						break
			elif(i >= chunkSize):
				if (j < M and adder[j] != -1):
					used[i] = adder[j]
					present[adder[j]] = i
					uCS += 1
					j += 1
				else:
					break
			i += 1
		chunkSize = uCS
		H = np.zeros(shape=(chunkSize,chunkSize))
		i = 0
		A = np.zeros(chunkSize)
		while (i < chunkSize):
			j = 0
			A[i] = target[used[i]]
			while (j < chunkSize):
				H[i,j] = target[used[i]]*target[used[j]]*dotProductP1P2(ind[used[i]],point[used[i]],ind[used[j]],point[used[j]])
				j += 1
			i += 1
		P = matrix(H)
		e = np.ones(chunkSize)
		q = matrix(-e)
		G = matrix(np.concatenate((-np.identity(chunkSize),np.identity(chunkSize)),axis = 0))
		h = matrix(np.concatenate((np.zeros(chunkSize),C*e),axis = 0))
		A = matrix(A).T
		sol = solvers.qp(P,q,G,h,A,d)
		solution = np.array(sol['x'])[:,0]
		L = sol['primal objective']
		# correct_solution()
		stoppingCriteria = check_Criteria();
		newCriteria = stop_Criteria3() # TODO
		if (stoppingCriteria == 1 and newCriteria == 0):
			print 'First Criteria stopped but new didnt'
		elif (stoppingCriteria == 0 and newCriteria == 1):
			print 'New Criteria stopped but First didnt'
		elif (stoppingCriteria == 1 and newCriteria == 1):
			print 'Both criteria stopped'
		if (newCriteria == 1):
			stoppingCriteria = 1
	accuracyTraining()						
	accuracyTest()

main()
print datetime.now() - startTime