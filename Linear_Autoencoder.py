import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import theano
from theano import tensor as T
from collections import OrderedDict
import time
from sklearn import decomposition


s=[1,0.7,0.2] #GAUSSIAN standard_deviation
# CREATING THE TRAINING DATA
x_cpu=np.hstack((np.random.normal(size=(1000,1),scale=s[0]),np.random.normal(size=(1000,1),scale=s[1]),np.random.normal(size=(1000,1),scale=s[2])))
# f1=plt.figure(1)
# plt.scatter(x_cpu[:,2],x_cpu[:,0])
# plt.show()
train_x=theano.shared(value=x_cpu) 

n_vis=3
n_hid=2


# AUTOENCODER PARAMETERS
W=theano.shared(value=np.random.random(size=(n_hid,n_vis))-0.5)
#SET b_v as the mean of the data
b_v=theano.shared(value=np.atleast_2d(np.mean(x_cpu,axis=0)).T,broadcastable=[False,True])

# MODEL
x_row=T.matrix('x_row')
h=T.dot(W,x_row.T)
#dummy1=theano.function([x_row],h)(train_x.get_value())
x_col_r=T.dot(W.T,h)+b_v
err=T.sum((x_row-x_col_r.T)**2,axis=0)

cost=T.mean(err)
cost_grad=T.grad(cost, W) #GRADIENT OF THE COST w.r.t. W

################ LEARNING RATE
################ [Becker 89] FROM http://infoscience.epfl.ch/record/82307/files/95-04.pdf?version=1


BECKER = True
fixed_lr=0.1
new_cost = cost_grad.sum() # TRICK https://groups.google.com/forum/#!topic/theano-users/tTdniwxjqT4
H_tmp = T.grad(new_cost, W)
H_diag=T.diag(H_tmp.flatten())
H_diag_norm=H_diag.norm(2)

h_norm = theano.function([x_row], H_diag_norm)
mu=0.01


################ Function to make the weights unit norm
unit_norm_rows = lambda rows:T.dot(
	T.cast(T.diag(1/rows.norm(2,axis=1)),'floatX'),rows
	)

updates=OrderedDict({})
updates[W]=unit_norm_rows(W-1*(T.grad(cost,W)/(mu+H_diag_norm))) if BECKER==True else unit_norm_rows(W-fixed_lr*T.grad(cost,W))

n_epochs=200

fig=plt.figure(1)
fig.clf()
ax=Axes3D(fig)	
plt.ion()
plt.show()

######## CALCULATING THE EIGENVECTORS FOR THE DATA
pca=decomposition.PCA()
pca.fit(x_cpu)
eig=pca.components_
e_val=pca.explained_variance_ratio_
to_plot_pca=np.zeros((n_vis*2,n_vis))
for i in range(n_vis):
	to_plot_pca[2*i,:]=eig[:,i]*np.sqrt(e_val[i]) ### array to hold the eigenvectors

############ TRAINING
for epoch in range(n_epochs):
	print epoch
	ax.clear()

	c,e=theano.function([x_row],[cost,err],updates=updates)(train_x.get_value())
	W_cpu=W.get_value()
	to_plot=np.zeros((n_hid*2,n_vis))
	for i in range(n_hid):
		to_plot[2*i,:]=W_cpu[i]

	###### PCA EIGENVECTORS IN GREEN
	ax.plot(to_plot_pca[:,0],to_plot_pca[:,1],to_plot_pca[:,2],'g') 

	###### AUTOENCODER WEIGHTS IN BLUE
	ax.plot(to_plot[:,0],to_plot[:,1],to_plot[:,2])
	###### ORIGIN IN RED
	ax.plot([0],[0],[0],'r.')

	ax.set_xlim3d([-1,1])
	ax.set_ylim3d([-1,1])
	ax.set_zlim3d([-1,1])	
	plt.draw()
	time.sleep(0.05)
	print 'cost %f, error %f' %(c,np.sqrt(np.dot(e,e)))

print 'done'
